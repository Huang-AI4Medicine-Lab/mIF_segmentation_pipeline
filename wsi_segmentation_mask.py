import numpy as np
import os
import tifffile
from skimage import segmentation, measure, morphology
from tqdm import tqdm
from region_utils import remove_duplicate_centroids
from segmentation_utils import labeled_mask_splitting, check_region_borders, filter_region_mask_artifacts
from utils import extract_row_col
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample_name', help='name of sample you are working with')
parser.add_argument('--intensity_im_path', help='path to multiplex image')
parser.add_argument('--mask_dir', help='directory that holds the segmentation masks that you want to analyze')
parser.add_argument('--overlap_method', default='centroid', help='method for combining the region masks')
parser.add_argument('--centroid_radius', type=int, default=10)
parser.add_argument('--row_region_size', type=int, default=1000)
parser.add_argument('--col_region_size', type=int, default=1300)
parser.add_argument('--region_overlap', type=float, default=0.20)
parser.add_argument('--artifact_filter_mode', dest='artifact_filter_mode', choices=['none', 'border', 'morphology', 'both'], default='border', help='artifact filtering mode for region masks')
parser.add_argument('--border_mode', dest='artifact_filter_mode', choices=['none', 'border', 'morphology', 'both'], help='deprecated alias for --artifact_filter_mode')
parser.add_argument('--flatness_threshold', type=float, default=0.85)
parser.add_argument('--min_solidity', type=float, default=0.80)
parser.add_argument('--min_artifact_area', type=int, default=20)
parser.add_argument('--border_width', type=int, default=1)
parser.add_argument('--dilate_mask', action='store_true', help='flag that allows you to perform a mask dilation')
parser.add_argument('--dilation_size', type=int, default=5)

args = parser.parse_args()
sample_name = args.sample_name
intensity_im_path = args.intensity_im_path
mask_dir = args.mask_dir
overlap_method = args.overlap_method
centroid_radius = args.centroid_radius
row_region_size = args.row_region_size
col_region_size = args.col_region_size
region_overlap = args.region_overlap
artifact_filter_mode = args.artifact_filter_mode
flatness_threshold = args.flatness_threshold
min_solidity = args.min_solidity
min_artifact_area = args.min_artifact_area
border_width = args.border_width
dilate_mask = args.dilate_mask
dilation_size = args.dilation_size

# mask_dir = f'/ix/yufeihuang/Hugh/melanoma_19_047/{panel_name}_seg_results/nuclei_segmentation_masks'
# overlap_method = 'centroid'
# centroid_radius = 10
# save_wsi_mask_dir = f'/ix/yufeihuang/Hugh/melanoma_19_047/wsi_seg_masks/{panel_name}'
# row_region_size, col_region_size = (1000, 1300)
# region_overlap = 0.2


print(f'Working on sample: {sample_name}')

intensity_im = tifffile.imread(intensity_im_path)

mask_fnames = os.listdir(mask_dir)
image_mask_fnames = [mask_fname for mask_fname in mask_fnames if sample_name in mask_fname]
row_border_size = int(row_region_size*region_overlap)
col_border_size = int(col_region_size*region_overlap)
template_image = np.zeros((intensity_im.shape[1], intensity_im.shape[2]), dtype=np.uint8)

del intensity_im

print(f'Creating wsi image mask...')
for region_mask_fname in tqdm(image_mask_fnames):

    roi_name = region_mask_fname.split('_seg')[0]
    row_val, col_val = extract_row_col(region_mask_fname)

    template_slice = template_image[row_val:row_val+row_region_size, col_val:col_val+col_region_size]

    region_mask = tifffile.imread(os.path.join(mask_dir, region_mask_fname))

    labeled_mask = measure.label(region_mask, connectivity=1)

    # --- Apply border artifact removal ---
    labeled_mask = filter_region_mask_artifacts(
        labeled_mask, 
        mode=artifact_filter_mode,
        extent_thresh=flatness_threshold,
        min_solidity=min_solidity,
        min_area=min_artifact_area,
        border_width=border_width
    )

    if overlap_method=='image_boundary':
        # preemptively drop any cells touching the boundary of the image        
        labeled_mask = segmentation.clear_border(labeled_mask)    

        border_cells = check_region_borders(
            template_slice, 
            labeled_mask, 
            row_border_size, 
            col_border_size,
            row_region_size,
            col_region_size
        )

        if len(border_cells)<1 and np.sum(template_slice)>1:
            print(f'Region: {roi_name} has a clear border yet also cells on the inside')
            continue

        if len(border_cells)<1:
            # assign the labeled mask instead of a binary mask - TODO clean this up or make an option for combining the mask in loop or after loop
            template_image[row_val:row_val+row_region_size, col_val:col_val+col_region_size] += (labeled_mask>0).astype(np.uint8)

        elif len(border_cells)>1:
            filter_mask = ~np.isin(labeled_mask, border_cells)
            filtered_labeled_mask = labeled_mask*filter_mask
            bin_labeled_mask = (filtered_labeled_mask>0).astype(np.uint8)
            # assign the labeled mask instead of the binary mask
            template_image[row_val:row_val+row_region_size, col_val:col_val+col_region_size] += bin_labeled_mask


    elif overlap_method=='centroid':
        # get centroids from the template image
        labeled_template_slice = measure.label(template_slice, connectivity=1)
        # if there is no template slice to compare to then continue
        if np.max(labeled_template_slice)<1:
            bin_labeled_mask = (labeled_mask>0).astype(np.uint8)
            template_image[row_val:row_val+row_region_size, col_val:col_val+col_region_size] += bin_labeled_mask

        # if we are adding a blank region then just skip it
        elif np.max(labeled_mask)<1:
            continue

        else:
            # get information for existing cells
            init_cell_info = measure.regionprops(labeled_template_slice)
            init_ids = np.array([region_dict.label for region_dict in init_cell_info])
            init_centroids = np.array([region_dict.centroid for region_dict in init_cell_info])

            initial_max_id = np.max(init_ids)

            # get information for the cells we wish to add
            add_cell_info = measure.regionprops(labeled_mask)
            add_ids = np.array([region_dict.label+initial_max_id for region_dict in add_cell_info])
            add_centroids = np.array([region_dict.centroid for region_dict in add_cell_info])

            combined_ids = np.concatenate([init_ids, add_ids])
            combined_centroids = np.concatenate([init_centroids, add_centroids], axis=0)

            id_array_mask = remove_duplicate_centroids(combined_centroids, radius=centroid_radius)

            # increment the ids in the image we are assing
            incremented_labeled_mask = np.copy(labeled_mask)
            incremented_labeled_mask[incremented_labeled_mask>0] += initial_max_id

            combined_ids_to_keep = combined_ids[id_array_mask]

            # take both images, sum them and then combine
            labeled_template_slice[~np.isin(labeled_template_slice, combined_ids_to_keep)] = 0
            incremented_labeled_mask[~np.isin(incremented_labeled_mask, combined_ids_to_keep)] = 0

            # combine the mask by summing - throw an error if we accidentally create a new id
            # binarize the masks and then take the union and see what we get
            bin_combined_mask = np.logical_or((labeled_template_slice>0), (incremented_labeled_mask>0))
            combined_labeled_mask = measure.label(bin_combined_mask, connectivity=1)
            # combined_labeled_mask = labeled_template_slice+incremented_labeled_mask
            final_ids = np.unique(combined_labeled_mask)
            final_ids = final_ids[final_ids!=0]
            # here perform the check
            # if np.any(~np.isin(final_ids, combined_ids_to_keep)):
            #     raise Exception('We have overlapping region ids that are creating new cell labels, please correct...')
            # perform mask splitting after the check
            split_labeled_mask = labeled_mask_splitting(combined_labeled_mask, kernel_size=1)
            bin_labeled_mask = (combined_labeled_mask>0).astype(np.uint8)
            # clear the boundary - this bit is essential
            bin_labeled_mask = segmentation.clear_border(bin_labeled_mask) 
            # this is for assinging back to the original image we can keep this as-is
            template_image[row_val:row_val+row_region_size, col_val:col_val+col_region_size] += bin_labeled_mask


labeled_mask_im = measure.label(template_image, connectivity=1).astype(np.uint32)
del template_image
# boundary_image = segmentation.find_boundaries(labeled_mask_im, connectivity=1, mode='inner')
mask_regions = measure.regionprops(labeled_mask_im)

print(f'Max cell id from the image: {np.max(labeled_mask_im)}')
# optionally dilate the final mask
if dilate_mask:
    labeled_mask_im = morphology.dilation(labeled_mask_im, footprint=np.ones((dilation_size, dilation_size)))
    print(f'Max cell id from image - after dilation: {np.max(labeled_mask_im)}')

# save the labeled image - clear memory
tifffile.imwrite(f'{sample_name}_wsi_mask.ome.tif', labeled_mask_im)


