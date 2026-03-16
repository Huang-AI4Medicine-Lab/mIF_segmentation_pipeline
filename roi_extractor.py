import tifffile
import numpy as np
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
from region_utils import *
from autofluorescence_utils import apply_saved_global_subtraction
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path')
parser.add_argument('--existing_mask_fname', default=None)
parser.add_argument('--sample_name')
parser.add_argument('--roi_row_size', default=1000)
parser.add_argument('--roi_col_size', default=1300)
parser.add_argument('--roi_foreground_ratio', default=0.01)
parser.add_argument('--downsample_factor', default=8)
parser.add_argument('--save_regions', action='store_true')
parser.add_argument('--remove_autofluorescence', action='store_true')
parser.add_argument('--autofluorescence_params', default=None)
parser.add_argument('--use_predefined_regions', action='store_true')
parser.add_argument('--predefined_region_path')
parser.add_argument('--af_channel', default=7)
parser.add_argument('--overlap', default=0.20)
parser.add_argument('--num_thresholds', default=2)
parser.add_argument('--thresh_mode', default='min')
parser.add_argument('--channel_reduce', default='max')
parser.add_argument('--single_channel', action='store_true')


args = parser.parse_args()

image_path = args.image_path
existing_mask_fname = args.existing_mask_fname
sample_name = args.sample_name
roi_row_size = int(args.roi_row_size)
roi_col_size = int(args.roi_col_size)
roi_foreground_ratio = float(args.roi_foreground_ratio)
downsample_factor = int(args.downsample_factor)
remove_autofluorescence = args.remove_autofluorescence
autofluorescence_params = args.autofluorescence_params
use_predefined_regions = args.use_predefined_regions
predefined_region_path = args.predefined_region_path
af_channel = int(args.af_channel)
prc_overlap = float(args.overlap)
num_thresholds = int(args.num_thresholds)
thresh_mode = args.thresh_mode
channel_reduce = args.channel_reduce
single_channel = args.single_channel

src_im = tifffile.imread(image_path)

save_folder="region_images"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)


# here process the mif image and remove autofluorescence
if remove_autofluorescence:
    print('Removing autofluorescence')
    src_im = apply_saved_global_subtraction(src_im, autofluorescence_params, af_channel=af_channel)
    

# load region ids
if use_predefined_regions:
    region_id_arr = np.load(predefined_region_path)
    
# grab tissue mask image here
if existing_mask_fname is not None:
    print('Loading existing mask...')
    tissue_mask = tifffile.imread(existing_mask_fname)
else:
    print('Segmenting Foreground...')

    tissue_mask = extract_tissue_mask(
        src_im, 
        reduce_method=channel_reduce,
        num_thresholds=num_thresholds, 
        thresh_mode=thresh_mode
    )

print(f'Source Image Shape: {src_im.shape}')
print(f'Tissue Mask Shape: {tissue_mask.shape}')

overlap_row_size = roi_row_size-(prc_overlap*roi_row_size)
overlap_col_size = roi_col_size-(prc_overlap*roi_col_size)

if not single_channel:
    wsi_row_size = src_im.shape[1]
    wsi_col_size = src_im.shape[2]
else:
    wsi_row_size = src_im.shape[0]
    wsi_col_size = src_im.shape[1]

row_end = wsi_row_size//overlap_row_size+1
col_end = wsi_col_size//overlap_col_size+1

print(f'Row Extend: {row_end}')
print(f'Col Extend: {col_end}')

print(f'Wsi Row Size: {wsi_row_size} - Wsi Col Size: {wsi_col_size}')

print(f'Total Number of rois: {row_end*col_end}')

# instead of saving a boundary image, instead save a painted in version of the boundary image
region_mask_image = np.zeros((wsi_row_size, wsi_col_size), dtype=np.uint8)

num_selected_regions = 0
overall_roi_id = 0


print(f'Creating ROI template image')
for row_index in tqdm(range(int(row_end))):
    for col_index in range(int(col_end)):
        overall_roi_id += 1
        new_row_index = row_index*int(overlap_row_size)
        new_col_index = col_index*int(overlap_col_size)

        # check if we exceed our boundaries, if so push them in and resample
        if new_row_index+roi_row_size>=wsi_row_size:
            new_row_index = wsi_row_size-roi_row_size-1
        if new_col_index+roi_col_size>=wsi_col_size:
            new_col_index = wsi_col_size-roi_col_size-1
        
        # if using a predefined list then check here
        if use_predefined_regions:
            if np.any(region_id_arr==overall_roi_id):
                if not single_channel:
                    region_im = src_im[:, new_row_index:new_row_index+roi_row_size, new_col_index:new_col_index+roi_col_size]
                else:
                    region_im = src_im[new_row_index:new_row_index+roi_row_size, new_col_index:new_col_index+roi_col_size]
                    
                region_mask_image[new_row_index:new_row_index+roi_row_size, new_col_index:new_col_index+roi_col_size] = 1

                if args.save_regions:
                    tifffile.imwrite(os.path.join(save_folder, f'{sample_name}_roi_{overall_roi_id}_row_{new_row_index}_col_{new_col_index}.ome.tif'), region_im)

            continue
                
        tissue_mask_subset = tissue_mask[new_row_index:new_row_index+roi_row_size, new_col_index:new_col_index+roi_col_size]
        if np.sum(tissue_mask_subset)/(roi_col_size*roi_col_size)<roi_foreground_ratio:
            continue

        num_selected_regions += 1
            
        # paint in the other template image
        region_mask_image[new_row_index:new_row_index+roi_row_size, new_col_index:new_col_index+roi_col_size] = 1

        # if the image is valid then save it
        if not single_channel:
            region_im = src_im[:, new_row_index:new_row_index+roi_row_size, new_col_index:new_col_index+roi_col_size]
        else:
            region_im = src_im[new_row_index:new_row_index+roi_row_size, new_col_index:new_col_index+roi_col_size]
        
        if args.save_regions:
            tifffile.imwrite(os.path.join(save_folder, f'{sample_name}_roi_{overall_roi_id}_row_{new_row_index}_col_{new_col_index}.ome.tif'), region_im)



# tifffile.imwrite(os.path.join(boundary_image_path, f'{sample_name}_roi_mask_im.tif'), region_mask_image)
tifffile.imwrite(f'{sample_name}_roi_mask_im.tif', region_mask_image)

# also make a downsampled version of the painted in image and save it
collapsed_src = np.squeeze(np.max(src_im, axis=0))
src_small = collapsed_src[::downsample_factor,::downsample_factor]
mask_small = region_mask_image[::downsample_factor, ::downsample_factor]

fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(6,4))
axes[0].imshow(src_small, cmap='bone')
axes[1].imshow(src_small, cmap='bone')
axes[1].imshow(mask_small, cmap='jet', alpha=0.35)
plt.title(f'{sample_name} tissue mask')

# fig.savefig(os.path.join(boundary_image_path, f'{sample_name}_region_mask_overlay.jpg'), bbox_inches='tight', dpi=300)
fig.savefig(f'{sample_name}_region_mask_overlay.jpg', bbox_inches='tight', dpi=300)

