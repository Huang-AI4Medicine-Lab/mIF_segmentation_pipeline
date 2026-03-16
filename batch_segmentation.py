import tifffile
import numpy as np
import PIL
from segmentation_utils import labeled_mask_splitting, tile_image, combine_tiles
from cellpose_segmentation_utils import cellpose_segmentation, custom_cellpose_segmentation
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir')
parser.add_argument('--sample_name')
parser.add_argument('--diameter', default=30)
parser.add_argument('--marker_name', default='nuclei')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--channel_type', required=True, choices=['single_channel', 'dual_channel'])
parser.add_argument('--tile', action='store_true')
parser.add_argument('--channel_axis', default=0)
parser.add_argument('--model_name', default='nuclei')
parser.add_argument('--cellpose_normalize', action='store_true')
parser.add_argument('--im_selection_dir', default=None)
parser.add_argument('--channel_selection_path', default=None)
parser.add_argument('--custom_model_path', default=None)
args = parser.parse_args()

image_dir = args.image_dir
sample_name = args.sample_name
diameter = int(args.diameter)
gpu_status = args.gpu
channel_type = args.channel_type
tile = args.tile
channel_axis = args.channel_axis
channel_selection_path = args.channel_selection_path
model_name = args.model_name
cellpose_normalize = args.cellpose_normalize
im_selection_dir = args.im_selection_dir
custom_model_path = args.custom_model_path

save_folder = f"{args.marker_name}_seg_ims"

if im_selection_dir is None:
    image_files = os.listdir(image_dir)
    image_files = [image_file for image_file in image_files if sample_name in image_file]
else:
    with open(os.path.join(im_selection_dir, f'{sample_name}_roi_paths.txt'), 'r') as f:
        image_files = f.read().split("\n")[:-1]

if diameter==-1:
    diameter=None

# Base directory paths (update these paths if needed)
# input_base_path = "D://FMT_MIF//roi_subset_copy//"
# channel_selection_path = "D://FMT_MIF//cellpose_training_samples//CD8_samples//cd8_channels.npy"
# custom_model_path = "D://FMT_MIF//cellpose_training_samples//CD8_samples//cellpose_ready_CD8//models//dapi_cd8_e_1000_lr_0_01"
# save_folder = "D://FMT_MIF//roi_subset_segmentation//cd8_segmentation_masks"

# Loop through each image file and run the command
for image_file in tqdm(image_files):
    input_image_path = f'{image_dir}//{image_file}'

    input_im = tifffile.imread(input_image_path)

    os.makedirs(save_folder, exist_ok=True)

    if channel_selection_path is not None:
        channel_subset = np.load(channel_selection_path)
        input_im = input_im[channel_subset,:,:]

    if channel_type=='single_channel':
        channels = [0,0]

        # subset the image file before passing it into the model
        if channel_axis==0:
            input_im = input_im[0,:,:]
            input_im = np.expand_dims(input_im, 0)
        else:
            input_im = input_im[:,:,0]
            input_im = np.expand_dims(input_im, -1)

        print(f'Running single channel segmentation...')
        if not tile:
            if custom_model_path is None:
                label_im = cellpose_segmentation(input_im, gpu=gpu_status, diam=diameter, model_name=model_name, channel_cellpose=channels, normalize=cellpose_normalize)
            else:
                label_im = custom_cellpose_segmentation(input_im, model_path=custom_model_path, diam=diameter, gpu=gpu_status, channel_cellpose=channels, normalize=cellpose_normalize)
        else:
            input_arr = np.squeeze(input_im)
            if input_arr.ndim<3:
                if channel_axis==0:
                    input_arr = input_arr[np.newaxis,:,:]
                else:
                    input_arr = input_arr[:,:,np.newaxis]
            
            tile_list = tile_image(input_arr, channel_axis=channel_axis)
            labeled_im_list = []
            for tile in tile_list:
                if custom_model_path is None:
                    label_im = cellpose_segmentation(tile, gpu=gpu_status, diam=diameter, model_name=model_name, channel_cellpose=channels, normalize=cellpose_normalize)
                else:
                    label_im = custom_cellpose_segmentation(tile, model_path=custom_model_path, diam=diameter, gpu=gpu_status, channel_cellpose=channels, normalize=cellpose_normalize)

                label_im = labeled_mask_splitting(np.squeeze(label_im))
                label_im = (label_im>0).astype(np.uint8)
                labeled_im_list.append(label_im)


    elif channel_type == 'dual_channel':
        channels = [1,2]
        print('Starting dual channel segmentation')
        if not tile:
            if custom_model_path is None:
                label_im = cellpose_segmentation(input_im, model_name=model_name, diam=diameter, gpu=gpu_status, channel_cellpose=channels, normalize=cellpose_normalize)

            else:
                label_im = custom_cellpose_segmentation(input_im, model_path=custom_model_path, diam=diameter, gpu=gpu_status, channel_cellpose=channels, normalize=cellpose_normalize)

        else:
            input_arr = np.squeeze(input_im)
            if input_arr.ndim<3:
                if channel_axis==0:
                    input_arr = input_arr[np.newaxis,:,:]
                else:
                    input_arr = input_arr[:,:,np.newaxis]

            tile_list = tile_image(input_arr, channel_axis=channel_axis)
            labeled_im_list = []
            for tile in tile_list:
                if custom_model_path is None:
                    label_im = cellpose_segmentation(input_im, model_name=model_name, diam=diameter, gpu=gpu_status, channel_cellpose=channels, normalize=cellpose_normalize)
                else:
                    label_im = custom_cellpose_segmentation(input_im, model_path=custom_model_path, diam=diameter, gpu=gpu_status, channel_cellpose=channels, normalize=cellpose_normalize)

                label_im = labeled_mask_splitting(np.squeeze(label_im))
                label_im = (label_im>0).astype(np.uint8)
                labeled_im_list.append(label_im)


    # get number of cells identified during segmentation
    print(f'Total Number of Cells Segmented: {label_im.max()}')

    if not tile:
        res = np.squeeze(label_im)
    else:
        if channel_axis==0:
            row_size, col_size = input_im.shape[1], input_im.shape[2]
        else:
            row_size, col_size = input_im.shape[0], input_im.shape[1]
        res = combine_tiles(labeled_im_list, row_size, col_size)

    segmented_arr = np.squeeze(res)
    print('Splitting labeled image and creating binary mask...')
    segmented_arr = labeled_mask_splitting(segmented_arr)
    binarized_image = (segmented_arr>0).astype(np.uint8)

    # save the binarized tiff image
    save_fname = image_file.split('.')[0]
    pil_im_segm = PIL.Image.fromarray(binarized_image)
    pil_im_segm.save(os.path.join(save_folder, f'{save_fname}_seg.tiff'))
    print('Saved binary mask')



