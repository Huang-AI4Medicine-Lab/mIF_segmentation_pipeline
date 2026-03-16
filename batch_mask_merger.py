import tifffile
import numpy as np
import os
from segmentation_utils import *
import argparse

# following code merges all cell masks for a given sample
parser = argparse.ArgumentParser()
parser.add_argument('--mask_root_dir', default=None, help='root directory holding all mask folders')
parser.add_argument('--mask_dirs', default=None, help='comma separated list of mask directories')
parser.add_argument('--im_selection_dir', default=None)
parser.add_argument('--exclude_nuclei', action='store_true')
parser.add_argument('--sample_name')
parser.add_argument('--method', default='centroid', choices=['centroid', 'ioq'])
parser.add_argument('--ioq_thresh', default=0.5)
parser.add_argument('--save_dir')

args = parser.parse_args()
mask_root_dir = args.mask_root_dir
mask_dirs = args.mask_dirs
sample_name = args.sample_name
exclude_nuclei = args.exclude_nuclei
im_selection_dir = args.im_selection_dir
method = args.method
ioq_thresh = float(args.ioq_thresh)
save_dir = args.save_dir

print(f'Selected Method: {method}')
print(f'Selected Score Threshold: {ioq_thresh}')


# Get list of mask directories
if args.mask_dirs is not None:
    # Option 2: Comma-separated list of directories
    mask_directories = [d.strip() for d in args.mask_dirs.split(',') if d.strip()]
elif args.mask_root_dir is not None:
    # Option 1: Search subdirectories of root
    mask_directories = []
    for folder in os.listdir(args.mask_root_dir):
        folder_path = os.path.join(args.mask_root_dir, folder)
        if os.path.isdir(folder_path):
            mask_directories.append(folder_path)
else:
    raise ValueError("Must specify either --mask_root_dir or --mask_dirs")

# Filter directories
valid_directories = []
for dir_path in mask_directories:
    dir_name = os.path.basename(dir_path)
    if 'merged' in dir_name:
        continue
    if exclude_nuclei and 'nuclei' in dir_name:
        continue
    if os.path.isdir(dir_path):
        valid_directories.append(dir_path)

print(f'Processing directories: {[os.path.basename(d) for d in valid_directories]}')

if len(valid_directories) == 0:
    raise ValueError("No valid mask directories found")

# Get list of image files to process from the first valid directory
if im_selection_dir is None:
    image_files = os.listdir(valid_directories[0])
    image_files = [f for f in image_files if sample_name in f and f.endswith('.tiff')]
else:
    with open(os.path.join(im_selection_dir, f'{sample_name}_roi_paths.txt'), 'r') as f:
        image_files = f.read().split("\n")
        image_files = [f for f in image_files if f.strip()]

print(f'Found {len(image_files)} images to process')

for image_fname in image_files:
    # load all of the binary masks
    mask_ims = []
    for mask_dir in valid_directories:
        mask_path = os.path.join(mask_dir, image_fname)
        
        if os.path.exists(mask_path):
            mask_ims.append(tifffile.imread(mask_path))
        else:
            print(f'Warning: {mask_path} not found, skipping')

    if len(mask_ims) == 0:
        print(f'No masks found for {image_fname}, skipping')
        continue

    template_size = mask_ims[0].shape
    initial_template = np.zeros(template_size)
    for count, mask_im in enumerate(mask_ims):
        new_template = merge_mask_centroid(initial_template, mask_im, method=method, ioq_thresh=ioq_thresh)
        initial_template = new_template

    # save the new segmentation image
    os.makedirs(save_dir, exist_ok=True)

    tifffile.imwrite(os.path.join(save_dir, image_fname), initial_template)

    # if method=='centroid':
    #     # tifffile.imwrite(os.path.join(save_dir, f'{sample_name}_component_data_seg_centroid.tiff'), initial_template)
    #     # tifffile.imwrite(os.path.join(save_dir, f'{sample_name}_seg_centroid.tiff'), initial_template)
    # else:
    #     # tifffile.imwrite(os.path.join(save_dir, f'{sample_name}_component_data_seg_ioq_{ioq_thresh}.tiff'), initial_template)
    #     tifffile.imwrite(os.path.join(save_dir, f'{sample_name}_seg_ioq_{ioq_thresh}.tiff'), initial_template)



