import tifffile
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import zarr
from region_utils import extract_tissue_mask

parser = argparse.ArgumentParser()
parser.add_argument('--image_path')
parser.add_argument('--sample_name')
parser.add_argument('--channel_reduce', default='max')
parser.add_argument('--downsample_factor', default=8)
parser.add_argument('--num_thresholds', default=2)
parser.add_argument('--thresh_mode', default='min')
parser.add_argument('--lazy_loading', action='store_true')
parser.add_argument('--tissue_channel', default=0)

args = parser.parse_args()

image_path = args.image_path
sample_name = args.sample_name
downsample_factor = args.downsample_factor
channel_reduce = args.channel_reduce
num_thresholds = int(args.num_thresholds)
thresh_mode = args.thresh_mode
lazy_loading = args.lazy_loading
tissue_channel = int(args.tissue_channel)

print(f'Loading multiplex image...')
# if performing lazy loading only load the DAPI channel (channel 0)
if not lazy_loading:
    src_im = tifffile.imread(image_path)
else:
    tif_obj = tifffile.TiffFile(image_path)
    im_store = tif_obj.aszarr(level=0)
    sample_zarr = zarr.open(im_store, mode='r')
    src_im = sample_zarr[tissue_channel]
    # restore the channel axis
    src_im = src_im[np.newaxis,:,:]


print(f'Segmenting foreground...')
tissue_mask = extract_tissue_mask(
    src_im,
    reduce_method=channel_reduce,
    num_thresholds=num_thresholds,
    thresh_mode=thresh_mode
)

tissue_mask = (tissue_mask>0).astype(np.uint8)
# tifffile.imwrite(os.path.join(save_dir, f'{sample_name}_tissue_mask.ome.tif'), tissue_mask)
# tifffile.imwrite(f'{sample_name}.ome.tif', tissue_mask)
tifffile.imwrite(f'{sample_name}_tissue_mask.ome.tif', tissue_mask)


# also create a visualization of the segmentation itself
src_im = np.squeeze(np.max(src_im, axis=0))
src_small = src_im[::downsample_factor,::downsample_factor]
mask_small = tissue_mask[::downsample_factor,::downsample_factor]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
axes[0].imshow(src_small, cmap='bone')
axes[1].imshow(src_small, cmap='bone')
axes[1].imshow(mask_small, cmap='jet', alpha=0.35)
plt.title(f'{sample_name} tissue mask')

# fig.savefig(os.path.join(save_dir, f'{sample_name}_tissue_mask_overlay.jpg'), bbox_inches='tight', dpi=300)
fig.savefig(f'{sample_name}_tissue_mask_overlay.jpg', bbox_inches='tight', dpi=300)
