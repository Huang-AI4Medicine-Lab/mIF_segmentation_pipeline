import argparse
import numpy as np
import pandas as pd
import os
import tifffile
import json
from typing import Dict
from tqdm import tqdm
from autofluorescence_utils import *

def main():
    parser = argparse.ArgumentParser(description="AF attenuation for multichannel (C,H,W) qptiff using tifffile.")
    parser.add_argument("--stack_path", required=True, help="Path to multichannel qptiff (C,H,W).")
    parser.add_argument("--sample_name", required=True)
    parser.add_argument("--af_channel", type=int, required=True, help="Index of AF channel in the stack.")
    parser.add_argument("--tissue_mask_path", required=True, help="(H,W) mask TIFF; nonzero = tissue.")
    parser.add_argument("--out_path", required=True, help="Output multichannel BigTIFF path.")
    parser.add_argument("--param_out_path", required=True, help="Path to saved estimated alphas...")
    parser.add_argument("--channels_to_correct", default="all",
                    help='Comma list of channel indices to correct (e.g., "1,2,3"), or "all" to correct all except AF.')
    parser.add_argument("--tile", type=int, default=4096, help="Tile size (no overlap).")
    parser.add_argument('--include_beta', action='store_true')
    parser.add_argument('--write_channel_ims', action='store_true')
    parser.add_argument("--sample_method", choices=["random","grid"], default="random")
    parser.add_argument("--n_samples", type=int, default=2_000_000)
    parser.add_argument("--grid_step", type=int, default=512)
    parser.add_argument("--drop_marker_top_p", type=float, default=95.0)
    parser.add_argument("--drop_af_top_p", type=float, default=99.0)
    parser.add_argument("--smooth_field", action="store_true", help="Fit smooth polynomial alpha(x,y).")
    parser.add_argument("--length_scale_px", type=int, default=2048, help="Controls spatial smoothness scale (for sampling density only).")
    parser.add_argument("--alpha_max_factor", type=float, default=1.25, help="Clamp spatial alpha to <= factor*global_alpha.")
    parser.add_argument("--save_alpha_maps_dir", default=None, help="If set, saves per-channel alpha previews (downsampled).")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f'Loading image stack...')
    stack = tifffile.imread(args.stack_path)
    if stack.ndim != 3:
        raise ValueError(f"Expected (C,H,W) stack, got shape {stack.shape}")
    C, H, W = stack.shape
    dtype = stack.dtype
    if not np.issubdtype(dtype, np.integer):
        raise ValueError("Expected integer dtype (uint16/uint8) for fluorescence images.")
    
    print(f'Loading tissue mask...')
    # Mask
    tissue_mask = tifffile.imread(args.tissue_mask_path)
    tissue_mask = np.squeeze(tissue_mask)
    tissue_mask = tissue_mask.astype(bool)
    if tissue_mask.shape != (H, W):
        raise ValueError(f"Mask shape {tissue_mask.shape} must match (H,W)=({H},{W})")

    # Determine target channels
    if args.channels_to_correct.lower() == "all":
        targets = [i for i in range(C) if i != args.af_channel]
    else:
        targets = [int(x) for x in args.channels_to_correct.split(",") if x.strip()!=""]
        for t in targets:
            if t == args.af_channel:
                print(f"Note: AF channel {t} will not be corrected; skipping.")
        targets = [t for t in targets if t != args.af_channel]

    print(f'Correcting the following channel indices: {targets}')

    # Downsample factor for fitting
    ds = max(1, args.grid_step // 2) if args.sample_method == "grid" else 8
    A_small = stack[args.af_channel, ::ds, ::ds].astype(np.float32)
    T_small = tissue_mask[::ds, ::ds]
    Hs, Ws = A_small.shape

    # don't use an output writer
    print(f'Saving parameters to: {args.param_out_path}')
    
    if args.write_channel_ims:
        os.makedirs(args.out_path, exist_ok=True)
        
    os.makedirs(args.param_out_path, exist_ok=True)

    # global parameter list
    global_param_list = []

    # make a tile-wise parameter dict
    tilewise_param_list = []

    
    # for now we can write channels separately
    for c in range(C):
        global_params = {}

        if c==args.af_channel or c not in targets:
            if args.write_channel_ims:
                print(f"Writing channel {c} unchanged...")
                tifffile.imwrite(os.path.join(args.out_path, f'{args.sample_name}_{c}_cleaned.ome.tif'), np.squeeze(stack[c,:,:]))
            print(f'Skipping channel {c}...')
            
            continue

        print(f'Processing Channel {c}...')

        channel_im = np.copy(np.squeeze(stack[c,:,:]))
        af_channel_im = np.copy(np.squeeze(stack[args.af_channel,:,:]))
        # Global fit on downsampled
        M_small = stack[c, ::ds, ::ds].astype(np.float32)
        alpha0, beta0, n_used = fit_global_alpha_beta(
            M_small, A_small, T_small, rng,
            sample_method=args.sample_method, n_samples=args.n_samples,
            grid_step=max(16, args.grid_step//ds),
            drop_marker_top_p=args.drop_marker_top_p,
            drop_AF_top_p=args.drop_af_top_p
        )
        print(f"Global alpha0={alpha0:.5f}, beta0={beta0:.3f}, samples={n_used}")

        global_params['channel'] = c
        global_params['alpha'] = alpha0
        global_params['beta'] = beta0
        
        print(f'{c} - {alpha0} - {beta0}')

        global_param_list.append(global_params)

        poly_beta = None
        if args.smooth_field:
            print(f'Fitting tilewise alpha...')
            # Fit tilewise alphas on the downsampled arrays (coarse tiling)
            step = max(512//ds, 1) * ds  # coarse tile ~512px at native
            centers, alphas = [], []
            for (y0,y1,x0,x1) in tqdm(compute_tiles(H, W, step)):
                ys0, ys1 = y0//ds, y1//ds
                xs0, xs1 = x0//ds, x1//ds
                M_t = M_small[ys0:ys1, xs0:xs1]
                A_t = A_small[ys0:ys1, xs0:xs1]
                T_t = T_small[ys0:ys1, xs0:xs1]
                if T_t.sum() < 500:
                    continue
                try:
                    a_t, b_t, n_t = fit_global_alpha_beta(
                        M_t, A_t, T_t, rng,
                        sample_method=args.sample_method,
                        n_samples=min(200_000, args.n_samples//10),
                        grid_step=max(16, args.grid_step//ds),
                        drop_marker_top_p=args.drop_marker_top_p,
                        drop_AF_top_p=args.drop_af_top_p
                    )
                    centers.append(( (y0+y1)//2, (x0+x1)//2 ))
                    alphas.append(a_t)
                except Exception:
                    continue
            if len(alphas) >= 6:
                centers_np = np.array(centers, dtype=np.int32)
                alphas_np = np.array(alphas, dtype=np.float32)
                poly_beta = fit_poly_alpha_from_tile_alphas(H, W, centers_np, alphas_np, length_scale_px=args.length_scale_px)
                print("  Smooth alpha field: polynomial fit computed.")
            else:
                print("  Not enough points for smooth field; using constant alpha.")

        print(f'Assinging alpha values...')
        # Process tiles without overlap, evaluating alpha per tile (constant or polynomial field)
        for (y0,y1,x0,x1) in tqdm(compute_tiles(H, W, args.tile)):
            M_tile = stack[c, y0:y1, x0:x1]
            A_tile = stack[args.af_channel, y0:y1, x0:x1]
            if poly_beta is not None:
                alpha_tile = eval_poly_alpha(poly_beta, H, W, y0, y1, x0, x1)
                alpha_tile = np.clip(alpha_tile, 0.0, args.alpha_max_factor * alpha0)
            else:
                alpha_tile = np.full(M_tile.shape, alpha0, dtype=np.float32)

            # print(f'Tile Median: {np.median(alpha_tile)} - Tile Max: {np.max(alpha_tile)} - Tile Min: {np.min(alpha_tile)}')

            out_tile = apply_subtraction(M_tile, A_tile, alpha_tile)
            channel_im[y0:y1, x0:x1] = out_tile

            # record tilewise parameters
            if args.smooth_field:
                tilewise_params = {}
                tilewise_params['channel'] = c
                tilewise_params['row_start'] = y0
                tilewise_params['row_end'] = y1
                tilewise_params['col_start'] = x0
                tilewise_params['col_end'] = x1
                tilewise_params['alpha'] = np.max(alpha_tile)
                tilewise_param_list.append(tilewise_params)

        # if we are only doing global alpha then just subtract directly
        # else:
            # print(f'applying global alpha...')
            # if args.include_beta:
            #     print(f'applying global beta...')
            #     channel_im = channel_im-(alpha0*af_channel_im+beta0)
            # else:
            #     channel_im = channel_im-(alpha0*af_channel_im)
            # channel_im[channel_im<0] = 0.0
            # print(f'mean after correction: {np.mean(channel_im)}')
            # channel_im = channel_im.astype(np.uint8)
            # print(f'mean after correction uint8: {np.mean(channel_im)}')

        # Optionally save downsampled alpha preview
        if args.save_alpha_maps_dir is not None:
            os.makedirs(args.save_alpha_maps_dir, exist_ok=True)
            if poly_beta is not None:
                # quick low-res preview
                Hs2, Ws2 = max(1, H//2048), max(1, W//2048)
                ys = np.linspace(0, H-1, Hs2).astype(int)
                xs = np.linspace(0, W-1, Ws2).astype(int)
                Y, X = np.meshgrid(ys, xs, indexing='ij')
                y0, y1, x0, x1 = int(ys[0]), int(ys[-1])+1, int(xs[0]), int(xs[-1])+1
                alpha_low = eval_poly_alpha(poly_beta, H, W, 0, Hs2, 0, Ws2)
                alpha_low = np.clip(alpha_low, 0, args.alpha_max_factor * alpha0)
                alpha8 = (255 * (alpha_low / (alpha_low.max() + 1e-6))).astype(np.uint8)
                # iio.imwrite(os.path.join(args.save_alpha_maps_dir, f"alpha_channel{c}.png"), alpha8)
            else:
                # constant alpha preview image
                alpha8 = np.full((256,256), int(255*min(1.0, alpha0/(alpha0+1e-6))), dtype=np.uint8)
                # iio.imwrite(os.path.join(args.save_alpha_maps_dir, f"alpha_channel{c}.png"), alpha8)
    
        if args.write_channel_ims:
            print(f'Writing corrected channel...')
            # Write this corrected channel
            tifffile.imwrite(os.path.join(args.out_path, f'{args.sample_name}_{c}_cleaned.ome.tif'), channel_im)

        # write background estimation params
        print(f'Writing parameters...')
        global_frame = pd.DataFrame(global_param_list)
        global_frame.to_csv(os.path.join(args.param_out_path, f'{args.sample_name}_global_params.csv'), index=False)
        if args.smooth_field:
            tilewise_frame = pd.DataFrame(tilewise_param_list)
            tilewise_frame.to_csv(os.path.join(args.param_out_path, f'{args.sample_name}_local_params.csv'), index=False)

        print(f'Clearing space...')
        del channel_im
        del af_channel_im
        


if __name__ == "__main__":
    main()
