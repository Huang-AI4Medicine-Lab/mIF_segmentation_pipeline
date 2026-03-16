from typing import Tuple, Optional, Iterable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import nnls
from tqdm import tqdm

def compute_tiles(H: int, W: int, tile: int) -> Iterable[Tuple[int,int,int,int]]:
    for y0 in range(0, H, tile):
        y1 = min(H, y0 + tile)
        for x0 in range(0, W, tile):
            x1 = min(W, x0 + tile)
            yield (y0, y1, x0, x1)

def jittered_grid_coords(mask: NDArray, step: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
    H, W = mask.shape
    ys = np.arange(step//2, H, step)
    xs = np.arange(step//2, W, step)
    y_list, x_list = [], []
    for y in ys:
        for x in xs:
            dy = rng.integers(-step//4, step//4 + 1)
            dx = rng.integers(-step//4, step//4 + 1)
            yy = int(np.clip(y + dy, 0, H-1))
            xx = int(np.clip(x + dx, 0, W-1))
            if mask[yy, xx]:
                y_list.append(yy); x_list.append(xx)
    if not y_list:
        raise ValueError("No grid samples in mask. Use smaller step or random sampling.")
    return np.array(y_list, dtype=np.int64), np.array(x_list, dtype=np.int64)

def random_sample_coords(mask: NDArray, n_samples: int, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("Tissue mask contains no True pixels.")
    idx = rng.choice(ys.size, size=min(n_samples, ys.size), replace=False)
    return ys[idx], xs[idx]

def fit_global_alpha_beta(M: NDArray, A: NDArray, tissue_mask: NDArray,
                          rng: np.random.Generator,
                          sample_method: str,
                          n_samples: int,
                          grid_step: int,
                          drop_marker_top_p: float,
                          drop_AF_top_p: float) -> Tuple[float, float, int]:
    if sample_method == "random":
        ys, xs = random_sample_coords(tissue_mask, n_samples, rng)
    else:
        ys, xs = jittered_grid_coords(tissue_mask, grid_step, rng)
    Ms = M[ys, xs].astype(np.float64)
    As = A[ys, xs].astype(np.float64)
    m_thr = np.percentile(Ms, drop_marker_top_p)
    a_thr = np.percentile(As, drop_AF_top_p)
    keep = (Ms <= m_thr) & (As <= a_thr)
    Ms = Ms[keep]; As = As[keep]
    if Ms.size < 1000:
        raise ValueError("Too few samples after exclusions.")
    X = np.stack([As, np.ones_like(As)], axis=1)
    
    # print(f'AF Input Shape: {X.shape}')
    # print(f'Signal Input Shape: {Ms.shape}')

    coef, _ = nnls(X, Ms)

    # print(f'Selected Coefs: {coef}')

    return float(coef[0]), float(coef[1]), Ms.size

def poly2_design_matrix(x: NDArray, y: NDArray) -> NDArray:
    # Basis: [1, x, y, x^2, xy, y^2]
    return np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=-1)

def fit_poly_alpha_from_tile_alphas(H:int, W:int, centers: NDArray, alphas: NDArray,
                                    length_scale_px: int = 2048) -> NDArray:
    # Normalize coords to [-1,1]
    ys = centers[:,0].astype(np.float64)
    xs = centers[:,1].astype(np.float64)
    y_n = 2.0*(ys/(H-1)) - 1.0
    x_n = 2.0*(xs/(W-1)) - 1.0
    X = poly2_design_matrix(x_n, y_n).reshape(-1,6)
    lam = 1e-3
    XtX = X.T @ X + lam * np.eye(6)
    beta = np.linalg.solve(XtX, X.T @ alphas.astype(np.float64))
    return beta  # coefficients for evaluation

def eval_poly_alpha(beta: NDArray, H:int, W:int, y0:int, y1:int, x0:int, x1:int) -> NDArray:
    ys = np.arange(y0, y1, dtype=np.float32)
    xs = np.arange(x0, x1, dtype=np.float32)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    y_n = 2.0*(Y/(H-1)) - 1.0
    x_n = 2.0*(X/(W-1)) - 1.0
    B0,B1,B2,B3,B4,B5 = beta
    alpha = (B0 + B1*x_n + B2*y_n + B3*(x_n**2) + B4*(x_n*y_n) + B5*(y_n**2)).astype(np.float32)
    return alpha

def apply_subtraction(M: NDArray, A: NDArray, alpha: NDArray,
                      strong_pos_cap_p: Optional[float] = 99.0,
                      local_cap_quantile: Optional[float] = 0.98,
                      local_cap_window: int = 256) -> NDArray:
    M32 = M.astype(np.float32)
    out = M32 - alpha.astype(np.float32) * A.astype(np.float32)
    out[out < 0] = 0.0
    if strong_pos_cap_p is not None:
        # print(f'Imposing positive cap on subtraction...')
        thr = np.percentile(M32, strong_pos_cap_p)
        maxv = float(M32.max()) if M32.size else thr
        if maxv > thr:
            w = np.clip((maxv - M32) / (maxv - thr), 0.0, 1.0)
            out = w * out + (1 - w) * M32
    if local_cap_quantile is not None:
        # print(f'Imposing cap on a local level...')
        H, W = M32.shape
        bs = max(1, local_cap_window // 2)
        Hs, Ws = (H + bs - 1)//bs, (W + bs - 1)//bs
        caps = np.zeros((Hs, Ws), dtype=np.float32)
        for by in range(Hs):
            for bx in range(Ws):
                y0, y1 = by*bs, min((by+1)*bs, H)
                x0, x1 = bx*bs, min((bx+1)*bs, W)
                caps[by, bx] = np.quantile(M32[y0:y1, x0:x1], local_cap_quantile)
        caps = np.repeat(np.repeat(caps, bs, axis=0), bs, axis=1)[:H, :W]
        delta = M32 - out
        too_much = delta > caps
        out[too_much] = M32[too_much] - caps[too_much]
        out[out < 0] = 0.0
    # Cast back to original dtype
    if M.dtype == np.uint16:
        return np.clip(out, 0, 65535).astype(np.uint16)
    elif M.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        return out.astype(M.dtype)
    

def apply_saved_global_subtraction(mif_im, param_path, af_channel, tile=4096, apply_beta=False):
    C, H, W = mif_im.shape

    # load global parameters
    global_param_frame = pd.read_csv(param_path)

    # get unique channels
    channels_analyzed = np.unique(global_param_frame['channel'])

    # grab the autofluorescence channel
    af_channel_im = np.copy(np.squeeze(mif_im[af_channel,:,:]))

    im_slices = []
    for c in range(C):
        # grab a row of the parameter frame where we have the channel of interest
        channel_slice = np.copy(np.squeeze(mif_im[c,:,:]))
        if not np.any(channels_analyzed==c):
            print(f'Not adjusting channel: {c}')
            im_slices.append(channel_slice[np.newaxis,:,:])
            continue

        param_row = global_param_frame[global_param_frame['channel']==c]
        alpha = param_row['alpha'].item()
        beta = param_row['beta'].item()
        
        print(f'Processing channel: {c}')
        for (y0,y1,x0,x1) in tqdm(compute_tiles(H,W,tile)):
            M_tile = channel_slice[y0:y1,x0:x1]
            A_tile = af_channel_im[y0:y1,x0:x1]

            alpha_tile = np.full(M_tile.shape, alpha, dtype=np.float32)

            out_tile = apply_subtraction(M_tile, A_tile, alpha_tile)
            channel_slice[y0:y1,x0:x1] = out_tile

        im_slices.append(channel_slice[np.newaxis,:,:])

    return np.concatenate(im_slices, axis=0)


# this is a TODO we don't need to deal with this for now
# def apply_saved_tilewise_subtraction(mif_im, param_path, af_channel):
#     C, H, W = mif_im.shape

#     tilewise_param_frame = pd.read_csv(param_path)

#     af_channel_im = np.copy(np.squeeze(mif_im[af_channel,:,:]))
#     channels_analyzed = np.unique(tilewise_param_frame['channel'])


#     im_slices = []
#     for c in C:
#         channel_slice = np.copy(np.squeeze(mif_im[c,:,:]))
#         if not np.any(channels_analyzed==c):
#             print(f'Not adjusting channel: {c}')
#             im_slices.append(channel_slice)
#             continue
    
#         channel_frame = tilewise_param_frame[tilewise_param_frame['channel']==c]

#         # order should not matter, we should be able to iterate over the image and apply the subtraction
#         # TODO handle this later


        

