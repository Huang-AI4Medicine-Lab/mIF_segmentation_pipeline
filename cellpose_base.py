"""
Cell Segmentation for Multiplex Immunofluorescence Images using Cellpose
=========================================================================
Segments cells from mIF images using DAPI (channel 0) as the nuclear channel.
Supports built-in Cellpose models and custom/finetuned models.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from cellpose import models, io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "dapi_channel": 0,          # Channel index for DAPI nuclear stain
    "model_type": "nuclei",      # Built-in model: cyto, cyto2, cyto3, nuclei
    "diameter": None,            # None = auto-estimate; set float for manual
    "flow_threshold": 0.4,      # Increase to get fewer masks, decrease for more
    "cellprob_threshold": 0.0,  # Increase to shrink masks, decrease to expand
    "min_size": 15,              # Minimum mask area in pixels
    "do_3d": False,              # Set True for volumetric segmentation
    "gpu": True,                 # Use GPU if available
    "batch_size": 8,             # Batch size for GPU inference
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_type: str = DEFAULT_CONFIG["model_type"],
    custom_model_path: Optional[str] = None,
    gpu: bool = DEFAULT_CONFIG["gpu"],
) -> models.CellposeModel:
    """
    Load a Cellpose model — either a built-in model or a custom/finetuned one.

    Parameters
    ----------
    model_type : str
        Name of a built-in model (e.g. 'cyto3', 'nuclei'). Ignored when
        *custom_model_path* is provided.
    custom_model_path : str or None
        Path to a finetuned / custom Cellpose model file. When provided,
        this model is loaded instead of a built-in one.
    gpu : bool
        Whether to attempt GPU acceleration.

    Returns
    -------
    models.CellposeModel
    """
    if custom_model_path is not None:
        p = Path(custom_model_path)
        if not p.is_file():
            raise FileNotFoundError(f"Custom model not found: {p}")
        logger.info("Loading custom model from %s", p)
        model = models.CellposeModel(pretrained_model=str(p), gpu=gpu)
    else:
        logger.info("Loading built-in model '%s'", model_type)
        model = models.CellposeModel(model_type=model_type, gpu=gpu)
    return model


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> np.ndarray:
    """
    Load a multiplex IF image (TIFF).

    Returns an array of shape (C, H, W) regardless of the on-disk layout.
    """
    img = tifffile.imread(image_path)
    logger.info("Raw image shape: %s, dtype: %s", img.shape, img.dtype)

    # Handle common dimension orders
    if img.ndim == 2:
        # Single-channel grayscale → (1, H, W)
        img = img[np.newaxis, ...]
    elif img.ndim == 3:
        # Could be (C, H, W) or (H, W, C)
        if img.shape[-1] <= 10:  # likely channels-last
            img = np.moveaxis(img, -1, 0)
    elif img.ndim == 4:
        # E.g. OME-TIFF with singleton Z: (1, C, H, W)
        img = img.squeeze()
        if img.ndim == 3 and img.shape[-1] <= 10:
            img = np.moveaxis(img, -1, 0)

    logger.info("Image reshaped to (C, H, W): %s", img.shape)
    return img


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment(
    image: np.ndarray,
    model: models.CellposeModel,
    dapi_channel: int = DEFAULT_CONFIG["dapi_channel"],
    diameter: Optional[float] = DEFAULT_CONFIG["diameter"],
    flow_threshold: float = DEFAULT_CONFIG["flow_threshold"],
    cellprob_threshold: float = DEFAULT_CONFIG["cellprob_threshold"],
    min_size: int = DEFAULT_CONFIG["min_size"],
    do_3d: bool = DEFAULT_CONFIG["do_3d"],
    membrane_channel: Optional[int] = None,
    batch_size: int = DEFAULT_CONFIG["batch_size"],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run Cellpose segmentation on a multiplex IF image.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (C, H, W).
    model : CellposeModel
        Loaded Cellpose model.
    dapi_channel : int
        Index of the DAPI channel (nuclear stain).
    diameter : float or None
        Expected cell diameter in pixels. None lets Cellpose auto-estimate.
    flow_threshold : float
        Error threshold on flows (higher → fewer masks).
    cellprob_threshold : float
        Cell probability threshold (higher → smaller masks).
    min_size : int
        Minimum connected-component area (pixels) to keep.
    do_3d : bool
        Volumetric segmentation flag.
    membrane_channel : int or None
        Optional second channel for cytoplasm/membrane signal. If None,
        Cellpose uses the nuclear channel only (channels=[0, 0]).
    batch_size : int
        Batch size for tiled GPU inference.

    Returns
    -------
    masks : np.ndarray  (H, W) int32 label image
    flows : np.ndarray  Cellpose flow fields
    diams  : dict       Estimated diameters
    """
    n_channels = image.shape[0]
    logger.info("Total channels in image: %d", n_channels)
    logger.info("Using DAPI channel: %d", dapi_channel)

    # Build the 2-channel input Cellpose expects:
    #   channel 0 → cytoplasm / membrane (or nuclear if no membrane)
    #   channel 1 → nucleus (DAPI)
    # When only DAPI is available we set channels=[0, 0].
    if membrane_channel is not None and membrane_channel != dapi_channel:
        logger.info("Using membrane channel %d as cytoplasm signal", membrane_channel)
        input_img = np.stack(
            [image[membrane_channel], image[dapi_channel]], axis=-1
        )  # (H, W, 2)
        channels = [1, 2]  # Cellpose convention: 1=green/cyto, 2=blue/nuclei
    else:
        input_img = image[dapi_channel]  # (H, W)
        channels = [0, 0]  # grayscale / nuclear-only

    logger.info(
        "Running segmentation (diameter=%s, flow_thresh=%.2f, cellprob_thresh=%.2f)",
        diameter,
        flow_threshold,
        cellprob_threshold,
    )

    masks, flows, _ = model.eval(
        input_img,
        diameter=diameter,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        do_3D=do_3d,
        batch_size=batch_size,
    )

    n_cells = masks.max()
    logger.info("Segmentation complete — %d cells detected", n_cells)
    return masks, flows, {"n_cells": n_cells, "diameter": diameter}


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(
    masks: np.ndarray,
    output_dir: str,
    stem: str,
    save_outlines: bool = True,
) -> Path:
    """Save the label mask (and optional outlines) to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask_path = out / f"{stem}_masks.tif"
    tifffile.imwrite(str(mask_path), masks.astype(np.int32), compression="zlib")
    logger.info("Masks saved → %s", mask_path)

    if save_outlines:
        from cellpose.utils import outlines_list

        outlines = outlines_list(masks)
        outline_img = np.zeros_like(masks, dtype=np.uint8)
        for outline in outlines:
            outline_img[outline[:, 0], outline[:, 1]] = 255
        outline_path = out / f"{stem}_outlines.tif"
        tifffile.imwrite(str(outline_path), outline_img, compression="zlib")
        logger.info("Outlines saved → %s", outline_path)

    return mask_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segment cells in multiplex IF images with Cellpose."
    )
    p.add_argument("image", type=str, help="Path to the input mIF TIFF image.")
    p.add_argument(
        "-o", "--output-dir", type=str, default="./segmentation_output",
        help="Directory for output masks (default: ./segmentation_output).",
    )

    # Model options
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--model-type", type=str, default=DEFAULT_CONFIG["model_type"],
        help="Built-in model name (default: cyto3). Options: cyto, cyto2, cyto3, nuclei.",
    )
    g.add_argument(
        "--custom-model", type=str, default=None,
        help="Path to a custom / finetuned Cellpose model file.",
    )

    # Segmentation parameters
    p.add_argument("--diameter", type=float, default=None,
                   help="Expected cell diameter in px (default: auto-estimate).")
    p.add_argument("--flow-threshold", type=float, default=DEFAULT_CONFIG["flow_threshold"])
    p.add_argument("--cellprob-threshold", type=float, default=DEFAULT_CONFIG["cellprob_threshold"])
    p.add_argument("--min-size", type=int, default=DEFAULT_CONFIG["min_size"])
    p.add_argument("--membrane-channel", type=int, default=None,
                   help="Channel index for a membrane/cytoplasm marker (optional).")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU inference.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--no-outlines", action="store_true",
                   help="Skip saving outline images.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load image
    image = load_image(args.image)

    # Load model (custom or built-in)
    model = load_model(
        model_type=args.model_type,
        custom_model_path=args.custom_model,
        gpu=not args.no_gpu,
    )

    # Run segmentation
    masks, flows, info = segment(
        image,
        model,
        dapi_channel=DEFAULT_CONFIG["dapi_channel"],
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        membrane_channel=args.membrane_channel,
        batch_size=args.batch_size,
    )

    # Save
    stem = Path(args.image).stem
    save_results(masks, args.output_dir, stem, save_outlines=not args.no_outlines)
    logger.info("Done. %d cells segmented.", info["n_cells"])


if __name__ == "__main__":
    main()
