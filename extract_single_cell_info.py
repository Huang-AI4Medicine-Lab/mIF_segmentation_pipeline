import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import measure, morphology

from autofluorescence_utils import apply_saved_global_subtraction


ROI_IMAGE_SUFFIXES = (".ome.tif", ".ome.tiff", ".tif", ".tiff")


def parse_marker_spec(marker_spec):
    if marker_spec is None or not marker_spec.strip():
        raise ValueError(
            "Marker specification is required. Use the format 'DAPI:0,CD3:5,SOX10:4'."
        )

    markers = []
    seen_names = set()
    for item in marker_spec.split(","):
        clean_item = item.strip()
        if not clean_item:
            continue

        if ":" not in clean_item:
            raise ValueError(
                f"Invalid marker entry '{clean_item}'. Expected 'marker_name:channel_index'."
            )

        marker_name, channel_index = clean_item.rsplit(":", 1)
        marker_name = marker_name.strip()
        channel_index = channel_index.strip()

        if not marker_name:
            raise ValueError(f"Invalid marker entry '{clean_item}': marker name is empty.")
        if marker_name in seen_names:
            raise ValueError(f"Duplicate marker name '{marker_name}' detected.")

        try:
            marker_index = int(channel_index)
        except ValueError as exc:
            raise ValueError(
                f"Invalid channel index '{channel_index}' for marker '{marker_name}'."
            ) from exc

        if marker_index < 0:
            raise ValueError(
                f"Invalid channel index '{marker_index}' for marker '{marker_name}'."
            )

        markers.append((marker_name, marker_index))
        seen_names.add(marker_name)

    if not markers:
        raise ValueError("No markers were parsed from the marker specification.")

    return markers


def parse_roi_metadata(filename, mask_suffix):
    stem = Path(filename).name
    if mask_suffix and stem.endswith(mask_suffix):
        stem = stem[: -len(mask_suffix)]
    else:
        for suffix in ROI_IMAGE_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

    match = re.match(
        r"(?P<sample_name>.+)_roi_(?P<region_id>\d+)_row_(?P<row_start>\d+)_col_(?P<col_start>\d+)$",
        stem,
    )
    if not match:
        raise ValueError(
            f"Could not parse ROI metadata from '{filename}'. "
            "Expected '{sample_name}_roi_{region_id}_row_{row_start}_col_{col_start}'."
        )

    metadata = match.groupdict()
    metadata["region_id"] = int(metadata["region_id"])
    metadata["row_start"] = int(metadata["row_start"])
    metadata["col_start"] = int(metadata["col_start"])
    metadata["region_name"] = stem
    return metadata


def prepare_labeled_mask(mask_im, dilate_masks=False, dilation_size=5):
    mask_arr = np.squeeze(mask_im)
    if mask_arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask image. Received shape {mask_arr.shape}.")

    if np.max(mask_arr) <= 1:
        labeled_mask = measure.label(mask_arr > 0, connectivity=1)
    else:
        labeled_mask = mask_arr.astype(np.uint32)

    if dilate_masks:
        labeled_mask = morphology.dilation(
            labeled_mask, footprint=np.ones((dilation_size, dilation_size), dtype=np.uint8)
        )

    return labeled_mask


def validate_stack_and_prepare_channels(intensity_im, marker_defs):
    if intensity_im.ndim != 3:
        raise ValueError(
            f"Expected a channel-first image stack with shape (C, H, W). Received {intensity_im.shape}."
        )

    channel_count = intensity_im.shape[0]
    channel_lookup = {}
    for marker_name, channel_index in marker_defs:
        if channel_index >= channel_count:
            raise ValueError(
                f"Marker '{marker_name}' requested channel {channel_index}, "
                f"but the image only has {channel_count} channels."
            )
        channel_lookup[marker_name] = np.asarray(intensity_im[channel_index])

    return channel_lookup


def build_output_columns(marker_defs):
    base_columns = [
        "mode",
        "sample_name",
        "src_image",
        "roi_image",
        "region_name",
        "roi_region_id",
        "roi_row_start",
        "roi_col_start",
        "dataset_cell_id",
        "roi_cell_id",
        "cell_centroid_y",
        "cell_centroid_x",
        "wsi_cell_centroid_y",
        "wsi_cell_centroid_x",
        "cell_area",
    ]

    marker_columns = []
    for marker_name, _ in marker_defs:
        marker_columns.append(f"{marker_name}_mean_intensity")
        marker_columns.append(f"{marker_name}_median_intensity")

    return base_columns + marker_columns


def extract_records_from_mask(
    mode,
    sample_name,
    src_image_name,
    roi_image_name,
    region_name,
    roi_region_id,
    row_start,
    col_start,
    labeled_mask,
    intensity_im,
    marker_defs,
    start_dataset_cell_id,
):
    channel_lookup = validate_stack_and_prepare_channels(intensity_im, marker_defs)
    records = []
    dataset_cell_id = start_dataset_cell_id

    for mask_region in measure.regionprops(labeled_mask):
        mask_coords = mask_region.coords
        centroid_y = float(mask_region.centroid[0])
        centroid_x = float(mask_region.centroid[1])

        cell_record = {
            "mode": mode,
            "sample_name": sample_name,
            "src_image": src_image_name,
            "roi_image": roi_image_name,
            "region_name": region_name,
            "roi_region_id": roi_region_id,
            "roi_row_start": row_start,
            "roi_col_start": col_start,
            "dataset_cell_id": dataset_cell_id,
            "roi_cell_id": int(mask_region.label),
            "cell_centroid_y": centroid_y,
            "cell_centroid_x": centroid_x,
            "wsi_cell_centroid_y": centroid_y + float(row_start),
            "wsi_cell_centroid_x": centroid_x + float(col_start),
            "cell_area": float(mask_region.area),
        }

        for marker_name, _ in marker_defs:
            channel_slice = channel_lookup[marker_name]
            intensity_vals = channel_slice[mask_coords[:, 0], mask_coords[:, 1]].astype(np.float64)
            cell_record[f"{marker_name}_mean_intensity"] = float(np.mean(intensity_vals))
            cell_record[f"{marker_name}_median_intensity"] = float(np.median(intensity_vals))

        records.append(cell_record)
        dataset_cell_id += 1

    return records, dataset_cell_id


def resolve_roi_image_path(roi_dir, region_name):
    for suffix in ROI_IMAGE_SUFFIXES:
        candidate_path = Path(roi_dir) / f"{region_name}{suffix}"
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(
        f"Could not find ROI image for '{region_name}' in '{roi_dir}'. "
        f"Checked suffixes: {', '.join(ROI_IMAGE_SUFFIXES)}."
    )


def extract_region_based_cells(args, marker_defs):
    roi_dir = Path(args.roi_dir)
    mask_dir = Path(args.mask_dir)
    mask_suffix = args.mask_suffix
    sample_prefix = f"{args.sample_name}_"

    mask_paths = sorted(
        [
            path
            for path in mask_dir.iterdir()
            if path.is_file() and path.name.startswith(sample_prefix) and path.name.endswith(mask_suffix)
        ]
    )

    if not mask_paths:
        raise FileNotFoundError(
            f"No ROI mask files matching sample '{args.sample_name}' were found in '{mask_dir}'."
        )

    all_records = []
    dataset_cell_id = 0

    for mask_path in mask_paths:
        roi_metadata = parse_roi_metadata(mask_path.name, mask_suffix)
        roi_image_path = resolve_roi_image_path(roi_dir, roi_metadata["region_name"])

        mask_im = tifffile.imread(mask_path)
        labeled_mask = prepare_labeled_mask(
            mask_im,
            dilate_masks=args.dilate_masks,
            dilation_size=args.dilation_size,
        )

        if np.max(labeled_mask) < 1:
            continue

        intensity_im = tifffile.imread(roi_image_path)
        region_records, dataset_cell_id = extract_records_from_mask(
            mode="region_based",
            sample_name=args.sample_name,
            src_image_name=args.sample_name,
            roi_image_name=roi_image_path.name,
            region_name=roi_metadata["region_name"],
            roi_region_id=roi_metadata["region_id"],
            row_start=roi_metadata["row_start"],
            col_start=roi_metadata["col_start"],
            labeled_mask=labeled_mask,
            intensity_im=intensity_im,
            marker_defs=marker_defs,
            start_dataset_cell_id=dataset_cell_id,
        )
        all_records.extend(region_records)

    return all_records


def extract_image_based_cells(args, marker_defs):
    image_path = Path(args.image_path)
    wsi_mask_path = Path(args.wsi_mask_path)

    intensity_im = tifffile.imread(image_path)
    if args.remove_autofluorescence:
        intensity_im = apply_saved_global_subtraction(
            intensity_im,
            args.autofluorescence_params,
            af_channel=args.af_channel,
        )

    mask_im = tifffile.imread(wsi_mask_path)
    labeled_mask = prepare_labeled_mask(
        mask_im,
        dilate_masks=args.dilate_masks,
        dilation_size=args.dilation_size,
    )

    if np.max(labeled_mask) < 1:
        return []

    records, _ = extract_records_from_mask(
        mode="image_based",
        sample_name=args.sample_name,
        src_image_name=image_path.name,
        roi_image_name=image_path.name,
        region_name=args.sample_name,
        roi_region_id=-1,
        row_start=0,
        col_start=0,
        labeled_mask=labeled_mask,
        intensity_im=intensity_im,
        marker_defs=marker_defs,
        start_dataset_cell_id=0,
    )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["region_based", "image_based"])
    parser.add_argument("--sample_name", required=True)
    parser.add_argument("--roi_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--wsi_mask_path", default=None)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--marker_spec", required=True)
    parser.add_argument("--mask_suffix", default="_seg.tiff")
    parser.add_argument("--dilate_masks", action="store_true")
    parser.add_argument("--dilation_size", type=int, default=5)
    parser.add_argument("--remove_autofluorescence", action="store_true")
    parser.add_argument("--autofluorescence_params", default=None)
    parser.add_argument("--af_channel", type=int, default=None)
    args = parser.parse_args()

    marker_defs = parse_marker_spec(args.marker_spec)

    if args.mode == "region_based":
        if args.roi_dir is None or args.mask_dir is None:
            raise ValueError("--roi_dir and --mask_dir are required for region_based mode.")
        records = extract_region_based_cells(args, marker_defs)
    else:
        if args.image_path is None or args.wsi_mask_path is None:
            raise ValueError("--image_path and --wsi_mask_path are required for image_based mode.")
        if args.remove_autofluorescence:
            if not args.autofluorescence_params:
                raise ValueError(
                    "--autofluorescence_params is required when --remove_autofluorescence is set."
                )
            if args.af_channel is None:
                raise ValueError("--af_channel is required when --remove_autofluorescence is set.")
        records = extract_image_based_cells(args, marker_defs)

    output_columns = build_output_columns(marker_defs)
    cell_frame = pd.DataFrame.from_records(records, columns=output_columns)

    output_path = Path(args.save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell_frame.to_csv(output_path, index=False)

    print(
        f"Saved {len(cell_frame)} extracted cells for sample '{args.sample_name}' "
        f"to '{output_path}'."
    )


if __name__ == "__main__":
    main()
