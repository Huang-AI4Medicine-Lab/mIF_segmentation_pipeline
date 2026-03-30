# mIF Sequential Pipeline

Nextflow DSL2 pipeline for sequential multiplex immunofluorescence (mIF) image processing. The workflow generates tissue masks, optionally estimates autofluorescence, extracts ROIs, runs nuclei segmentation, optionally runs marker-channel segmentation and mask merging, produces whole-slide segmentation masks, and can optionally export per-cell intensity tables.

## Repository Contents

- `modular_pipeline_sequential.nf`: main Nextflow workflow
- `modular_pipeline.config`: runtime parameters, paths, and executor settings
- `*.py`: Python utilities for masking, ROI extraction, segmentation, mask merging, and WSI mask generation

## Main Steps

1. Generate tissue masks from input images
2. Estimate autofluorescence parameters
3. Extract tissue-rich ROIs
4. Segment nuclei in ROIs
5. Optionally segment marker channels and merge masks
6. Reconstruct final WSI-level masks
7. Optionally extract per-cell intensity measurements

## Running

Update the paths in `modular_pipeline.config` before running, especially:

- `script_dir`
- `sample_list`
- `image_list`
- output directories
- Conda environment paths
- SLURM/executor settings

If you enable cell extraction, also set:

- `run_cell_extraction`
- `cell_extraction_mode`
- `cell_extraction_markers`

`cell_extraction_mode = 'region_based'` measures cells from the per-ROI masks and records ROI-local plus WSI-adjusted centroids. `cell_extraction_mode = 'image_based'` measures cells from the reconstructed whole-slide mask and can apply the saved autofluorescence correction parameters to the whole-slide image before intensity extraction.

Run with:

```bash
nextflow run modular_pipeline_sequential.nf -c modular_pipeline.config
```
