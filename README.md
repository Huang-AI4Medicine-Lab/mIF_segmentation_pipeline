# WSI Multiplex Pipeline

Nextflow pipeline for WSI multiplex imaging segmentation. The workflow generates tissue masks, optionally estimates autofluorescence, extracts ROIs, runs nuclei segmentation, optionally runs marker-channel segmentation and mask merging, and produces whole-slide segmentation masks.

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

## Running

Update the paths in `modular_pipeline.config` before running, especially:

- `script_dir`
- `sample_list`
- `image_list`
- output directories
- Conda environment paths
- SLURM/executor settings

Run with:

```bash
nextflow run modular_pipeline_sequential.nf -c modular_pipeline.config
```
