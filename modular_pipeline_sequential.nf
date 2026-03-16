#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// ============================================================================
// PROCESS DEFINITIONS
// ============================================================================

process GenerateTissueMask {
    publishDir params.tissue_mask_output ?: './tissue_masks', mode: 'copy', enabled: params.save_intermediate, pattern: '*_tissue_mask.ome.tif'
    publishDir params.tissue_mask_metadata ?: './tissue_mask_metadata', mode: 'copy', enabled: params.save_intermediate, pattern: '*_tissue_mask_overlay.jpg'
    
    input:
    tuple val(sample_name), path(image_path)
    
    output:
    tuple val(sample_name), path(image_path), path("${sample_name}_tissue_mask.ome.tif"), emit: masks
    path "${sample_name}_tissue_mask_overlay.jpg", emit: overlays
    
    script:
    """
    python "${params.script_dir}/tissue_mask_generator.py" \
        --image_path "$image_path" \
        --sample_name "$sample_name" \
        --channel_reduce ${params.channel_reduce} \
        --num_thresholds ${params.num_thresholds} \
        --thresh_mode ${params.thresh_mode}
    """
}

process EstimateAutofluorescence {
    publishDir params.af_param_output ?: './af_params', mode: 'copy', enabled: params.save_intermediate, pattern: "*_global_params.csv"
    publishDir params.af_param_output ?: './af_params', mode: 'copy', enabled: params.save_intermediate, pattern: "*_local_params.csv"
    
    input:
    tuple val(sample_name), path(image_path), path(tissue_mask)
    
    output:
    tuple val(sample_name), path(image_path), path(tissue_mask), path("af_params/${sample_name}_global_params.csv"), emit: af_params
    
    script:
    def smooth_field = params.af_smooth_field ? "--smooth_field" : ""
    def write_channels = params.af_write_channel_ims ? "--write_channel_ims" : ""
    def channels_arg = params.af_channels_to_correct ? "--channels_to_correct ${params.af_channels_to_correct}" : ""
    def af_dest = params.af_param_output ?: './af_params'
    """
    python ${params.script_dir}/autofluorescence_filtering.py \
        --stack_path "$image_path" \
        --sample_name "$sample_name" \
        --af_channel ${params.af_channel} \
        --tissue_mask_path "$tissue_mask" \
        --out_path af_corrected_channels \
        --param_out_path af_params \
        --tile ${params.af_tile_size} \
        --sample_method ${params.af_sample_method} \
        --n_samples ${params.af_n_samples} \
        --drop_marker_top_p ${params.af_drop_marker_top_p} \
        --drop_af_top_p ${params.af_drop_af_top_p} \
        ${smooth_field} \
        ${write_channels} \
        ${channels_arg}

    if [ "${params.save_intermediate}" = "true" ]; then
        mkdir -p "${af_dest}"
        cp af_params/*.csv "${af_dest}/"
    fi
    """
}


process ExtractRegions {
    publishDir params.roi_check_output ?: './roi_metadata', mode: 'copy', enabled: params.save_intermediate, pattern: "*_roi_mask_im.tif"
    publishDir params.roi_check_output ?: './roi_metadata', mode: 'copy', enabled: params.save_intermediate, pattern: "*_region_mask_overlay.jpg"

    input:
    tuple val(sample_name), path(image_path), path(mask_path), path(af_params)
    
    output:
    tuple val(sample_name), path("region_images/"), emit: roi_dir
    tuple val(sample_name), path("${sample_name}_roi_mask_im.tif"), emit: boundary_mask
    tuple val(sample_name), path("${sample_name}_region_mask_overlay.jpg"), emit: region_overlay
    
    script:
    def roi_dest = params.roi_output ?: './extracted_rois'
    def af_flag = (params.apply_af_correction && af_params.name != 'NO_AF_PARAMS') ? "--remove_autofluorescence --autofluorescence_params ${af_params} --af_channel ${params.af_channel}" : ""
    """
    python ${params.script_dir}/roi_extractor.py \
        --image_path "$image_path" \
        --sample_name "$sample_name" \
        --existing_mask_fname "$mask_path" \
        --num_thresholds ${params.num_thresholds} \
        --thresh_mode ${params.thresh_mode} \
        --channel_reduce ${params.channel_reduce} \
        --roi_foreground_ratio ${params.roi_foreground_ratio} \
        --roi_row_size ${params.roi_row_size} \
        --roi_col_size ${params.roi_col_size} \
        --overlap ${params.region_overlap} \
        ${af_flag} \
        --save_regions

    if [ "${params.save_intermediate}" = "true" ]; then
        mkdir -p "${roi_dest}"
        cp region_images/*.ome.tif "${roi_dest}/"
    fi
    """
}

process SegmentNuclei {
    input:
    tuple val(sample_name), path(roi_dir)
    
    output:
    tuple val(sample_name), path("nuclei_seg_ims/"), emit: seg_dir
    
    script:
    def custom_model = params.nuclei_custom_model ? "--custom_model_path ${params.nuclei_custom_model}" : ""
    def use_gpu = params.nuclei_gpu ? "--gpu" : ""
    def seg_dest = params.segmentation_output ?: './segmentation_results'
    """
    python ${params.script_dir}/batch_segmentation.py \
        --image_dir "$roi_dir" \
        --sample_name "$sample_name" \
        --diameter ${params.nuclei_diameter} \
        --channel_type ${params.nuclei_channel_type} \
        --model_name ${params.nuclei_model_name} \
        ${use_gpu} \
        ${custom_model} \
        --cellpose_normalize

    if [ "${params.save_intermediate}" = "true" ]; then
        mkdir -p "${seg_dest}/nuclei_seg_ims"
        cp nuclei_seg_ims/*_seg.tiff "${seg_dest}/nuclei_seg_ims/"
    fi
    """
}

process SegmentMarkerChannel {
    tag "${sample_name}:${marker_name}"
    
    input:
    tuple val(sample_name), path(roi_dir), val(marker_name), val(channel_indices), path(custom_model)
    
    output:
    tuple val(sample_name), val(marker_name), path("${marker_name}_seg_ims/"), emit: seg_dir

    script:
    def use_gpu = params.nuclei_gpu ? "--gpu" : ""
    def seg_dest = params.segmentation_output ?: './segmentation_results'
    """
    # Create channel selection file for this marker
    python -c "import numpy as np; np.save('channel_selection.npy', np.array([${channel_indices}]))"
    
    python ${params.script_dir}/batch_segmentation.py \
        --image_dir "$roi_dir" \
        --sample_name "$sample_name" \
        --marker_name "$marker_name" \
        --diameter ${params.marker_diameter} \
        --channel_type dual_channel \
        --channel_selection_path channel_selection.npy \
        --custom_model_path "$custom_model" \
        ${use_gpu} \
        --cellpose_normalize
    

   if [ "${params.save_intermediate}" = "true" ]; then
        mkdir -p "${seg_dest}/${marker_name}_seg_ims"
        cp ${marker_name}_seg_ims/*_seg.tiff "${seg_dest}/${marker_name}_seg_ims/"
    fi
    """
}


process MergeSegmentationMasks {
    input:
    tuple val(sample_name), path(seg_dirs)
    
    output:
    tuple val(sample_name), path("merged_seg_ims/"), emit: merged_dir

    script:
    def exclude_nuclei = params.merge_exclude_nuclei ? "--exclude_nuclei" : ""
    def seg_dest = params.segmentation_output ?: './segmentation_results'
    def dir_list = seg_dirs.collect { it.toString() }.join(',')
    """
    python ${params.script_dir}/batch_mask_merger.py \
        --mask_dirs "${dir_list}" \
        --sample_name "$sample_name" \
        --method ${params.merge_method} \
        --ioq_thresh ${params.merge_ioq_thresh} \
        --save_dir merged_seg_ims \
        ${exclude_nuclei}

    if [ "${params.save_intermediate}" = "true" ]; then
        mkdir -p "${seg_dest}/merged_seg_ims"
        cp merged_seg_ims/*.tiff "${seg_dest}/merged_seg_ims/"
    fi
    """
}

process GenerateWSIMask {
    publishDir params.final_output_dir ?: './final_wsi_masks', mode: 'copy', pattern: "*_wsi_mask.ome.tif"
    
    input:
    tuple val(sample_name), path(image_path), path(seg_mask_dir)
    
    output:
    tuple val(sample_name), path("${sample_name}_wsi_mask.ome.tif")
    
    script:
    def dilate = params.dilate_final_mask ? "--dilate_mask --dilation_size ${params.dilation_size}" : ""
    """
    python ${params.script_dir}/wsi_segmentation_mask.py \
        --sample_name "$sample_name" \
        --intensity_im_path "$image_path" \
        --mask_dir $seg_mask_dir \
        --overlap_method ${params.overlap_method} \
        --centroid_radius ${params.centroid_radius} \
        --row_region_size ${params.roi_row_size} \
        --col_region_size ${params.roi_col_size} \
        --region_overlap ${params.region_overlap} \
        --artifact_filter_mode ${params.artifact_filter_mode} \
        --flatness_threshold ${params.flatness_threshold} \
        --min_solidity ${params.min_solidity} \
        --min_artifact_area ${params.min_artifact_area} \
        --border_width ${params.border_width} \
        ${dilate}
    """
}

// ============================================================================
// MAIN WORKFLOW ENTRY POINT
// ============================================================================

workflow {
    // Read input files
    sample_names = Channel.fromPath(params.sample_list).splitText().map { it.trim() }
    image_paths = Channel.fromPath(params.image_list).splitText().map { it.trim() }
    
    samples_ch = sample_names
        .merge(image_paths)
        .map { sample, image -> tuple(sample, image) }
    
    // Step 1: Generate tissue masks
    GenerateTissueMask(samples_ch)

    // Step 2: Optionally estimate autofluorescence, then extract regions
    if (params.apply_af_correction) {
        // Estimate AF parameters
        EstimateAutofluorescence(GenerateTissueMask.out.masks)
        extract_input_ch = EstimateAutofluorescence.out.af_params
    } else {
        // Add placeholder for AF params to maintain consistent tuple structure
        extract_input_ch = GenerateTissueMask.out.masks
            .map { sample, image, mask -> tuple(sample, image, mask, file("NO_AF_PARAMS")) }
    }

    // step 3: extract regions
    ExtractRegions(extract_input_ch)
    roi_dir_ch = ExtractRegions.out.roi_dir

    // Step 4: Segment nuclei
    SegmentNuclei(roi_dir_ch)

    // Step 5: Optionally segment marker channels and merge
    if (params.run_marker_segmentation) {
        // Create channel for marker configurations
        marker_configs_ch = Channel.from(params.marker_configs)
            .map { config -> 
                tuple(
                    config.marker_name,
                    config.channel_indices,
                    file(config.model_path)
                )
            }
        
        // Combine ROI dirs with marker configs
        marker_input_ch = roi_dir_ch
            .combine(marker_configs_ch)
            .map { sample, roi_dir, marker_name, channels, model -> 
                tuple(sample, roi_dir, marker_name, channels, model)
            }

        // Run marker segmentation for each sample/marker combination
        SegmentMarkerChannel(marker_input_ch)
        
        // Group marker segmentation outputs by sample
        marker_segs_grouped = SegmentMarkerChannel.out.seg_dir
            .map { sample, marker_name, seg_dir -> tuple(sample, seg_dir) }
            .groupTuple(by: 0)

        // Combine nuclei seg with marker segs into single list per sample
        // nuclei: [sample, nuclei_dir]
        // markers: [sample, [marker1_dir, marker2_dir, ...]]
        // result: [sample, [nuclei_dir, marker1_dir, marker2_dir, ...]]
        merge_input_ch = SegmentNuclei.out.seg_dir
            .join(marker_segs_grouped)
            .map { sample, nuclei_dir, marker_dirs -> 
                def all_dirs = [nuclei_dir] + marker_dirs
                tuple(sample, all_dirs)
            }
        
        // Merge all segmentation masks
        MergeSegmentationMasks(merge_input_ch)
        
        // Use merged masks for WSI generation
        final_seg_ch = MergeSegmentationMasks.out.merged_dir
    } else {
        // Use nuclei-only segmentation for WSI generation
        final_seg_ch = SegmentNuclei.out.seg_dir
    }

    samples_ch
        .join(final_seg_ch)
        .map { sample, img, seg_dir -> tuple(sample, img, seg_dir) }
        .set { wsi_input_ch }
    
    // Step 4: Generate WSI masks
    GenerateWSIMask(wsi_input_ch)
}

