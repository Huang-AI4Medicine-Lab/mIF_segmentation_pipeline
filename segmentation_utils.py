import numpy as np 
from skimage import measure, segmentation, morphology
from scipy import ndimage as ndi
from tqdm import tqdm


def detect_flat_edge_masks(labeled_mask, extent_thresh=0.85, min_solidity=0.80, min_area=20):
    """
    Detect masks that have unnaturally flat edges (border truncation artifacts).
    
    These artifacts occur when a cell is cut off at a region boundary, producing
    a mask with one or more perfectly straight edges. Such masks tend to have:
      - High 'extent' (ratio of mask area to bounding box area), since the flat
        edge makes the mask fill its bounding box more completely
      - The flat edge itself can be detected by checking if a large fraction of
        the bounding box boundary pixels are filled
    
    Parameters:
    - labeled_mask: labeled image (integer labels)
    - extent_thresh: masks with extent above this are candidates for removal
    - min_solidity: used as a secondary filter
    - min_area: ignore very small masks
    
    Returns:
    - set of label IDs to remove
    """
    regions = measure.regionprops(labeled_mask)
    labels_to_remove = set()
    
    for region in regions:
        if region.area < min_area:
            continue
        
        # Extract the bounding box of this mask
        min_row, min_col, max_row, max_col = region.bbox
        bbox_h = max_row - min_row
        bbox_w = max_col - min_col
        
        if bbox_h < 3 or bbox_w < 3:
            continue
        
        # Get the local binary mask within the bounding box
        local_mask = labeled_mask[min_row:max_row, min_col:max_col] == region.label
        
        # Check each edge of the bounding box for flatness
        # A flat edge means the mask fills a large fraction of that bounding box edge
        top_edge_fill = np.mean(local_mask[0, :])
        bottom_edge_fill = np.mean(local_mask[-1, :])
        left_edge_fill = np.mean(local_mask[:, 0])
        right_edge_fill = np.mean(local_mask[:, -1])
        
        # A natural cell mask should not fill an entire edge of its bounding box
        # (unless it's very round and small). A truncated mask will have ~1.0 fill
        # on the cut edge.
        edge_fill_thresh = 0.8
        has_flat_edge = any(f > edge_fill_thresh for f in 
                          [top_edge_fill, bottom_edge_fill, left_edge_fill, right_edge_fill])
        
        if has_flat_edge and region.extent > extent_thresh:
            labels_to_remove.add(region.label)
            continue
        
        # Secondary check: high extent + low solidity can indicate weird truncation
        # (solidity = area / convex_hull_area)
        if region.extent > extent_thresh and region.solidity < min_solidity:
            labels_to_remove.add(region.label)
    
    return labels_to_remove

def remove_border_masks(labeled_mask, border_width=1):
    """
    Remove any masks that touch the border of the region image.
    
    This is the simplest and most reliable way to remove truncated masks,
    since cells cut off at region boundaries will always touch the image edge.
    
    Parameters:
    - labeled_mask: labeled image (integer labels)
    - border_width: pixel width of border to check (default 1)
    
    Returns:
    - filtered labeled mask with border-touching masks set to 0
    """
    return segmentation.clear_border(labeled_mask, buffer_size=border_width)


def filter_region_mask_artifacts(
    labeled_mask,
    mode='both',
    extent_thresh=0.85,
    min_solidity=0.80,
    min_area=20,
    border_width=1,
):
    """
    Apply artifact removal filters to a labeled region mask.
    
    Parameters:
    - labeled_mask: labeled image (integer labels)
    - mode: 'border', 'morphology', 'both', or 'none'
    - extent_thresh: threshold for morphology-based detection
    - min_solidity: minimum solidity for morphology-based detection
    - min_area: ignore very small masks during morphology checks
    - border_width: pixel width used for border-touch checks
    
    Returns:
    - filtered labeled mask
    """
    if mode == 'none':
        return labeled_mask
    
    filtered = np.copy(labeled_mask)
    
    if mode in ('border', 'both'):
        filtered = remove_border_masks(filtered, border_width=border_width)
    
    if mode in ('morphology', 'both'):
        flat_labels = detect_flat_edge_masks(
            filtered, 
            extent_thresh=extent_thresh,
            min_solidity=min_solidity,
            min_area=min_area,
        )
        if flat_labels:
            mask = np.isin(filtered, list(flat_labels))
            filtered[mask] = 0
    
    return filtered


def labeled_mask_splitting(label_image, kernel_size=3):
    out_labeled_image = np.copy(label_image)
    boundaries = segmentation.find_boundaries(label_image, mode='inner', background=0)
    boundary_arr = np.array(np.where(boundaries>0)).T
    kernel_lower = kernel_size//2
    kernel_upper = kernel_lower+1
    for index in range(0, boundary_arr.shape[0]):
        y, x = boundary_arr[index, :]
        kernel_slice = label_image[int(y-kernel_lower):int(y+kernel_upper), int(x-kernel_lower):int(x+kernel_upper)]
        if np.sum(np.unique(kernel_slice)>0) >= 2:
            out_labeled_image[int(y-kernel_lower):int(y+kernel_upper), int(x-kernel_lower):int(x+kernel_upper)] = 0
    
    return out_labeled_image

def filter_small_masks(bin_im, min_area=50):
    ''' Takes a binary segmentation image and filters all masks with a pixel area less than min area '''
    labeled_seg_im = measure.label(bin_im, connectivity=1)
    mask_regions = measure.regionprops(labeled_seg_im)
    for mask_region in mask_regions:
        current_label = mask_region.label
        mask_area = mask_region.area
        if mask_area<min_area:
            labeled_seg_im[labeled_seg_im==current_label] = 0
    
    return (labeled_seg_im>0).astype(np.uint8)

def split_weak_masks(bin_mask, dilation_radius=1):
    ''' Split any 1 or more pixel connections between masks '''
    struct_elem = morphology.disk(dilation_radius)
    eroded_masks = morphology.binary_erosion(bin_mask, footprint=struct_elem)

    return eroded_masks


def get_iou(mask_1,mask_2):
    mask_1 = mask_1.astype(bool)
    mask_2 = mask_2.astype(bool)
    
    intersection = np.logical_and(mask_1,mask_2).sum()
    union = np.logical_or(mask_1,mask_2).sum()

    if union==0.0:
        return 0.0

    return intersection/union

def get_ioq(mask_1, mask_2):
    ''' Here take two segmentation masks and evaluate intersection-over-query '''
    ''' This is defined as the ratio of the intersection of the area between two masks divided by the area of the smaller mask '''
    mask_1 = mask_1.astype(bool)
    mask_2 = mask_2.astype(bool)

    if np.sum(mask_1)<np.sum(mask_2):
        small_mask = mask_1
    else:
        small_mask = mask_2

    intersection = np.logical_and(mask_1,mask_2).sum()

    return intersection/np.sum(small_mask)


def tile_image(img, tile_overlap=0.20, tile_size=0.30, channel_axis=0):
    if channel_axis == 0:
        row_size, col_size = img.shape[1], img.shape[2]
    else:
        row_size, col_size = img.shape[0], img.shape[1]

    row_interval = int(np.floor(row_size*tile_size))
    col_interval = int(np.floor(col_size*tile_size))
    img_slice_list = []
    print(f'Creating tiles')
    for row_index in tqdm(range(0, row_size, row_interval)):
        for col_index in range(0, col_size, col_interval):
            if row_index>=(row_size-1):
                row_end = row_size-1
            else:
                row_end = row_index+row_interval
            if col_index>=(col_size-1):
                col_end = col_size-1
            else:
                col_end = col_index+col_interval

            if channel_axis==0:
                img_slice = img[:, row_index:row_end, col_index:col_end]
            else:
                img_slice = img[row_index:row_end, col_index:col_end, :]

            img_slice_list.append(img_slice)

    return img_slice_list

def combine_tiles(bin_tile_list, row_size, col_size, tile_overlap=0.20, tile_size=0.30):
    row_interval = int(np.floor(row_size*tile_size))
    col_interval = int(np.floor(col_size*tile_size))

    template_im = np.zeros((row_size, col_size), dtype=np.uint32)
    tile_count = 0
    for row_index in range(0, row_size, row_interval):
        for col_index in range(0, col_size, col_interval):
            if row_index>=(row_size-1):
                row_end = row_size-1
            else:
                row_end = row_index+row_interval
            if col_index>=(col_size-1):
                col_end = col_size-1
            else:
                col_end = col_index+col_interval

            template_im[row_index:row_end, col_index:col_end] += bin_tile_list[tile_count]
            tile_count += 1
    
    # binarize the image and separate labels
    template_im = (template_im>0)
    template_im = measure.label(template_im)
    template_im = labeled_mask_splitting(template_im)

    return template_im


def mask_separation_watershed(query_image, comp_image):
    """ Function ingests two masks that we want to merge """
    """ Both masks should be binary images """
    candidate_im = np.logical_or(query_image.astype(bool), comp_image.astype(bool))
    # calculate distances separately and subset the larger distance with the shorter distances
    dist_query = ndi.distance_transform_edt(query_image.astype(bool))
    dist_comp = ndi.distance_transform_edt(comp_image.astype(bool))
    if np.sum(query_image)>np.sum(comp_image):
        comp_coords = np.nonzero(dist_comp)
        dist_query[comp_coords] = dist_comp[comp_coords]
        distance=dist_query
    else:
        query_coords = np.nonzero(dist_query)
        dist_comp[query_coords] = dist_query[query_coords]
        distance=dist_comp

    new_mask = np.zeros(distance.shape, dtype=bool)
    query_centroid = measure.centroid(query_image).astype(np.uint16).reshape(1,-1)
    comp_centroid = measure.centroid(comp_image).astype(np.uint16).reshape(1,-1)
    combined_centroids = np.concatenate([query_centroid, comp_centroid], axis=0)
    new_mask[combined_centroids[:,0], combined_centroids[:,1]] = True
    markers, _ = ndi.label(new_mask)
    labeled_im = segmentation.watershed(-distance, markers, mask=candidate_im, watershed_line=True)
    # plt.imshow(labeled_im, cmap='jet')
    # plt.show()
    # print(np.unique(labeled_im))
    # split_arr = labeled_mask_splitting(labeled_im)
    bin_seg_im = (labeled_im>0).astype(bool)

    return bin_seg_im


def mask_merge_iou(base_im, mask_im, iou_threshold=0.5):
    """ Merge two sets of binary masks """
    template_size = base_im.shape
    # we need a template to add masks to and one to hold seg from previous images, they cannot
    # serve the same role
    # we assign to template seg im and use the base image to get masks from the original
    template_seg_im = np.copy(base_im)

    # here handle the mask merging
    labeled_mask_im = measure.label(mask_im)
    mask_regions = measure.regionprops(labeled_mask_im)
    
    for mask_region in tqdm(mask_regions):
        mask_coords = np.nonzero(labeled_mask_im==mask_region['label'])
        if np.sum(template_seg_im[mask_coords])<1:
            template_seg_im[mask_coords] = 1.0
        else:
            labeled_template = measure.label(base_im)
            add_mask = np.zeros(template_size)
            add_mask[mask_coords] = 1.0
            intersecting_mask_ids = np.unique((labeled_template*add_mask).flatten())[1:]
            print(f'Intersecting Mask IDs: {intersecting_mask_ids}')
            print(f'Candidate Mask ID: {mask_region.label}')

            candidate_iou_scores = []
            for mask_id in intersecting_mask_ids:
                bin_template = (labeled_template==mask_id)
                iou_score = get_iou(bin_template, add_mask)
                candidate_iou_scores.append(iou_score)
            
            print(f'IOU Scores: {candidate_iou_scores}')

            # if there is only a single intersection candidate then we just check if we should merge or watershed
            if len(intersecting_mask_ids)<2:
                if candidate_iou_scores[0]>iou_threshold:
                    # get merge coordinates and add to the image
                    merge_coords = np.nonzero(np.logical_or(add_mask.astype(bool), (labeled_template==intersecting_mask_ids[0])))
                    template_seg_im[merge_coords] = 1.0
                else:
                    # here we perform watershed and split the labeled masks
                    candidate_im = np.logical_or(add_mask.astype(bool), (labeled_template==intersecting_mask_ids[0]))
                    distance = ndi.distance_transform_edt(candidate_im)
                    # don't use the local minima or maxima to determine seeds for watershed
                    # instead just use the mask centers for existing masks
                    new_mask = np.zeros(distance.shape, dtype=bool)
                    add_centroid = measure.centroid(add_mask).astype(np.uint16).reshape(1,-1)
                    print(f'Orig Image ID: {intersecting_mask_ids[0]}')
                    intersecting_centroid = measure.centroid((labeled_template==intersecting_mask_ids[0])).astype(np.uint16).reshape(1,-1)
                    print(f'Unaltered centroid: {measure.centroid(add_mask)}')
                    sample_centroids = np.concatenate([add_centroid, intersecting_centroid], axis=0)
                    print(f'sample centroids: {sample_centroids}')
                    # print(sample_centroids)
                    # print(f'template image shape: {new_mask.shape} \n')
                    # new_mask[sample_centroids[:,0], sample_centroids[:,1]] = True
                    for centroid in sample_centroids:
                        new_mask[centroid[0], centroid[1]] = True
                    markers, _ = ndi.label(new_mask)
                    # this is the old code 
                    # coords = feature.peak_local_max(distance, footprint=(np.ones((3,3))), labels=candidate_im)
                    # new_mask = np.zeros(distance.shape, dtype=bool)
                    # new_mask[tuple(coords.T)] = True
                    # markers, _ = ndi.label(new_mask)
                    labels = segmentation.watershed(-distance, markers, mask=candidate_im)
                    # zero out the existing mask and add the new split to the template
                    current_mask_coords = np.nonzero((labeled_template==intersecting_mask_ids[0]))
                    template_seg_im[current_mask_coords] = 0.0
                    split_arr = labeled_mask_splitting(labels)
                    new_coords = np.nonzero((split_arr>0).astype(bool))
                    template_seg_im[new_coords] = 1.0

            # here we handle multiple masks
            else:
                multi_mask_template = np.zeros(template_size, dtype=np.uint8)
                new_coord_centroid_list = []
                # zero out all existing masks
                for mask_id in intersecting_mask_ids:
                    template_seg_im[template_seg_im==mask_id] = 0.0
                # if max iou is above the threshold we merge
                max_iou = np.max(candidate_iou_scores)
                if max_iou>iou_threshold:
                    merge_mask_id = intersecting_mask_ids[np.argmax(candidate_iou_scores)]
                    merge_coords = np.nonzero(np.logical_or(add_mask.astype(bool), (labeled_template==merge_mask_id)))
                    multi_mask_template[merge_coords] =  1
                    new_coord_centroid_list.append(np.array(measure.centroid(multi_mask_template)).astype(np.uint16).reshape(1,-1))
                else:
                    merge_mask_id=None
                
                for existing_mask in intersecting_mask_ids:
                    if existing_mask==merge_mask_id:
                        continue
                    sub_im = np.zeros(template_size, dtype=np.uint8)
                    candidate_coords = np.nonzero(np.logical_or(add_mask.astype(bool), (labeled_template==existing_mask)))
                    multi_mask_template[candidate_coords] = 1
                    sub_im[candidate_coords] = 1
                    new_coord_centroid_list.append(np.array(measure.centroid(sub_im)).astype(np.uint16).reshape(1,-1))

                # perform watershed on the new image
                distance = ndi.distance_transform_edt(multi_mask_template)
                new_centroids = np.concatenate(new_coord_centroid_list, axis=0)
                new_mask = np.zeros(distance.shape, dtype=bool)
                new_mask[new_centroids[:,0], new_centroids[:,1]] = True
                markers, _ = ndi.label(new_mask)
                # coords = feature.peak_local_max(distance, footprint=np.ones((3,3)), labels=multi_mask_template)
                # new_mask = np.zeros(distance.shape, dtype=bool)
                # new_mask[tuple(coords.T)] = True
                # markers, _ = ndi.label(new_mask)
                labels = segmentation.watershed(-distance, markers, mask=multi_mask_template)
                # get coordinates from the labels and label the new image
                split_arr = labeled_mask_splitting(labels)
                new_coords = np.nonzero(split_arr>0)
                template_seg_im[new_coords] = 1.0
    
    return template_seg_im


def merge_mask_centroid(base_im, mask_im, method='centroid', ioq_thresh=0.5):
    ''' Instead of merging masks based on IOU threshold we should merge if the center of one mask is within the area of another '''
    ''' Otherwise we use watershed to split them '''
    ''' funtion should be adjusted such that we have two methods for merging, centroid and ioq '''
    ''' for IOQ a threshold of 0.5 seeems reasonable'''
    template_size = base_im.shape
    template_seg_im = np.copy(base_im)

    labeled_mask_im = measure.label(mask_im, connectivity=1)
    mask_regions = measure.regionprops(labeled_mask_im)

    labeled_base = measure.label(base_im, connectivity=1)
    # plt.imshow(labeled_base, cmap='jet')
    # plt.show()


    for mask_region in tqdm(mask_regions):
        mask_coords = np.nonzero(labeled_mask_im==mask_region['label'])
        if np.sum(template_seg_im[mask_coords])<1:
            # before adding the mask, dilate the area around the mask
            # perform binary dilation in advance before asssigning
            check_template = np.zeros(template_size)
            check_template[mask_coords] = 1.0
            check_template = morphology.binary_dilation(check_template, morphology.square(5))
            coords_to_zero = np.nonzero(check_template)
            template_seg_im[coords_to_zero] = 0.0
            template_seg_im[mask_coords] = 1.0
        else:
            add_mask = np.zeros(template_size)
            add_mask[mask_coords] = 1.0
            intersecting_mask_ids = np.unique((labeled_base*add_mask).flatten())[1:]
            # print(f'Intersecting Mask IDs: {intersecting_mask_ids}')
            # print(f'Candidate Mask ID: {mask_region.label}')

            # first we can just write the case for one instance of intersection
            if len(intersecting_mask_ids)<2:
                # get area of the masks we are interested in
                add_mask_area = np.sum(add_mask)
                base_mask_area = np.sum(labeled_base==intersecting_mask_ids[0])
                if add_mask_area>base_mask_area:
                    search_mask = add_mask
                    search_centroid = measure.centroid((labeled_base==intersecting_mask_ids[0])).astype(np.uint16).reshape(1,-1)
                else:
                    search_mask = (labeled_base==intersecting_mask_ids[0])
                    search_centroid = measure.centroid(add_mask).astype(np.uint16).reshape(1,-1)
                # if the center of the mask we want to add is in our base image mask then we merge
                if method=='centroid':
                    if search_mask[search_centroid[:,0], search_centroid[:,1]]!=0.0:
                        merge_coords = np.nonzero(np.logical_or(add_mask.astype(bool), (labeled_base==intersecting_mask_ids[0])))
                        # perform binary dilation in advance before asssigning
                        check_template = np.zeros(template_size)
                        check_template[merge_coords] = 1.0
                        check_template = morphology.binary_dilation(check_template, morphology.square(5))
                        coords_to_zero = np.nonzero(check_template)
                        template_seg_im[coords_to_zero] = 0.0
                        template_seg_im[merge_coords] = 1.0
                    else:
                        # print(f'Performing Watershed')
                        # use the new functionalized code for watershed
                        combined_seg = mask_separation_watershed(add_mask, (labeled_base==intersecting_mask_ids[0]))
                        new_coords = np.nonzero(combined_seg)
                        # perform binary dilation in advance before asssigning
                        check_template = np.zeros(template_size)
                        check_template[new_coords] = 1.0
                        check_template = morphology.binary_dilation(check_template, morphology.square(5))
                        coords_to_zero = np.nonzero(check_template)
                        template_seg_im[coords_to_zero] = 0.0
                        template_seg_im[new_coords] = 1.0

                elif method=='ioq':
                    ioq_score = get_ioq(add_mask, labeled_base==intersecting_mask_ids[0])
                    if ioq_score>ioq_thresh:
                        merge_coords = np.nonzero(np.logical_or(add_mask.astype(bool), (labeled_base==intersecting_mask_ids[0])))
                        # perform binary dilation in advance before asssigning
                        check_template = np.zeros(template_size)
                        check_template[merge_coords] = 1.0
                        check_template = morphology.binary_dilation(check_template, morphology.square(5))
                        coords_to_zero = np.nonzero(check_template)
                        template_seg_im[coords_to_zero] = 0.0
                        template_seg_im[merge_coords] = 1.0
                    else:
                        # print(f'Performing Watershed')
                        # use the new functionalized code for watershed
                        combined_seg = mask_separation_watershed(add_mask, (labeled_base==intersecting_mask_ids[0]))
                        new_coords = np.nonzero(combined_seg)
                        # perform binary dilation in advance before asssigning
                        check_template = np.zeros(template_size)
                        check_template[new_coords] = 1.0
                        check_template = morphology.binary_dilation(check_template, morphology.square(3))
                        coords_to_zero = np.nonzero(check_template)
                        template_seg_im[coords_to_zero] = 0.0
                        template_seg_im[new_coords] = 1.0

                    # otherwise do watershed to separate the centers
                    # candidate_im = np.logical_or(add_mask.astype(bool), (labeled_base==intersecting_mask_ids[0]))
                    # distance = ndi.distance_transform_edt(candidate_im)
                    # new_mask = np.zeros(distance.shape, dtype=bool)
                    # add_centroid = measure.centroid(add_mask).astype(np.uint16).reshape(1,-1)
                    # # print(f'Orig Image ID: {intersecting_mask_ids[0]}')
                    # intersecting_centroid = measure.centroid((labeled_base==intersecting_mask_ids[0])).astype(np.uint16).reshape(1,-1)
                    # # print(f'Unaltered centroid: {measure.centroid(add_mask)}')
                    # sample_centroids = np.concatenate([add_centroid, intersecting_centroid], axis=0)
                    # # print(f'sample centroids: {sample_centroids}')
                    # for centroid in sample_centroids:
                    #     new_mask[centroid[0], centroid[1]] = True
                    # markers, _ = ndi.label(new_mask)
                    # labels = segmentation.watershed(-distance, markers, mask=candidate_im)
                    # # zero out the existing mask and add the new split to the template
                    # current_mask_coords = np.nonzero((labeled_base==intersecting_mask_ids[0]))
                    # template_seg_im[current_mask_coords] = 0.0
                    # split_arr = labeled_mask_splitting(labels)
                    # new_coords = np.nonzero((split_arr>0).astype(bool))
                    # perform binary dilation in advance before asssigning
                    # check_template = np.zeros(template_size)
                    # check_template[new_coords] = 1.0
                    # check_template = morphology.binary_dilation(check_template, morphology.square(3))
                    # coords_to_zero = np.nonzero(check_template)
                    # template_seg_im[coords_to_zero] = 0.0
                    # template_seg_im[new_coords] = 1.0

            # now handle multiple cases of intersection
            else:
                # print(f'Handling multiple instances of mask intersection...')
                multi_mask_template = np.zeros(template_size, dtype=np.uint8)
                # print(f'Original Multi Mask Template:')
                # plt.imshow(multi_mask_template)
                # plt.show()
                # zero out the locations in our template for the existing masks
                # for mask_id in intersecting_mask_ids:
                #     # perform dilation before zeroing here as well
                #     coords_to_remove = np.nonzero(labeled_base==mask_id)
                #     check_template = np.zeros(template_size)
                #     check_template[coords_to_remove] = 1.0
                #     check_template = morphology.binary_dilation(check_template, morphology.square(5))
                #     coords_to_zero = np.nonzero(check_template)
                #     template_seg_im[coords_to_zero] = 0.0
                # measure the areas of all of the base masks and the mask to add
                # consider the largest mask as the 'base' and fuse any masks that lie within its center
                # for all other masks perform watershed
                mask_areas = [np.sum(add_mask)]
                for mask_id in intersecting_mask_ids:
                    mask_areas.append(np.sum(labeled_base==mask_id))
                base_mask_index = np.argmax(mask_areas)
                if base_mask_index==0:
                    skip_add_mask=True
                    check_inter_mask=False
                    multi_mask_template += add_mask.astype(np.uint8)
                else:
                    skip_add_mask=False
                    check_inter_mask=True
                    corrected_index = base_mask_index-1
                    multi_mask_template += (labeled_base==intersecting_mask_ids[corrected_index])

                # print(f'Added Largest Mask:')
                # plt.imshow(multi_mask_template)
                # plt.show()

                # go through all non-largest masks and either merge or segment
                if not skip_add_mask:
                    if method=='centroid':
                        add_centroid = measure.centroid(add_mask).astype(np.uint16).reshape(1,-1)
                        if multi_mask_template[add_centroid[:,0], add_centroid[:,1]] != 0:
                            merge_coords = np.nonzero(np.logical_or(multi_mask_template.astype(bool), add_mask.astype(bool)))
                            current_mask_coords = np.nonzero((multi_mask_template>0))
                            check_template = np.zeros(template_size)
                            check_template[current_mask_coords] = 1.0
                            check_template = morphology.binary_dilation(check_template, morphology.square(5))
                            coords_to_zero = np.nonzero(check_template)
                            multi_mask_template[coords_to_zero] = 0.0

                            multi_mask_template[merge_coords] = 1
                        else:
                            # again replace watershed with the functionalized code
                            combined_seg = mask_separation_watershed(add_mask.astype(bool), (multi_mask_template>0).astype(bool))
                            # perform dilation on the mask removal the multi mask template
                            new_coords = np.nonzero(combined_seg)
                            # zero out existing mask in the template
                            current_mask_coords = np.nonzero((multi_mask_template>0))
                            check_template = np.zeros(template_size)
                            check_template[current_mask_coords] = 1.0
                            check_template = morphology.binary_dilation(check_template, morphology.square(5))
                            coords_to_zero = np.nonzero(check_template)
                            multi_mask_template[coords_to_zero] = 0.0
                            multi_mask_template[new_coords] = 1
                    elif method=='ioq':
                        ioq_score = get_ioq(add_mask, multi_mask_template>0)
                        if ioq_score>ioq_thresh:
                            merge_coords = np.nonzero(np.logical_or(multi_mask_template.astype(bool), add_mask.astype(bool)))
                            multi_mask_template[merge_coords] = 1
                        else:
                            # again replace watershed with the functionalized code
                            combined_seg = mask_separation_watershed(add_mask.astype(bool), (multi_mask_template>0).astype(bool))
                            new_coords = np.nonzero(combined_seg)
                            # zero out existing mask in the template
                            current_mask_coords = np.nonzero((multi_mask_template>0))
                            check_template = np.zeros(template_size)
                            check_template[current_mask_coords] = 1.0
                            check_template = morphology.binary_dilation(check_template, morphology.square(5))
                            coords_to_zero = np.nonzero(check_template)
                            multi_mask_template[coords_to_zero] = 0.0
                            multi_mask_template[new_coords] = 1
                
                # print(f'Added (add mask):')
                # plt.imshow(multi_mask_template)
                # plt.show()

                # now go through all intersecting masks and add them:
                for intersecting_mask_id in intersecting_mask_ids:
                    # skip the largest mask
                    if check_inter_mask:
                        if intersecting_mask_id==intersecting_mask_ids[corrected_index]:
                            continue
                    current_mask_im = (labeled_base==intersecting_mask_id)
                    current_centroid = measure.centroid(current_mask_im).astype(np.uint16).reshape(1,-1)
                    # again decide whether to work with ioq or centroids and then perform merging accordingly
                    if method=='centroid':
                        # if the centroid lies within the largest image merge - otherwise do watershed
                        if multi_mask_template[current_centroid[:,0], current_centroid[:,1]]!=0:
                            merge_coords = np.nonzero(np.logical_or(multi_mask_template, current_mask_im.astype(bool)))
                            current_mask_coords = np.nonzero((multi_mask_template>0))
                            check_template = np.zeros(template_size)
                            check_template[current_mask_coords] = 1.0
                            check_template = morphology.binary_dilation(check_template, morphology.square(5))
                            coords_to_zero = np.nonzero(check_template)
                            multi_mask_template[coords_to_zero] = 0.0
                            multi_mask_template[merge_coords] = 1
                        else:
                            # apply the watershed function here as well - do watershed merging between the current im and all masks in the template
                            labeled_template = measure.label(multi_mask_template, connectivity=1)
                            # print(f'Labeled Multi Mask Step')
                            # plt.imshow(labeled_template, cmap='jet')
                            # plt.show()
                            template_mask_props = measure.regionprops(labeled_template)
                            for mask_prop in template_mask_props:
                                # if the masks are not intersecting then continue
                                current_loop_mask = (labeled_template==mask_prop.label)
                                if mask_prop.label==1:
                                    incremented_mask = current_loop_mask+1
                                else:
                                    incremented_mask = current_loop_mask
                                if not np.any((current_mask_im*incremented_mask)>1):
                                    continue
                                # print(f'Handling intersection watershed')
                                # plt.imshow(current_mask_im)
                                # plt.show()
                                # plt.imshow(current_loop_mask)
                                # plt.show()
                                combined_seg = mask_separation_watershed(current_mask_im, current_loop_mask)
                                new_coords = np.nonzero(combined_seg)
                                current_mask_coords = np.nonzero((current_loop_mask>0))
                                check_template = np.zeros(template_size)
                                check_template[current_mask_coords] = 1.0
                                check_template = morphology.binary_dilation(check_template, morphology.square(5))
                                coords_to_zero = np.nonzero(check_template)
                                multi_mask_template[coords_to_zero] = 0.0
                                multi_mask_template[new_coords] = 1
                            # non loop code here    
                            # current_mask = mask_prop.label
                            # combined_seg = mask_separation_watershed(current_mask_im, (multi_mask_template>0))
                            # new_coords = np.nonzero(combined_seg)
                            # current_mask_coords = np.nonzero((multi_mask_template>0))
                            # check_template = np.zeros(template_size)
                            # check_template[current_mask_coords] = 1.0
                            # check_template = morphology.binary_dilation(check_template, morphology.square(5))
                            # coords_to_zero = np.nonzero(check_template)
                            # multi_mask_template[coords_to_zero] = 0.0
                            # multi_mask_template[new_coords] = 1
                    elif method=='ioq':
                        ioq_score = get_ioq(current_mask_im, multi_mask_template>0)
                        if ioq_score>ioq_thresh:
                            merge_coords = np.nonzero(np.logical_or(multi_mask_template, current_mask_im.astype(bool)))
                            current_mask_coords = np.nonzero((multi_mask_template>0))
                            check_template = np.zeros(template_size)
                            check_template[current_mask_coords] = 1.0
                            check_template = morphology.binary_dilation(check_template, morphology.square(5))
                            coords_to_zero = np.nonzero(check_template)
                            multi_mask_template[coords_to_zero] = 0.0
                            multi_mask_template[merge_coords] = 1
                        else:
                            # try new code here
                            labeled_template = measure.label(multi_mask_template, connectivity=1)
                            # print(f'Labeled Multi Mask Step')
                            # plt.imshow(labeled_template, cmap='jet')
                            # plt.show()
                            template_mask_props = measure.regionprops(labeled_template)
                            for mask_prop in template_mask_props:
                                # if the masks are not intersecting then continue
                                current_loop_mask = (labeled_template==mask_prop.label)
                                if mask_prop.label==1:
                                    incremented_mask = current_loop_mask+1
                                else:
                                    incremented_mask = current_loop_mask
                                if not np.any((current_mask_im*incremented_mask)>1):
                                    continue
                                # print(f'Handling intersection watershed')
                                # plt.imshow(current_mask_im)
                                # plt.show()
                                # plt.imshow(current_loop_mask)
                                # plt.show()
                                combined_seg = mask_separation_watershed(current_mask_im, current_loop_mask)
                                new_coords = np.nonzero(combined_seg)
                                current_mask_coords = np.nonzero((current_loop_mask>0))
                                check_template = np.zeros(template_size)
                                check_template[current_mask_coords] = 1.0
                                check_template = morphology.binary_dilation(check_template, morphology.square(5))
                                coords_to_zero = np.nonzero(check_template)
                                multi_mask_template[coords_to_zero] = 0.0
                                multi_mask_template[new_coords] = 1
                            # below is old code
                            # apply the watershed function here as well
                            # combined_seg = mask_separation_watershed(current_mask_im, (multi_mask_template>0))
                            # new_coords = np.nonzero(combined_seg)
                            # current_mask_coords = np.nonzero((multi_mask_template>0))
                            # check_template = np.zeros(template_size)
                            # check_template[current_mask_coords] = 1.0
                            # check_template = morphology.binary_dilation(check_template, morphology.square(5))
                            # coords_to_zero = np.nonzero(check_template)
                            # multi_mask_template[coords_to_zero] = 0.0
                            # multi_mask_template[new_coords] = 1
                        # old code below
                        # candidate_im = np.logical_or(current_mask_im.astype(bool), ((multi_mask_template)>0).astype(bool))
                        # distance = ndi.distance_transform_edt(candidate_im)
                        # new_mask = np.zeros(distance.shape, dtype=bool)
                        # multi_mask_centroid = measure.centroid(multi_mask_template).astype(np.uint16).reshape(1,-1)
                        # sample_centroids = np.concatenate([current_centroid,multi_mask_centroid], axis=0)
                        # for centroid in sample_centroids:
                        #     new_mask[centroid[0], centroid[1]] = True
                        # markers, _ = ndi.label(new_mask)
                        # labels = segmentation.watershed(-distance, markers, mask=candidate_im)
                        # # zero out the existing mask and add the new split to the template
                        # current_mask_coords = np.nonzero((multi_mask_template>0))
                        # multi_mask_template[current_mask_coords] = 0.0
                        # split_arr = labeled_mask_splitting(labels)
                        # new_coords = np.nonzero((split_arr>0).astype(bool))
                        # multi_mask_template[new_coords] = 1

                #     print(f'Added intersecting mask')
                #     plt.imshow(multi_mask_template)
                #     plt.show()


                # print(f'Final Multi Mask')
                # plt.imshow(multi_mask_template)
                # plt.show()

                multi_mask_coords = np.nonzero(multi_mask_template>0.0)
                # plt.imshow(multi_mask_template)
                # plt.show()
                # perform binary dilation in advance before asssigning
                check_template = np.zeros(template_size)
                check_template[multi_mask_coords] = 1.0
                check_template = morphology.binary_dilation(check_template, morphology.square(5))
                coords_to_zero = np.nonzero(check_template)
                template_seg_im[coords_to_zero] = 0.0
                template_seg_im += multi_mask_template
                # if the image is not binary throw an error
                if np.any(template_seg_im>1.0):
                    raise Exception("New masks are being added over old masks, resolve errors in existing merge function")

    return template_seg_im


def check_region_borders(
    src_im_slice, 
    region_mask_im, 
    row_border_size, 
    col_border_size,
    row_region_size,
    col_region_size):
    
    # we check the region boundaries and return a list of cell ids to drop
    
    # first border to check is top upper
    right_start = col_region_size-col_border_size
    bot_start = row_region_size-row_border_size
    top_border_src = src_im_slice[0:row_border_size, :]
    left_border_src = src_im_slice[:, 0:col_border_size]
    right_border_src = src_im_slice[:, right_start:right_start+col_border_size]
    bottom_border_src = src_im_slice[bot_start:bot_start+row_border_size, :]
    
    middle_section_top = src_im_slice[0:row_border_size, col_border_size:right_start]
    middle_section_bottom = src_im_slice[bot_start:bot_start+row_border_size, col_border_size:right_start]
    
    left_section_middle = src_im_slice[row_border_size:bot_start, 0:col_border_size]
    right_section_middle = src_im_slice[row_border_size:bot_start, right_start:right_start+col_border_size]
    
    top_left_corner = src_im_slice[0:row_border_size, 0:col_border_size]
    top_right_corner = src_im_slice[0:row_border_size, right_start:right_start+col_border_size]
    bottom_left_corner = src_im_slice[bot_start:bot_start+row_border_size, 0:col_border_size]
    bottom_right_corner = src_im_slice[bot_start:bot_start+row_border_size, right_start:right_start+col_border_size]
    
    # find cells to drop
    if np.sum(top_border_src)>0 and np.sum(middle_section_top)>0:
        top_cells = np.unique(region_mask_im[0:row_border_size, :])
    else:
        top_cells = np.array([])
    
    if np.sum(bottom_border_src)>0 and np.sum(middle_section_bottom)>0:
        bottom_cells = np.unique(region_mask_im[bot_start:bot_start+row_border_size, :])
    else:
        bottom_cells = np.array([])
    
    if np.sum(right_border_src)>0 and np.sum(right_section_middle)>0:
        right_cells = np.unique(region_mask_im[:, right_start:right_start+col_border_size])
    else:
        right_cells = np.array([])
        
    if np.sum(left_border_src)>0 and np.sum(left_section_middle)>0:
        left_cells = np.unique(region_mask_im[:, 0:col_border_size])

    else:
        left_cells = np.array([])
    
    # check all of the corners as well 
    if np.sum(top_left_corner)>0:
        top_left_cells = np.unique(region_mask_im[0:row_border_size, 0:col_border_size])
    else:
        top_left_cells = np.array([])
    
    if np.sum(top_right_corner)>0:
        top_right_cells = np.unique(region_mask_im[0:row_border_size, right_start:right_start+col_border_size])
    else:
        top_right_cells = np.array([])
        
    if np.sum(bottom_left_corner)>0:
        bottom_left_cells = np.unique(region_mask_im[bot_start:bot_start+row_border_size, 0:col_border_size])
    else:
        bottom_left_cells = np.array([])
    
    if np.sum(bottom_right_corner)>0:
        bottom_right_cells = np.unique(region_mask_im[bot_start:bot_start+row_border_size, right_start:right_start+col_border_size])
    else:
        bottom_right_cells = np.array([])
    
    
    combined_border_cells = np.concatenate([top_cells, bottom_cells, right_cells, left_cells, top_left_cells, top_right_cells, bottom_left_cells, bottom_right_cells])
    
    return np.unique(combined_border_cells)







