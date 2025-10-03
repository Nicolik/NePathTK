import json
import os

import cv2
import numpy as np
from PIL import Image
from skimage.measure import label
from shapely.geometry import Polygon

from semseg.parser import build_aggregate_semseg_parser
from semseg.qupath.utils_aggr import normalize_mask, get_aggregate_masks

if __name__ == '__main__':
    parser = build_aggregate_semseg_parser()
    args = parser.parse_args()
    wsi_name = args.wsi
    path_wsi_masks = args.wsi_masks
    path_masks = args.masks

    with open(args.json_metadata, 'r') as fp:
        dict_metadata = json.load(fp)

    dimension = (dict_metadata['Y'], dict_metadata['X'])

    # Undersample original dimensions
    new_dimensions = tuple(int(dim / args.undersampling) for dim in dimension)
    overall_mask = np.zeros(new_dimensions, dtype=np.uint8)
    print(f"Creating mask with shape: {overall_mask.shape}")

    # Get masks of the tiles
    masks_directory = os.path.join(path_masks, wsi_name)

    aggregate_masks = get_aggregate_masks(args.segmentation_type)

    class_masks, count_masks, sum_masks, average_masks = aggregate_masks(masks_directory, new_dimensions, args.undersampling)

    # CREATING A BINARY TISSUE MASK
    if args.segmentation_type == "cmc":
        tissue_mask = np.zeros(new_dimensions, dtype=np.uint8)

    # Export aggregated masks as images
    for cls in average_masks:
        if cls != "Background":
            average_mask = average_masks[cls]
            final_mask_name = f"{wsi_name}_{cls}_mask.png"
            final_mask_path = os.path.join(path_wsi_masks, final_mask_name)
            Image.fromarray(average_mask).save(final_mask_path)

            count_mask = normalize_mask(count_masks[cls])
            final_mask_name = f"{wsi_name}_{cls}_mask_count.png"
            final_mask_path = os.path.join(path_wsi_masks, final_mask_name)
            Image.fromarray(count_mask).save(final_mask_path)

            sum_mask = normalize_mask(sum_masks[cls])
            final_mask_name = f"{wsi_name}_{cls}_mask_sum.png"
            final_mask_path = os.path.join(path_wsi_masks, final_mask_name)
            Image.fromarray(sum_mask).save(final_mask_path)

            if args.segmentation_type == "cmc":
                tissue_mask[average_mask > 0] = 255

    if args.segmentation_type == "cmc":
        # Export tissue bounding boxes
        # Contours
        contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        # Connected components labeling
        labeled_mask = label(tissue_mask)

        upsampled_contours = [contour * args.undersampling for contour in contours]
        polygons = {
            'BiopsyTissue': []
        }

        for contour in upsampled_contours:
            if len(contour) >= 4 and len(np.unique(contour, axis=0)) >= 4:
                polygon = Polygon(shell=contour.squeeze())
                polygons['BiopsyTissue'].append(polygon)

                # Calcola bounding box
                minx, miny, maxx, maxy = polygon.bounds
                bounding_boxes.append({
                    "top_left": [minx, miny],
                    "bottom_right": [maxx, maxy]
                })
            else:
                print("Contour len < 4")

        dict_metadata["bounding_boxes"] = bounding_boxes
        with open(args.json_metadata, 'w') as fp:
            json.dump(dict_metadata, fp)
            print('bbox saved')
