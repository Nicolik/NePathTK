import numpy as np
import cv2
from detection.qupath.defs import MIN_AREA_MASK_UM2, MIN_AREA_BBOX_UM2


def multimask_cc_analysis(class_masks, magnification, undersampling, connectivity=4):
    output_class_masks = {}
    for cls in class_masks:
        if cls != "Background":
            binary_mask = class_masks[cls]

            closing_structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            opening_structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            if cls == "Glomerulus":
                after_opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, opening_structuring_element)
            else:
                after_closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_structuring_element)
                after_opening = cv2.morphologyEx(after_closing, cv2.MORPH_OPEN, opening_structuring_element)

            output = cv2.connectedComponentsWithStats(after_opening, connectivity, cv2.CV_32S)
            num_labels, label_ids, stats, centroid = output
            # stats has num_labels x 5
            # centroid has num_labels x 2
            # label_ids is the same shape as binary_mask, with values from 0 to num_labels-1

            leftmost_x = stats[:, cv2.CC_STAT_LEFT]
            topmost_y = stats[:, cv2.CC_STAT_TOP]
            width = stats[:, cv2.CC_STAT_WIDTH]
            height = stats[:, cv2.CC_STAT_HEIGHT]
            area = stats[:, cv2.CC_STAT_AREA]
            small_cnt = 0

            min_area = MIN_AREA_MASK_UM2[cls]
            min_area_bbox = MIN_AREA_BBOX_UM2[cls]

            for i in range(num_labels):
                px_to_um_factor = 0.25 * 40 / magnification * undersampling
                label_area = area[i] * px_to_um_factor
                label_bbox_area = width[i] * height[i] * px_to_um_factor

                print(f"[multimask_cc_analysis] px_to_um_factor: {px_to_um_factor}, "
                      f"label_area: {label_area}, label_bbox_area: {label_bbox_area}")

                if label_area < min_area:
                    print(f"[multimask_cc_analysis] Class: {cls}, mask, area={label_area} < min_area={min_area}")
                    small_cnt += 1
                    # label_ids[label_area == i] = 0
                elif label_bbox_area < min_area_bbox:
                    print(f"[multimask_cc_analysis] Class: {cls}, bbox, area={label_bbox_area} < min_area={min_area_bbox}")
                    small_cnt += 1
                    # label_ids[label_area == i] = 0

            print(f"[multimask_cc_analysis] Class: {cls}, Small: {small_cnt}/{num_labels}")
            label_ids[label_ids > 0] = 1
            output_class_masks[cls] = label_ids.astype(np.uint8)

    return output_class_masks
