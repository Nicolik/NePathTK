import os
import numpy as np
import cv2
from PIL import Image
from shapely import Polygon


def multimask2polygon(class_masks, undersampling):
    polygons_dict = {}

    for cls in class_masks:
        if cls != "Background":
            binary_mask = class_masks[cls]
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            upsampled_contours = [contour * undersampling for contour in contours]

            class_polygons = []
            for contour in upsampled_contours:
                if len(contour) >= 4 and len(np.unique(contour, axis=0)) >= 4:
                    polygon = Polygon(shell=contour.squeeze())
                    class_polygons.append(polygon)
                else:
                    print(f"Contour len < 4 for {cls}")

            polygons_dict[cls] = class_polygons

    return polygons_dict
