import numpy as np
import cv2


def poly2mask(poly, mask_size):
    mask = np.zeros(mask_size, dtype=np.uint8)
    coords = poly.exterior.coords.xy
    xx, yy = coords
    c_list = np.array([(x, y) for x, y in zip(xx, yy) if 0 <= x < mask_size[1] and 0 <= y < mask_size[0]], dtype=np.int32)
    if len(c_list) > 0:
        mask = cv2.fillPoly(mask, [c_list], 1)
    return mask
