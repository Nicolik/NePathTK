import numpy as np
import cv2


def hconcat_with_border(images, border_thickness, border_color=(0, 0, 0)):
    images_with_borders = add_borders(images, border_thickness, border_color=border_color, vertical=False)
    return cv2.hconcat(images_with_borders)


def vconcat_with_border(images, border_thickness, border_color=(0, 0, 0)):
    images_with_borders = add_borders(images, border_thickness, border_color=border_color, vertical=True)
    return cv2.vconcat(images_with_borders)


def add_borders(images, border_thickness, border_color=(0, 0, 0), vertical=False):
    if vertical:
        width = images[0].shape[1]
        border = np.full((border_thickness, width, 3), border_color, dtype=np.uint8)
    else:
        height = images[0].shape[0]
        border = np.full((height, border_thickness, 3), border_color, dtype=np.uint8)

    images_with_borders = []
    for i in range(len(images)):
        images_with_borders.append(images[i])
        if i != len(images) - 1:
            images_with_borders.append(border)
    return images_with_borders


def mask_to_color(grayscale_mask, palette_dict):
    color_mask = np.zeros((grayscale_mask.shape[0], grayscale_mask.shape[1], 3), dtype=np.uint8)
    for class_id in palette_dict['mask_values']:
        class_mask = grayscale_mask == class_id
        color_mask[class_mask] = palette_dict['palette'][class_id]

    return color_mask


def add_contours(image, mask, palette_dict, thickness=2):
    for class_id in palette_dict['mask_values']:
        class_mask = mask == class_id
        class_mask_uint8 = np.uint8(255 * class_mask)
        contours, _ = cv2.findContours(class_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, palette_dict['contours'][class_id], thickness=thickness)


palette_dict_compartment = {
    "AAG": dict(
        mask_values=[0, 1, 2, 3],
        classes=('Background', 'Glomerulus', 'Artery', 'Arteriole'),
        palette=[[200, 200, 200], [0, 255, 0],  [217, 27, 80], [91, 10, 176]],
        contours=[[200, 200, 200], [0, 150, 0],  [150, 0, 0], [0, 0, 150]]
    ),
    "CMC": dict(
        mask_values=[0, 1, 2, 3],
        classes=('Background', 'Cortex', 'Medulla', 'CapsuleOther'),
        palette=[[200, 200, 200], [239, 88, 231],  [15, 20, 12], [30, 221, 157]],
        contours=[[200, 200, 200], [255, 100, 255],  [10, 10, 10], [0, 255, 255]],
    ),
    "IFTA": dict(
        mask_values=[0, 1],
        classes=('Background', 'IFTACortex'),
        palette=[[200, 200, 200], [255, 68, 51]],
        contours=[[200, 200, 200], [255, 50, 50]]
    )
}
