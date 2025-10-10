import os
import cv2
from PIL import Image

COLOR_BACKGROUND = "#c8c8c8"
COLOR_BIOPSY_TISSUE = "#aaaaaa"
COLOR_CORTEX = "#ef58e7"
COLOR_MEDULLA = "#0f140c"
COLOR_CAPSULE_OTHER = "#1edd9d"
COLOR_GLOMERULUS = "#00ff00"
COLOR_ARTERY = "#d91b50"
COLOR_ARTERIOLE = "#5b0ab0"
COLOR_IFTA = "#ff4433"

COLOR_BACKGROUND_TUPLE = (200, 200, 200)
COLOR_BIOPSY_TISSUE_TUPLE = (170, 170, 170)
COLOR_CORTEX_TUPLE = (239, 88, 231)
COLOR_MEDULLA_TUPLE = (15, 20, 12)
COLOR_CAPSULE_OTHER_TUPLE = (30, 221, 157)
COLOR_GLOMERULUS_TUPLE = (0, 255, 0)
COLOR_ARTERY_TUPLE = (217, 27, 80)
COLOR_ARTERIOLE_TUPLE = (91, 10, 176)
COLOR_IFTA_TUPLE = (255, 68, 51)

PALETTE_COLOR_CORTEX_MEDULLA_CAPSULE = (COLOR_BACKGROUND_TUPLE, COLOR_CORTEX_TUPLE, COLOR_MEDULLA_TUPLE, COLOR_CAPSULE_OTHER_TUPLE)
PALETTE_COLOR_GLOMERULUS_ARTERY_ARTERIOLE = (COLOR_BACKGROUND_TUPLE, COLOR_GLOMERULUS_TUPLE, COLOR_ARTERY_TUPLE, COLOR_ARTERIOLE_TUPLE)
PALETTE_COLOR_COMPLETE = (COLOR_BACKGROUND_TUPLE, COLOR_CORTEX_TUPLE, COLOR_MEDULLA_TUPLE, COLOR_CAPSULE_OTHER_TUPLE, COLOR_GLOMERULUS_TUPLE, COLOR_ARTERY_TUPLE, COLOR_ARTERIOLE_TUPLE)

CLASS_NAMES_CORTEX_MEDULLA_CAPSULE = ('Background', 'Cortex', 'Medulla', 'CapsuleOther')
CLASS_NAMES_GLOMERULUS_ARTERY_ARTERIOLE = ('Background', 'Glomerulus', 'Artery', 'Arteriole')
CLASS_NAMES_COMPLETE = ('Background', 'Cortex', 'Medulla', 'CapsuleOther', 'Glomerulus', 'Artery', 'Arteriole')

category_colors = {
    'BiopsyTissue': COLOR_BIOPSY_TISSUE,
    'Cortex': COLOR_CORTEX,
    'Medulla': COLOR_MEDULLA,
    'CapsuleOther': COLOR_CAPSULE_OTHER,
    'Glomerulus': COLOR_GLOMERULUS,
    'Artery': COLOR_ARTERY,
    'Arteriole': COLOR_ARTERIOLE
}

category_colors_tuple = {
    'BiopsyTissue': COLOR_BIOPSY_TISSUE_TUPLE,
    'Cortex': COLOR_CORTEX_TUPLE,
    'Medulla': COLOR_MEDULLA_TUPLE,
    'CapsuleOther': COLOR_CAPSULE_OTHER_TUPLE,
    'Glomerulus': COLOR_GLOMERULUS_TUPLE,
    'Artery': COLOR_ARTERY_TUPLE,
    'Arteriole': COLOR_ARTERIOLE_TUPLE
}


def get_coordinates_from_filename_(filename):
    coords = {}

    # BBOX_{xb}_{yb}_{wb}_{hb}
    if '__BBOX_' in filename:
        coords_str = filename.split('__BBOX_')[1].split('__')[0]
        coords['BBOX'] = list(map(int, map(float, coords_str.split('_'))))

    # ROI_{x}_{y}_{actual_h}_{actual_w}
    if '__ROI_' in filename:
        coords_str = filename.split('__ROI_')[1].split('__')[0]
        coords['ROI'] = list(map(int, coords_str.split('_')))

    # f"PAD_{padd_info['top']}_{padd_info['bottom']}_{padd_info['left']}_{padd_info['right']}"
    if '__PAD_' in filename:
        coords_str = filename.split('__PAD_')[1].split('.')[0]
        coords['PAD'] = list(map(int, coords_str.split('_')))

    return coords


def get_coordinates_from_filename(filename):
    coords_str = filename.split('__ROI_')[-1].split('.')[0]
    x1, y1, x2, y2 = map(int, coords_str.split('_'))
    return x1, y1, x2, y2


def get_wsi_name(wsi):
    wsi_basename = os.path.basename(wsi)
    if wsi_basename.endswith(('.ome.tif', '.ome.tiff')):
        wsi_name_noext = wsi_basename.rsplit('.', 2)[0]
    else:
        wsi_name_noext = os.path.splitext(wsi_basename)[0]
    return wsi_name_noext


def show_contours(mask, contours):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(mask_rgb, contours, -1, (255, 0, 0), 2)
    mask_pil = Image.fromarray(mask_rgb)
    mask_pil.show()


def point_from_pixels_to_physical(wsi, point, lvl=0):
    '''takes the coordinates of a point and converts it into physical coordinates.
    By default, lvl=0, meaning that the pixel coordinates correspond to the dimensions
    of the wsi at lvl 0'''
    if 'hamamatsu.XOffsetFromSlideCentre' in wsi.properties:
        offset_x = float(wsi.properties['hamamatsu.XOffsetFromSlideCentre'])
        offset_y = float(wsi.properties['hamamatsu.YOffsetFromSlideCentre'])
    else:
        offset_x = offset_y = 0
    mpp_x = float(wsi.properties['openslide.mpp-x'])
    mpp_y = float(wsi.properties['openslide.mpp-y'])
    width_lvl_0 = int(wsi.properties['openslide.level[{}].width'.format(lvl)])
    height_lvl_0 = int(wsi.properties['openslide.level[{}].height'.format(lvl)])

    slide_center_x = offset_x / (mpp_x * 1000 * (2 ** lvl))
    slide_center_y = offset_y / (mpp_y * 1000 * (2 ** lvl))

    x_0 = slide_center_x - width_lvl_0 / 2
    y_0 = slide_center_y - height_lvl_0 / 2

    x = (point[0] + x_0) * mpp_x * 1000 * (2 ** lvl)
    y = (point[1] + y_0) * mpp_y * 1000 * (2 ** lvl)

    return (x, y)


def wsi_physical_dimensions(wsi, resolution_factor=1):
    mpp_x = float(wsi.properties['openslide.mpp-x'])/resolution_factor
    mpp_y = float(wsi.properties['openslide.mpp-y'])/resolution_factor

    image_dimensions = (int(wsi.properties['openslide.level[0].width']), int(wsi.properties['openslide.level[0].height']))

    # Calcola le dimensioni fisiche in micrometri
    physical_dimensions_x = mpp_x * image_dimensions[0]
    physical_dimensions_y = mpp_y * image_dimensions[1]

    print(f"Dimensioni pixel X: {image_dimensions[0]} pixel")
    print(f"Dimensioni pixel Y: {image_dimensions[1]} pixel")
    print(f"Dimensioni fisiche Y: {physical_dimensions_y} µm")
    print(f"Dimensioni fisiche Y: {physical_dimensions_y} µm")

    return physical_dimensions_x, physical_dimensions_y
