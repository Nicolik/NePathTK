import json
import os
from enum import Enum
import cv2
import numpy as np

from semseg.qupath.utils import get_wsi_name


def tile_region(roi, tile_size, tile_stride, small_tiles=True):
    x, y, w, h = roi
    size_x, size_y = tile_size
    stride_x, stride_y = tile_stride

    size_x = min(size_x, w)
    size_y = min(size_y, h)

    tiles = []
    for xx in range(x, x + w, stride_x):
        size_x_ = x + w - xx if small_tiles and xx + size_x > x + w else size_x
        for yy in range(y, y + h, stride_y):
            size_y_ = y + h - yy if small_tiles and yy + size_y > y + h else size_y
            tiles.append([xx, yy, size_x_, size_y_])
    return tiles


def find_nearest(array, value):
    array = np.asarray(array)
    print(f"[find_nearest] array: {[round(a, 2) for a in array]}, value: {value}")
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def find_nearest_greater(array, value):
    EPS_ZERO = 0.01
    array = np.asarray(array)
    print(f"[find_nearest_greater] array: {[round(a, 2) for a in array]}, value: {value}")
    valid_idx = np.where(array-value+EPS_ZERO >= 0)[0]
    idx = valid_idx[array[valid_idx].argmin()]
    return idx, array[idx]


def crop(x, y, w, h, X, Y):
    x = min(max(x, 0), X-1)
    y = min(max(y, 0), Y-1)
    w = X - x if x + w > X else w
    h = Y - y if y + h > Y else h
    return x, y, w, h


def PIL2cv2(PIL_image):
    """ Convert PIL image to cv2 image

    :param PIL_image: original PIL image
    :return: cv2 image
    """
    PIL_image = PIL_image.convert('RGB')
    opencv_img = np.array(PIL_image)
    return opencv_img


def is_foreground(image_np_rgb, THRESHOLD_BINARY=100, THRESHOLD_FOREGROUND=0.3):
    image_np_gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)
    ret, thresh_np = cv2.threshold(image_np_gray, THRESHOLD_BINARY, 255, cv2.THRESH_BINARY_INV)
    image_size = image_np_gray.size
    foreground_pixels = thresh_np.sum() // 255
    ratio_foreground = foreground_pixels / image_size
    return ratio_foreground > THRESHOLD_FOREGROUND


def is_black(image_np_rgb):
    image_np_gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)
    return np.mean(image_np_gray == 0) == 1


def is_not_too_small(image_np):
    if image_np is not None:
        h, w, c = image_np.shape
        if h > 64 and w > 64:
            return True
    return False


class ReaderType(Enum):
    NONE = 0
    SVS  = 1
    NDPI = 2
    SCN  = 3
    MRXS = 4
    TIFF = 5
    OME_TIFF = 6
    CZI = 7


def dir_name_from_wsi(path):
    collate = 2 if ('.ome.tif' in path or '.ome.tiff' in path) else 1
    return '.'.join(path.split('.')[:-collate])


def init_data_dict_segm():
    data_dict = init_data_dict()
    data_dict['maskname'] = []
    return data_dict


def init_data_dict():
    return {
        'image-id': [],
        'filename': [],
        'path-to-wsi': [],
        'ext': [],
        's': [],
        'x': [],
        'y': [],
        'w': [],
        'h': []
    }


def bbox2xywh(bbox, multiple_of=32):
    x, y = bbox["top_left"]
    w = bbox["bottom_right"][0] - bbox["top_left"][0]
    h = bbox["bottom_right"][1] - bbox["top_left"][1]
    if multiple_of:
        rem_w = w % multiple_of
        rem_h = h % multiple_of
        w = w - rem_w + multiple_of
        h = h - rem_h + multiple_of
    roi = (int(x), int(y), int(w), int(h))
    return roi


def get_wsi_json_info(wsi, path_json):
    wsi_name = get_wsi_name(wsi) + '.json'
    json_metadata = os.path.join(path_json, wsi_name)
    with open(json_metadata, 'r') as fp:
        dict_metadata = json.load(fp)
    return dict_metadata


def get_wsi_multi_json_info(wsi, path_json):
    wsi_name = get_wsi_name(wsi)
    json_files = os.listdir(path_json)
    json_files = [os.path.join(path_json, f) for f in json_files if f.endswith('.json') and wsi_name in f]
    dict_list = []
    for json_file in json_files:
        with open(json_file, 'r') as fp:
            dict_metadata = json.load(fp)
        dict_list.append(dict_metadata)
    return dict_list
