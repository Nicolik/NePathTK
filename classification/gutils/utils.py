import numpy as np
import cv2
import torch
import argparse


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def vconcat(im_list_1d):
    return cv2.vconcat(im_list_1d)


def hconcat(im_list_1d):
    return cv2.hconcat(im_list_1d)


def get_proper_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.has_mps:
        return torch.device("mps")
    else:
        torch.device("cpu")


def mask_to_polygonal(mask, xy=False):
    mask = mask.astype(np.uint8)
    mask_contours, hier_rec = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polygonal_xy = mask_contours[0]
    polygonal_xy = np.squeeze(polygonal_xy, axis=1)
    polygonal_yx = np.zeros(polygonal_xy.shape)
    polygonal_yx[:, 0] = polygonal_xy[:, 1].copy()
    polygonal_yx[:, 1] = polygonal_xy[:, 0].copy()
    return polygonal_xy if xy else polygonal_yx


def get_bbox_from_polygonal(poly):
    ymin = int(poly[:, 0].min())
    ymax = int(poly[:, 0].max())
    xmin = int(poly[:, 1].min())
    xmax = int(poly[:, 1].max())
    return ymin, ymax, xmin, xmax


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
