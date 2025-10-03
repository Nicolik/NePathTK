import torch
import cv2
from detection.io.openslide_reader import OpenslideReader
from prepare.config import DESIRED_OP_TISSUE
from semseg.model import get_unet_tissue
from semseg.data import hwc2chw


def tissue_segment(wsi: OpenslideReader, roi: tuple):
    device = 'cuda'
    x, y, w, h = roi
    wsi_resized = wsi.read_resolution(x, y, w, h, DESIRED_OP_TISSUE)
    unet = get_unet_tissue()
    t_image = torch.tensor(hwc2chw(wsi_resized)).unsqueeze(0).float().to(device)
    mask = unet.predict(t_image)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(contour) for contour in contours]

    return bboxes
