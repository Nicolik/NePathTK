import os
import cv2
import torch
import ttach as tta
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from semseg.model import BinarySegmentation, MulticlassSegmentation
from semseg.paths import get_final_model_bin, get_final_model_cmc, get_final_model_aag, get_ckpt_cmc, get_ckpt_aag
from semseg.tma.download import download_cmc_semseg_model, download_aag_semseg_model


def model_to_segmenter(model, use_tta):
    model.eval()
    if use_tta:
        model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    return model


def get_cortex_medulla_capsule_checkpoint(arch="unet", enc_name="resnet34", device="cuda", use_tta=True):
    ckpt_cmc = get_ckpt_cmc(arch, enc_name)
    subfiles = [f for f in os.listdir(ckpt_cmc) if f.endswith('.ckpt')]
    ckpt_path = os.path.join(ckpt_cmc, subfiles[0])
    kwargs = {'arch': arch, 'encoder_name': enc_name, 'in_channels': 3, 'out_classes': 4}
    model = MulticlassSegmentation.load_from_checkpoint(ckpt_path, **kwargs).to(device)
    return model_to_segmenter(model, use_tta)


def get_cortex_medulla_capsule_segmenter(arch="unet", enc_name="resnet34", device="cuda", use_tta=True):
    cmc_segmenter_path = get_final_model_cmc(arch, enc_name)
    if not os.path.exists(cmc_segmenter_path):
        cmc_segmenter_path = download_cmc_semseg_model(arch, enc_name)
    state_dict = torch.load(cmc_segmenter_path)
    model = MulticlassSegmentation(arch, enc_name, in_channels=3, out_classes=4).to(device)
    model.load_state_dict(state_dict)
    return model_to_segmenter(model, use_tta)


def get_glomerulus_artery_arteriole_checkpoint(arch="unet", enc_name="resnet34", device="cuda", use_tta=True):
    ckpt_aag = get_ckpt_aag(arch, enc_name)
    subfiles = [f for f in os.listdir(ckpt_aag) if f.endswith('.ckpt')]
    ckpt_path = os.path.join(ckpt_aag, subfiles[0])
    kwargs = {'arch': arch, 'encoder_name': enc_name, 'in_channels': 3, 'out_classes': 4}
    model = MulticlassSegmentation.load_from_checkpoint(ckpt_path, **kwargs).to(device)
    return model_to_segmenter(model, use_tta)


def get_glomerulus_artery_arteriole_segmenter(arch="unet", enc_name="resnet34", device="cuda", use_tta=True):
    aag_segmenter_path = get_final_model_aag(arch, enc_name)
    if not os.path.exists(aag_segmenter_path):
        aag_segmenter_path = download_aag_semseg_model(arch, enc_name)
    state_dict = torch.load(aag_segmenter_path)
    model = MulticlassSegmentation(arch, enc_name, in_channels=3, out_classes=4).to(device)
    model.load_state_dict(state_dict)
    return model_to_segmenter(model, use_tta)


def area_filter(mask, area_threshold):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_threshold:
            mask[labels == i] = 0


def detect_and_solve_touching_objects(mask, min_distance=None, num_peaks=3):
    num_labels, labels = cv2.connectedComponents(mask)
    print(f"[START] There are {num_labels} connected components")
    final_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        print(f"Component: {i} out of {num_labels}")
        mask_cv = (labels == i).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask_cv)
        print(f"Rect: {(x, y, w, h)}")
        roi = mask_cv[y:y + h, x:x + w]

        distance = ndi.distance_transform_edt(roi)
        coords = peak_local_max(distance, min_distance=40, labels=roi, num_peaks=num_peaks)
        mask_np = np.zeros(distance.shape, dtype=bool)
        mask_np[tuple(coords.T)] = True
        markers, _ = ndi.label(mask_np)
        labels_ws = watershed(-distance, markers, mask=roi)
        unique_labels = np.unique(labels_ws)

        if len(unique_labels) == 1:
            labels_ws = roi
        iter_len = max(2, len(unique_labels))

        for l in range(1, iter_len):
            level_mask = (labels_ws == l).astype(np.uint8)
            level_mask = cv2.erode(level_mask, kernel=np.ones((3, 3), np.uint8), iterations=3)
            final_mask[y:y+h, x:x+w] += level_mask

    num_labels_final, _ = cv2.connectedComponents(final_mask)
    print(f"[END] From {num_labels}, now there are {num_labels_final} connected components")

    return final_mask
