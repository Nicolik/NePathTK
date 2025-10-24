import os
from enum import IntEnum

from definitions import ROOT_DIR

MIN_AREA_BBOX_JSON = {
    'Glomerulus': 5000,
    'Artery': 5000,
    'Arteriole': 2500,
    'IFTACortex': 2500,
    'Cortex': 5000,
    'CapsuleOther': 5000,
    'Medulla': 5000,
    'BiopsyTissue': 5000
}

MIN_AREA_MASK_UM2 = {
    'Glomerulus': 1000,
    'Artery': 1000,
    'Arteriole': 250,
    'IFTACortex': 250,
    'Cortex': 1000,
    'CapsuleOther': 1000,
    'Medulla': 1000,
    'BiopsyTissue': 1000
}

MIN_AREA_BBOX_UM2 = {
    'Glomerulus': 1600,
    'Artery': 1600,
    'Arteriole': 400,
    'IFTACortex': 400,
    'Cortex': 1600,
    'CapsuleOther': 1600,
    'Medulla': 1600,
    'BiopsyTissue': 1600
}


class PathMESCnn:
    TILE = os.path.join(ROOT_DIR, "detection", "qupath", "tile.py")
    PKL2QU = os.path.join(ROOT_DIR, "detection", "qupath", "pkl2qu.py")
    QU2JSON = os.path.join(ROOT_DIR, "detection", "qupath", "qu2json.py")
    QU2ROI = os.path.join(ROOT_DIR, "detection", "qupath", "qu2roi.py")
    JSON2EXP = os.path.join(ROOT_DIR, "detection", "qupath", "json2exp.py")

    # Semantic segmentation
    TILE_BBOX = os.path.join(ROOT_DIR, "detection", "qupath", "tile_bbox.py")
    SEMSEG_CMC = os.path.join(ROOT_DIR, "semseg", "tma", "semseg_cmc.py")
    SEMSEG_CMC_MM = os.path.join(ROOT_DIR, "semseg", "tma", "semseg_cmc_mm.py")
    SEMSEG_AAG = os.path.join(ROOT_DIR, "semseg", "tma", "semseg_aag.py")
    SEMSEG_AAG_MM = os.path.join(ROOT_DIR, "semseg", "tma", "semseg_aag_mm.py")
    SEMSEG_IFTA_MM = os.path.join(ROOT_DIR, "semseg", "tma", "semseg_ifta_mm.py")
    AGGREGATE_WSI_BIN_BBOX = os.path.join(ROOT_DIR, "semseg", "qupath", "aggregate_wsi_bin_bbox.py")
    POST_PROCESSING_MASK = os.path.join(ROOT_DIR, "semseg", "qupath", "post_processing_mask.py")
    EXPORT = os.path.join(ROOT_DIR, "semseg", "qupath", "export.py")


class GlomerulusDetection(IntEnum):
    BACKGROUND = 0
    GLOMERULUS = 1


class MulticlassSegmentation(IntEnum):
    BACKGROUD = 0
    GLOMERULUS = 1
    ARTERY = 2
    ARTERIOLE = 3
    CORTEX = 4
    MEDULLA = 5
    CAPSULE = 6


CATEGORY_ID = {
    'Background': MulticlassSegmentation.BACKGROUD,
    'Glomerulus': MulticlassSegmentation.GLOMERULUS,
    'Artery': MulticlassSegmentation.ARTERY,
    'Arteriole': MulticlassSegmentation.ARTERIOLE,
    'Cortex': MulticlassSegmentation.CORTEX,
    'Medulla': MulticlassSegmentation.MEDULLA,
    'Capsule': MulticlassSegmentation.CAPSULE,
}
