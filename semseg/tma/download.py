import os.path
from download import download_local
from definitions import ROOT_DIR
from semseg.mm.config import MODELS_DICT_MMSEG_AAG, MODELS_DICT_MMSEG_CMC, MODELS_DICT_MMSEG_IFTA
from semseg.tma.config import MODELS_DICT_LIGHTNING_AAG, MODELS_DICT_LIGHTNING_CMC


REPO_ID_SEMSEG_TMA = {
    "paraffin": "NePathTK/tma-segmentation",
    "vivascope": "NePathTK/vivascope-segmentation",
}


def download_semseg_model(segmentation_type, segmentation_net, backbone, microscopy_type="paraffin"):
    assert segmentation_type in ("cmc", "aag"), f"Segmentation type not supported: {segmentation_type}"
    if segmentation_type == "cmc":
        models_dict = MODELS_DICT_LIGHTNING_CMC
    elif segmentation_type == "aag":
        models_dict = MODELS_DICT_LIGHTNING_AAG

    model_dir = f"semseg/lightning/weights/{microscopy_type}/"
    filename = f"{model_dir}{models_dict[segmentation_net]}.pt"
    local_dir = ROOT_DIR
    local_download_dir = os.path.join(local_dir, model_dir)
    repo_id = REPO_ID_SEMSEG_TMA[microscopy_type]
    return download_local(filename, local_download_dir, repo_id)


def download_mmseg_model(segmentation_type, segmentation_net, backbone=None, microscopy_type="paraffin"):
    models_dict = {}
    assert segmentation_type in ("cmc", "aag", "ifta"), f"Segmentation type not supported: {segmentation_type}"
    if segmentation_type == "cmc":
        models_dict = MODELS_DICT_MMSEG_CMC
    elif segmentation_type == "aag":
        models_dict = MODELS_DICT_MMSEG_AAG
    elif segmentation_type == "ifta":
        models_dict = MODELS_DICT_MMSEG_IFTA

    model_dir = f"semseg/mm/weights/{microscopy_type}/"
    filename = f"{model_dir}{models_dict[segmentation_net]}.pth"
    local_dir = ROOT_DIR
    local_download_dir = os.path.join(local_dir, model_dir)
    repo_id = REPO_ID_SEMSEG_TMA[microscopy_type]
    return download_local(filename, local_download_dir, repo_id)


def download_cmc_semseg_model(segmentation_net, backbone):
    return download_semseg_model("cmc", segmentation_net, backbone)


def download_aag_semseg_model(segmentation_net, backbone):
    return download_semseg_model("aag", segmentation_net, backbone)
