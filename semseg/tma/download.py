import os.path

from huggingface_hub import hf_hub_download

from definitions import ROOT_DIR
from config_semseg import HF_TOKEN_TMA
from semseg.mm.config import MODELS_DICT_MMSEG_AAG, MODELS_DICT_MMSEG_CMC, MODELS_DICT_MMSEG_IFTA


def download_semseg_model(segmentation_type, segmentation_net, backbone):
    model_dir = f"semseg/lightning_logs/{segmentation_type}/{segmentation_net}_{backbone}/"
    filename = f"{model_dir}model_final.pt"
    local_dir = ROOT_DIR
    local_download_dir = os.path.join(local_dir, model_dir)
    return _download_local(filename, local_download_dir)


def download_mmseg_model(segmentation_type, segmentation_net, backbone=None, microscopy_type="paraffin"):
    models_dict = {}
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
    return _download_local(filename, local_download_dir)


def _download_local(filename, local_download_dir):
    os.makedirs(local_download_dir, exist_ok=True)
    print(f"Creating {local_download_dir}")
    print(f"Attempting to download {filename}")
    return hf_hub_download(
        repo_id="MESCnn/TMA",
        filename=filename,
        token=HF_TOKEN_TMA,
        local_dir=ROOT_DIR,
        local_dir_use_symlinks=False,
        force_download=True,
    )


def download_cmc_semseg_model(segmentation_net, backbone):
    return download_semseg_model("cmc", segmentation_net, backbone)


def download_aag_semseg_model(segmentation_net, backbone):
    return download_semseg_model("aag", segmentation_net, backbone)
