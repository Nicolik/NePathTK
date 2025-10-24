import os.path
from download import download_local
from definitions import ROOT_DIR
from classification.config import MODELS_DICT_CLASSIFICATION_TMA_ART2, MODELS_DICT_CLASSIFICATION_TMA_GLOM


REPO_ID_CLASSIFICATION_TMA = {
    "tma_instance": "NePathTK/tma-classification",
    "vivascope": "NePathTK/vivascope-classification",
}


def download_tma_classification_model(compartment_type, classification_net):
    task_type = "tma_instance"
    assert compartment_type in ("art2", "glom"), f"Compartment type not supported: {compartment_type}"
    if compartment_type == "art2":
        models_dict = MODELS_DICT_CLASSIFICATION_TMA_ART2
    elif compartment_type == "glom":
        models_dict = MODELS_DICT_CLASSIFICATION_TMA_GLOM

    model_dir = f"classification/weights/{task_type}/"
    filename_model = f"{model_dir}{models_dict[classification_net]}.pth"
    filename_metrics = f"{model_dir}metrics_trends_{models_dict[classification_net]}.pkl"
    local_dir = ROOT_DIR
    local_download_dir = os.path.join(local_dir, model_dir)
    repo_id = REPO_ID_CLASSIFICATION_TMA[task_type]
    downloaded_model = download_local(filename_model, local_download_dir, repo_id)
    downloaded_metrics = download_local(filename_metrics, local_download_dir, repo_id)
    return downloaded_model, downloaded_metrics
