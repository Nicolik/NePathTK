import os
from definitions import HF_TOKEN_TMA, ROOT_DIR
from huggingface_hub import hf_hub_download


def download_local(filename, local_download_dir, repo_id):
    os.makedirs(local_download_dir, exist_ok=True)
    print(f"Creating {local_download_dir}")
    print(f"Attempting to download {filename}")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=HF_TOKEN_TMA,
        local_dir=ROOT_DIR,
        local_dir_use_symlinks=False,
        force_download=True,
    )
