import os
from definitions import ROOT_DIR, DATASET_DIR


def get_logs_path(root_dir=None):
    root_dir = ROOT_DIR if root_dir is None else root_dir
    logs_dir = os.path.join(root_dir, 'classification', 'logs', 'tma_instance')
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def get_vis_path(root_dir=None):
    root_dir = ROOT_DIR if root_dir is None else root_dir
    vis_dir = os.path.join(root_dir, 'classification', 'vis', 'tma_instance')
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir


def get_root_dir():
    return DATASET_DIR


def rectify_ext_ids(ext_ids, base_dir):
    corrected = []
    base_dir = os.path.abspath(base_dir)

    for eid in ext_ids:
        eid_abs = os.path.abspath(eid)
        if not eid_abs.startswith(base_dir):
            eid_abs = os.path.join(base_dir, os.path.basename(eid))
        corrected.append(eid_abs)

    return corrected
