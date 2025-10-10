import os
from definitions import ROOT_DIR
from semseg.tma.config import MODELS_DICT_LIGHTNING_CMC, MODELS_DICT_LIGHTNING_AAG

LIGHTNING_LOGS_DIR = os.path.join(ROOT_DIR, 'semseg', 'lightning_logs')
LOGS_DIR_BINARY = os.path.join(LIGHTNING_LOGS_DIR, 'bin')
LOGS_DIR_CORTEX_MEDULLA_CAPSULE = os.path.join(LIGHTNING_LOGS_DIR, 'cmc')
LOGS_DIR_ARTERIOLE_ARTERY_GLOMERULUS = os.path.join(LIGHTNING_LOGS_DIR, 'aag')


def get_figure_dir():
    return os.path.join(LIGHTNING_LOGS_DIR, 'figures')


def get_logs_bin(arch, enc_name):
    return os.path.join(LOGS_DIR_BINARY, f"{arch.lower()}_{enc_name.lower()}")


def get_ckpt_bin(arch, enc_name):
    return os.path.join(get_logs_bin(arch, enc_name), 'ckpt')


def get_final_model_bin(arch, enc_name):
    return os.path.join(get_logs_bin(arch, enc_name), 'model_final.pt')


def get_logs_cmc(arch, enc_name, use_ckpt=False):
    if use_ckpt:
        return os.path.join(LOGS_DIR_CORTEX_MEDULLA_CAPSULE, f"{arch.lower()}_{enc_name.lower()}_ckpt")
    else:
        return os.path.join(LOGS_DIR_CORTEX_MEDULLA_CAPSULE, f"{arch.lower()}_{enc_name.lower()}")


def get_ckpt_cmc(arch, enc_name):
    return os.path.join(get_logs_cmc(arch, enc_name), 'ckpt')


def get_final_model_cmc(arch, enc_name, microscopy_type="paraffin"):
    # return os.path.join(get_logs_cmc(arch, enc_name), 'model_final.pt')
    base_root = os.path.join(ROOT_DIR, 'semseg', 'lightning')
    weight_file = f'{MODELS_DICT_LIGHTNING_CMC[arch]}.pt'
    model_file = os.path.join(base_root, 'weights', microscopy_type, weight_file)
    return model_file


def get_logs_aag(arch, enc_name, use_ckpt=False):
    if use_ckpt:
        return os.path.join(LOGS_DIR_ARTERIOLE_ARTERY_GLOMERULUS, f"{arch.lower()}_{enc_name.lower()}_ckpt")
    else:
        return os.path.join(LOGS_DIR_ARTERIOLE_ARTERY_GLOMERULUS, f"{arch.lower()}_{enc_name.lower()}")


def get_ckpt_aag(arch, enc_name):
    return os.path.join(get_logs_aag(arch, enc_name), 'ckpt')


def get_final_model_aag(arch, enc_name, microscopy_type="paraffin"):
    # return os.path.join(get_logs_aag(arch, enc_name), 'model_final.pt')
    base_root = os.path.join(ROOT_DIR, 'semseg', 'lightning')
    weight_file = f'{MODELS_DICT_LIGHTNING_AAG[arch]}.pt'
    model_file = os.path.join(base_root, 'weights', microscopy_type, weight_file)
    return model_file
