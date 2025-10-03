import torch
import numpy as np


def reshape_transform(tensor, height=24, width=24):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_dict_label_rgb_mesc():
    return {
        'M': {
            0: (0, 255, 0, 255),  # Green
            1: (255, 255, 0, 255),  # Yellow
            2: (255, 0, 255, 255)  # Magenta
        },
        'E': {
            0: (255, 0, 0, 255),  # Red
            1: (0, 255, 255, 255),  # Cyan
        },
        'S': {
            0: (0, 255, 0, 255),  # Green
            1: (255, 255, 0, 255),  # Yellow
            2: (255, 0, 255, 255)  # Magenta
        },
        'C': {
            0: (255, 0, 0, 255),  # Red
            1: (0, 255, 255, 255),  # Cyan
        },
    }


def get_dict_border_rgb_mesc():
    return {
        'M': {
            0: (0, 180, 0, 255),  # Green
            1: (180, 180, 0, 255),  # Yellow
            2: (180, 0, 180, 255)  # Magenta
        },
        'E': {
            0: (180, 0, 0, 255),  # Red
            1: (0, 180, 180, 255),  # Cyan
        },
        'S': {
            0: (0, 180, 0, 255),  # Green
            1: (180, 180, 0, 255),  # Yellow
            2: (180, 0, 180, 255)  # Magenta
        },
        'C': {
            0: (180, 0, 0, 255),  # Red
            1: (0, 180, 180, 255),  # Cyan
        },
    }


def get_dict_label_rgb_sclerosis():
    return {
        'two-class': {
            0: (0, 255, 0, 255),    # Green
            1: (255, 255, 0, 255),  # Magenta
        },
        'three-class': {
            0: (0, 255, 0, 255),     # Green
            1: (255, 0, 255, 255),   # Yellow
            2: (255, 255, 0, 255)    # Magenta
        },
    }


def get_dict_border_rgb_sclerosis():
    return {
        'two-class': {
            0: (0, 180, 0, 255),    # Green
            1: (180, 180, 0, 255),  # Magenta
        },
        'three-class': {
            0: (0, 180, 0, 255),     # Green
            1: (180, 0, 180, 255),   # Yellow
            2: (180, 180, 0, 255)    # Magenta
        },
    }
