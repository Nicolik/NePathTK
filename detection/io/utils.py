import numpy as np


def normalise(x):
    if x.max() - x.min() == 0:
        return x
    return (x - x.min()) / (x.max() - x.min())
