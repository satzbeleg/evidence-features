from typing import List
import numpy as np


def divide_by_1st_col(feats: List[List[float]]):
    n_feats = feats.shape[-1] - 1
    denom = np.maximum(feats[:, 0], 1)
    return feats[:, 1:] / np.tile(denom.reshape(-1, 1), n_feats)


def divide_by_sum(feats: List[List[float]]):
    n_feats = feats.shape[-1]
    denom = np.maximum(feats.sum(axis=1), 1)
    return feats / np.tile(denom.reshape(-1, 1), n_feats)
