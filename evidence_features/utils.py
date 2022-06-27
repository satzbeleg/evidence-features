from typing import List
import numpy as np


def divide_by_1st_col(feats: List[List[float]]):
    n_feats = feats.shape[-1] - 1
    return feats[:, 1:] / np.tile(feats[:, 0].reshape(-1, 1), n_feats)


def divide_by_sum(feats: List[List[float]]):
    n_feats = feats.shape[-1]
    return feats / np.tile(feats.sum(axis=1).reshape(-1, 1), n_feats)
