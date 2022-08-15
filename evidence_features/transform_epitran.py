import epitran
import ipasymbols
import numpy as np
from typing import List
from .utils import divide_by_1st_col


# Load the pretrained model
model_epi = epitran.Epitran('deu-Latn')


def consonant_to_float(sentences: List[str]):
    feats = consonant_to_int16(sentences)
    return divide_by_1st_col(feats)


def consonant_to_int16(sentences: List[str]):
    feats = []
    for sent in sentences:
        # convert to IPA
        ipatxt = model_epi.transliterate(sent)
        # identify clusters of 2 and 3, and count consonants
        clusters = ipasymbols.count_clusters(
            ipatxt, query={"type": [
                "pulmonic", "non-pulmonic",
                "affricate", "co-articulated"]},
            phonlen=3, min_cluster_len=1)
        # save features
        feats.append((
            len(ipatxt),
            clusters.get(1, 0),
            clusters.get(2, 0),
            clusters.get(3, 0)
        ))
    # done
    feats = np.maximum(np.iinfo(np.int16).min, feats)
    feats = np.minimum(np.iinfo(np.int16).max, feats)
    return np.vstack(feats).astype(np.int16)


def consonant_names():
    return [f"consonant_{j}" for j in [1, 2, 3]]
