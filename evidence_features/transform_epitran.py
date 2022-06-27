import epitran
import ipasymbols
import numpy as np
from typing import List
import numpy as np

# Load the pretrained model
model_epi = epitran.Epitran('deu-Latn')


def consonant_to_float(sentences: List[str]):
    feats = consonant_to_int16(sentences)
    n_feats = feats.shape[-1] - 1  # should be 3
    return feats[:, 1:] / np.tile(feats[:, 0].reshape(-1, 1), n_feats)


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
    return np.vstack(feats).astype(np.int16)
