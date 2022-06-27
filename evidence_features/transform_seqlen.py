from typing import List
import numpy as np


def seqlen_to_float(sentences: List[str]):
    feats = other_to_int8(sentences)
    return np.log(feats + 1.)


def seqlen_to_int16(sentences: List[str]):
    # sentence length
    feats = []
    for sent in sentences:
        words = txt.split(" ")
        feats8.append((
            len(txt),
            len(words)
        ))
    return np.vstack(feats).astype(np.int16)