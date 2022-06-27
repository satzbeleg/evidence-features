from typing import List
import numpy as np


def seqlen_to_float(sentences: List[str]):
    feats = seqlen_to_int16(sentences)
    return np.log(feats + 1.)


def seqlen_to_int16(sentences: List[str]):
    # sentence length
    feats = []
    for sent in sentences:
        words = sent.split(" ")
        feats.append((
            len(sent),
            len(words)
        ))
    return np.vstack(feats).astype(np.int16)
