from typing import List
import numpy as np


def seqlen_i2f(feats):
    return np.log(feats + 1.)


def seqlen_to_float(sentences: List[str]):
    feats = seqlen_to_int16(sentences)
    return seqlen_i2f(feats)


def seqlen_to_int16(sentences: List[str]):
    # sentence length
    feats = []
    for sent in sentences:
        words = sent.split(" ")
        feats.append((
            len(sent),
            len(words)
        ))
    feats = np.maximum(np.iinfo(np.int16).min, feats)
    feats = np.minimum(np.iinfo(np.int16).max, feats)
    return np.vstack(feats).astype(np.int16)


def seqlen_names():
    return ["seqlen_string", "seqlen_words"]
