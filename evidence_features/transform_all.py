from .transform_sbert import sbert_to_bool, sbert_names
from .transform_trankit import trankit_to_float, trankit_names
from .transform_epitran import consonant_to_float, consonant_names
from .transform_derechar import (
    derechar_to_float, derebigram_to_float,
    derechar_names, derebigram_names
)
from .transform_cow import cow_to_float, cow_names
from .transform_smor import smor_to_float, smor_names
from .transform_seqlen import seqlen_to_float, seqlen_names
import numpy as np
from typing import List


def to_float(sentences: List[str]):
    feats1 = sbert_to_bool(sentences).astype(np.float32)
    feats2, feats3, feats4 = trankit_to_float(sentences)
    feats5 = consonant_to_float(sentences)
    feats6 = derechar_to_float(sentences)
    feats7 = derebigram_to_float(sentences)
    feats8 = cow_to_float(sentences)
    feats9, feats10, feats11 = smor_to_float(sentences)
    feats12 = seqlen_to_float(sentences)
    return np.hstack([
        feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8,
        feats9, feats10, feats11, feats12
    ])


def get_names():
    names = []
    names.extend(sbert_names())  # 1
    for tmp in trankit_names():  # 2/3/4
        names.extend(tmp)
    names.extend(consonant_names())  # 5
    names.extend(derechar_names())  # 6
    names.extend(derebigram_names())  # 7
    names.extend(cow_names())  # 8
    for tmp in smor_names():  # 9/10/11
        names.extend(tmp)
    names.extend(seqlen_names())  # 12
    return names
