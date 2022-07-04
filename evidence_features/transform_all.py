from .transform_sbert import (
    sbert_to_bool,
    sbert_to_int8,
    sbert_i2b,
    sbert_names
)
from .transform_trankit import (
    trankit_to_float,
    trankit_to_int8,
    trankit_names
)
from .transform_epitran import (
    consonant_to_float,
    consonant_to_int16,
    consonant_names
)
from .transform_derechar import (
    derechar_to_float,
    derechar_to_int16,
    derechar_names,
    derebigram_to_float,
    derebigram_to_int16,
    derebigram_names
)
from .transform_cow import (
    cow_to_float,
    cow_to_int8,
    cow_names
)
from .transform_smor import (
    smor_to_float,
    smor_to_int8,
    smor_names
)
from .transform_seqlen import (
    seqlen_to_float,
    seqlen_to_int16,
    seqlen_i2f,
    seqlen_names
)
from .utils import (
    divide_by_1st_col,
    divide_by_sum
)
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


def to_int(sentences: List[str]):
    feats1 = sbert_to_int8(sentences)
    feats2, feats3, feats4 = trankit_to_int8(sentences)
    feats5 = consonant_to_int16(sentences)
    feats6 = derechar_to_int16(sentences)
    feats7 = derebigram_to_int16(sentences)
    feats8 = cow_to_int8(sentences)
    feats9, feats10, feats11 = smor_to_int8(sentences)
    feats12 = seqlen_to_int16(sentences)
    return (
        feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8,
        feats9, feats10, feats11, feats12
    )


def i2f(feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8,
        feats9, feats10, feats11, feats12):
    return np.hstack([
        sbert_i2b(feats1),  # sbert
        divide_by_1st_col(feats2),  # trankit
        divide_by_1st_col(feats3),  # trankit
        divide_by_sum(feats4),  # trankit
        divide_by_1st_col(feats5),  # consonant
        divide_by_1st_col(feats6),  # char
        divide_by_1st_col(feats7),  # bigram
        divide_by_1st_col(feats8),  # cow
        divide_by_1st_col(feats9),  # smor
        divide_by_1st_col(feats10),  # smor
        divide_by_1st_col(feats11),  # smor
        seqlen_i2f(feats12)  # seqlen
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
