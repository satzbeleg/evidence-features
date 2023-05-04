from .transform_sbert import (
    sbert_to_bool,
    sbert_to_int8,
    sbert_i2b,
    sbert_names
)
from .transform_trankit import (
    trankit_to_float,
    trankit_to_int,
    trankit_names
)
from .transform_kshingle import (
    kshingle_to_int32
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
from .transform_fasttext176 import (
    fasttext176_to_float,
    fasttext176_to_int8,
    fasttext176_i2f,
    fasttext176_names
)
from .transform_emoji import (
    emoji_to_float,
    emoji_to_int8,
    emoji_names
)
from .utils import (
    divide_by_1st_col,
    divide_by_sum
)
import numpy as np
from typing import List

from timeit import default_timer as timer
import logging

# start logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_float(sentences: List[str], 
             masked: List[str] = None):
    if masked is None:
        feats1 = sbert_to_bool(sentences).astype(np.float32)
    else:
        feats1 = sbert_to_bool(masked).astype(np.float32)
    feats2, feats3, feats4 = trankit_to_float(sentences)
    feats5 = consonant_to_float(sentences)
    feats6 = derechar_to_float(sentences)
    feats7 = derebigram_to_float(sentences)
    feats8 = cow_to_float(sentences)
    feats9 = smor_to_float(sentences)
    feats12 = seqlen_to_float(sentences)
    feats13 = fasttext176_to_float(sentences)
    feats14 = emoji_to_float(sentences)
    return np.hstack([
        feats1, feats2, feats3, feats4,
        feats5, feats6, feats7, feats8,
        feats9, feats12, feats13, feats14
    ])


def to_int(sentences: List[str],
           measure_time=False,
           sbert_masking=False,
           document_level=False):
    """ Transform a list of sentences to a numpy array of integers.

    Args:
        sentences: 
            list of sentences
        
        measure_time: 
            measure time for each transformation
        
        sbert_masking: 
            use masked sentences for SBert
        
        document_level: 
            use document-level Trankit.
            The number of sentences might change if document_level=True.
              Then use the output `sentences_sbd` afterwards
    """
    if measure_time:
        start = timer()
        (
            feats2, feats3, feats4, hashes15,
            sentences_sbd, lemmata17, masked, spans, annotations
        ) = trankit_to_int(sentences, document_level=document_level)
        if document_level:
            sentences = sentences_sbd
        logger.info(
            f"{timer() - start: .6f} sec. elapsed (2/3/4/15/17/m/s/a) Trankit")

        start = timer()
        if sbert_masking:
            feats1 = sbert_to_int8(
                [elem for sublist in masked for elem in sublist])
        else:
            feats1 = sbert_to_int8(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (1) SBert")

        start = timer()
        feats5 = consonant_to_int16(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (5) Consonant")

        start = timer()
        feats6 = derechar_to_int16(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (6) DeReChar")

        start = timer()
        feats7 = derebigram_to_int16(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (7) DeReBigram")

        start = timer()
        feats8 = cow_to_int8(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (8) COW")

        start = timer()
        feats9 = smor_to_int8(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (9) SMOR")

        start = timer()
        feats12 = seqlen_to_int16(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (12) Seqlen")

        start = timer()
        feats13 = fasttext176_to_int8(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (13) FastText")

        start = timer()
        feats14 = emoji_to_int8(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (14) Emoji")

        start = timer()
        hashes16 = kshingle_to_int32(sentences)
        logger.info(f"{timer() - start: .6f} sec. elapsed (16) Fingerprint")
    else:
        # Trankit
        (
            feats2, feats3, feats4, hashes15,
            sentences_sbd, lemmata17, masked, spans, annotations
        ) = trankit_to_int(sentences, document_level=document_level)
        if document_level:
            sentences = sentences_sbd
        # for sbert N masked sentences or 1 original sentence
        if sbert_masking:
            feats1 = sbert_to_int8(
                [elem for sublist in masked for elem in sublist])
        else:
            feats1 = sbert_to_int8(sentences)
        # other feautes
        feats1 = sbert_to_int8(sentences)
        feats5 = consonant_to_int16(sentences)
        feats6 = derechar_to_int16(sentences)
        feats7 = derebigram_to_int16(sentences)
        feats8 = cow_to_int8(sentences)
        feats9 = smor_to_int8(sentences)
        feats12 = seqlen_to_int16(sentences)
        feats13 = fasttext176_to_int8(sentences)
        feats14 = emoji_to_int8(sentences)
        hashes16 = kshingle_to_int32(sentences)

    # done
    return (
        feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8,
        feats9, feats12, feats13, feats14, hashes15, hashes16,
        sentences_sbd, lemmata17, spans, annotations
    )


def i2f(feats1, feats2, feats3, feats4,
        feats5, feats6, feats7, feats8,
        feats9, feats12, feats13, feats14):
    # convert to numpy
    feats1 = np.array(feats1, dtype=np.int8)
    feats2 = np.array(feats2, dtype=np.int8)
    feats3 = np.array(feats3, dtype=np.int8)
    feats4 = np.array(feats4, dtype=np.int8)
    feats5 = np.array(feats5, dtype=np.int16)
    feats6 = np.array(feats6, dtype=np.int16)
    feats7 = np.array(feats7, dtype=np.int16)
    feats8 = np.array(feats8, dtype=np.int8)
    feats9 = np.array(feats9, dtype=np.int8)
    feats12 = np.array(feats12, dtype=np.int16)
    feats13 = np.array(feats13, dtype=np.int8)
    feats14 = np.array(feats14, dtype=np.int8)
    # convert to floating-point features
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
        seqlen_i2f(feats12),  # seqlen
        fasttext176_i2f(feats13),   # fasttext176 langdetect
        divide_by_1st_col(feats14)   # emoji
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
    names.extend(smor_names())  # 9
    names.extend(seqlen_names())  # 12
    names.extend(fasttext176_names())  # 13
    names.extend(emoji_names())  # 14
    return names
