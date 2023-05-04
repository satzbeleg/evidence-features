import sys
sys.path.append('../..')

import evidence_features as evf
import glob

# path to the pretrained trankit model
import trankit
import os
import torch

MODELPATH = os.getenv("MODELFOLDER", "./models")
MODELPATH = os.path.join(MODELPATH, "trankit")

# Load the pretrained model
model_trankit = trankit.Pipeline(
    lang='german-hdt',
    gpu=torch.cuda.is_available(),
    cache_dir=MODELPATH
)

# Timer
from timeit import default_timer as timer
import logging

# start logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# FILES = glob.glob("data/*.txt")
# for FILE in FILES:

FILE = "data/schmoller_volkswirtschaftslehre_1893.txt"
with open(FILE, 'r') as fp:
    sentences = fp.readlines()
    sentences = [s.strip() for s in sentences]
    #sentences = sentences[:10]  # For dev
    logger.info(f"Batch Size: {len(sentences)}")

    # sentence-level processing with trankit
    (
        feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8,
        feats9, feats12, feats13, feats14, hashes15, hashes16,
        sentences_parsed, lemmata17, spans, annotations
    ) = evf.to_int(sentences, measure_time=True, sbert_masking=True, document_level=False)

    # document-level processing with trankit
    (
        feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8,
        feats9, feats12, feats13, feats14, hashes15, hashes16,
        sentences_parsed, lemmata17, spans, annotations
    ) = evf.to_int(sentences, measure_time=True, sbert_masking=True, document_level=True)

    # check how much tim just trankit needs
    start = timer()
    snt = model_trankit("\n".join(sentences))
    logger.info(f"{timer() - start: .6f} sec. elapsed, Trankit Only")
