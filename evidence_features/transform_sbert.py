import os
import sentence_transformers as sbert
from typing import List

# path to the pretrained sbert model
MODELPATH = os.getenv("MODELFOLDER", "./models")
MODELPATH = os.path.join(MODELPATH, "sbert")

# Load the pretrained model
model_sbert = sbert.SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2',
    cache_folder=MODELPATH
)


def sbert_to_float(sentences: List[str]):
    return model_sbert.encode(sentences)


def sbert_to_int8(sentences: List[str]):
    pass
