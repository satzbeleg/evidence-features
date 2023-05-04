import os
import sentence_transformers as sbert
import keras_hrp as khrp
import numpy as np
from typing import List
import torch
import tensorflow as tf

# path to the pretrained sbert model
MODELPATH = os.getenv("MODELFOLDER", "./models")
MODELPATH = os.path.join(MODELPATH, "sbert")

# Load the pretrained model
model_sbert = sbert.SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2',
    cache_folder=MODELPATH,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)


# HRP runs on Tensorflow, so we need to limit the GPU memory usage
if torch.cuda.is_available():
    gpus = tf.config.list_physical_devices('GPU')
    # limit to using only the 1st GPU
    tf.config.set_visible_devices(gpus[0], device_type='GPU')
    # limit memory usage to 20% of GPU memory or max 4Gb
    avail_gb = torch.cuda.mem_get_info()[0] // 1024**2
    log_dev_conf = tf.config.LogicalDeviceConfiguration(
        memory_limit=min(4 * 1024, avail_gb * 0.2)
    )
    tf.config.set_logical_device_configuration(
        gpus[0], [log_dev_conf])

# HRP layer
model_hrp = khrp.HashedRandomProjection(
    output_size=1024,
    random_state=42
)


def sbert_i2b(encoded):
    return np.vstack([khrp.int8_to_bool(enc) for enc in encoded])


def sbert_to_bool(sentences: List[str]):
    encoded = sbert_to_int8(sentences)
    return sbert_i2b(encoded)


def sbert_to_int8(sentences: List[str]):
    # run SBert
    feats = model_sbert.encode(sentences)
    torch.cuda.empty_cache()
    # project to boolean
    hashed = model_hrp(feats)
    # encode to int8
    return np.vstack([
        khrp.bool_to_int8(h.reshape(-1))
        for h in hashed.numpy().astype(bool)])


def sbert_names():
    return [f"sbert_{j}" for j in range(1024)]
