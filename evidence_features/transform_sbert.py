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


# CUDA settings
if torch.cuda.is_available():
    GPUID = int(os.getenv("BERT_GPUID", 0))
    device = torch.device(f"cuda:{GPUID}")
else:
    device = torch.device('cpu')


# Balance GPU Memory for Tensorflow vs PyTorch on GPUID
if torch.cuda.is_available():
    # Limit TensorFlow to using only the desired GPUID
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[GPUID], device_type='GPU')

    # Limit TensorFlow to 49% of GPU memory or max 8Gb
    avail_gb = torch.cuda.mem_get_info()[GPUID] // 1024**2
    log_dev_conf = tf.config.LogicalDeviceConfiguration(
        memory_limit=min(8 * 1024, avail_gb * 0.49)
    )
    tf.config.set_logical_device_configuration(
        gpus[GPUID], [log_dev_conf])

    # Limit PyTorch to 49% of GPU memory on GPUID
    torch.cuda.set_per_process_memory_fraction(0.49, GPUID)


# Load the pretrained model
model_sbert = sbert.SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2',
    cache_folder=MODELPATH, device=device)


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
