#!/usr/bin/env python
"""
This scripts reads `masked.jsonl`, compute SBERT embeddings, applies HRP,
and writes these to `hashed.jsonl`.
"""
import logging
from timeit import default_timer as timer
import argparse
import torch
import sentence_transformers as sbert
import torch_hrp as thrp
import os
from typing import List
import jsonlines
import numpy as np


# start logger
logger = logging.getLogger(__name__)


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-file', default="./masked.jsonl", type=str
)
parser.add_argument(
    '--output-file', default="./hashed.jsonl", type=str
)

parser.add_argument(
    '--sbert-path', default="./models/sbert", type=str
)
parser.add_argument(
    '--hrp-filepath', default="./models/hrp.pth", type=str
)

parser.add_argument(
    '--batch-size', default=int(18e6), type=int
)  # 18 Mio
parser.add_argument(
    '--sbert-chunk-size', default=int(12000), type=int
)  # try int(12000)  for 2x GPUs with 80 Gb each (peak 74 Gb)
parser.add_argument(
    '--hrp-chunk-size', default=int(45e5), type=int
)  # try int(45e5) for 2x GPUs with 80 Gb each (peak 68 Gb)
args = parser.parse_args()



# CUDA settings
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
logger.info(f"Using device: {device}")

# Load the pretrained model
model_sbert = sbert.SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2',
    cache_folder=args.sbert_path, device=device)


# HRP layer
model_hrp = thrp.HashedRandomProjection(
    output_size=1024,
    input_size=384,
    random_state=42
)
# load or save model params
if os.path.exists(args.hrp_filepath):
    model_hrp.load_state_dict(torch.load(args.hrp_filepath))
else:
    torch.save(model_hrp.state_dict(), args.hrp_filepath)



def sbert_to_bool_gpu(sentences: List[str]):
    # run SBert in multiprocessing mode
    start = timer()
    pool = model_sbert.start_multi_process_pool()
    feats_float = model_sbert.encode_multi_process(
        sentences, pool, 
        batch_size=args.sbert_chunk_size,  # set batch_size=chunk_size
        chunk_size=args.sbert_chunk_size)
    model_sbert.stop_multi_process_pool(pool)
    torch.cuda.empty_cache()
    logger.info(f"{timer() - start: .6f} seconds")

    # random projection to boolean
    start = timer()
    pool = model_hrp.start_pool()
    feats_hashed = model_hrp.infer(
        feats_float, pool, 
        chunk_size=args.hrp_chunk_size)
    model_hrp.stop_pool(pool)
    torch.cuda.empty_cache()
    logger.info(f"{timer() - start: .6f} seconds")

    # done
    return feats_hashed


def sbert_to_bool_cpu(sentences: List[str]):
    # run SBert
    start = timer()
    feats_float = model_sbert.encode(sentences)
    logger.info(f"{timer() - start: .6f} seconds")

    # random projection to boolean
    start = timer()
    feats_hashed = model_hrp(torch.tensor(feats_float))
    logger.info(f"{timer() - start: .6f} seconds")

    # done
    return feats_hashed.detach().numpy()


def sbert_to_bool(sentences: List[str]):
    if torch.cuda.is_available():
        return sbert_to_bool_gpu(sentences)
    else:
        return sbert_to_bool_cpu(sentences)



if __name__ == '__main__':  # multiprocessing spawning requires main
    # loop over all batches
    masked, data = [], []
    with jsonlines.open(args.input_file) as reader:
        for obj in reader:  
            masked.append(obj.pop("masked"))
            data.append(obj)
            if len(masked) >= args.batch_size:
                hashed = sbert_to_bool(masked)  # compute!
                # save results
                with jsonlines.open(args.output_file, mode='a') as writer:
                    for i, ex in enumerate(data):
                        writer.write({**ex, "hashed": hashed[i].astype(np.int8).tolist()})
                    masked, data = [], []  # reset
    # process the last examples
    if len(masked) > 0:
        hashed = sbert_to_bool(masked)  # compute!
        # save results
        with jsonlines.open(args.output_file, mode='a') as writer:
            for i, ex in enumerate(data):
                writer.write({**ex, "hashed": hashed[i].astype(np.int8).tolist()})
            masked, data = [], []  # reset

# Speed Tests
# Sbert: 187.872131 seconds (1 Mio Examples; Chunk size: 12k)
# Sbert: 4817.592480 seconds (20 Mio Examples; Chunk size: 12k)
# HRP: 175.793451 seconds (20 Mio Examples; Chunk size: 4,5 Mio)
