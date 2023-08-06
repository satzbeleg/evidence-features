from typing import List
import logging
from timeit import default_timer as timer
import argparse
import ray
import gc
import os
import psutil
import jsonlines
import nltk
import json
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-file', default="./words.jsonl", type=str
)
parser.add_argument(
    '--output-file', default="./feats67.jsonl", type=str
)
parser.add_argument(
    '--batch-size', default=500000, type=int
)
parser.add_argument(
    '--model-folder', default="./models", type=str
)
args = parser.parse_args()


# CPU settings
PCT_CPU = float(os.environ.get("DECOW_PCT_CPU", "0.9"))
NUM_CPU = os.environ.get("DECOW_NUM_CPU")
if NUM_CPU is None:
    NUM_CPU = max(1, int(psutil.cpu_count(logical=False) * PCT_CPU))
ray.init(num_cpus=NUM_CPU)


# path to the COW datasets
MODELPATH1 = os.path.join(args.model_folder, "decow.csv")
MODELPATH2 = os.path.join(args.model_folder, "decow.json")

# load NLTK stemmer
stemmer = nltk.stem.Cistem()

# load COW frequency list
if os.path.isfile(MODELPATH2):
    with open(MODELPATH2, "r") as fp:
        decow = json.load(fp)
else:
    df_decow = pd.read_csv(MODELPATH1, index_col=['word'])
    df_decow = df_decow[df_decow["freq"] > 100]  # removes ~97% of rows!
    df_decow = np.log(df_decow + 1.)
    df_decow = df_decow / df_decow.max()
    decow = {row[0]: row[1].values[0] for row in df_decow.iterrows()}
    del df_decow
    gc.collect()
    with open(MODELPATH2, "w") as fp:
        json.dump(decow, fp)


decow_id = ray.put(decow)

# brackets = np.percentile(
#     [num for _, num in decow.items()],
#     q=[100 / 6, 100 / 3, 50, 200 / 3, 500 / 6, 100])

brackets = [
    0.24802303790806676, 
    0.2643387072189492, 
    0.28548954454370207, 
    0.31516192742326904, 
    0.3665744052229642, 
    1.0
]


# util fun
def clip_int8(x):
    return max(-128, min(127, x))


def find_percentile(f, brackets):
    for i, b in enumerate(brackets):
        if f <= b:
            return i
    return len(brackets) - 1


@ray.remote
def cow_to_int8_wrapper(words: List[str], decow):
    # lookup frequency
    freqs = [decow.get(stemmer.stem(w), 0) for w in words]
    # assign to percentile
    percentile = [find_percentile(f, brackets) for f in freqs]
    # count percentile
    cnt = [0] * 6
    for p in percentile:
        cnt[p] += 1
    # done
    return (
        clip_int8(len(words)),
        *[clip_int8(c) for c in cnt]
    )



def cow_to_int8(batch_words: List[str]):
    try:
        return ray.get([
            cow_to_int8_wrapper.remote(words, decow_id)
            for words in batch_words])
    except Exception as e:
        logger.error(e)
        ray.shutdown()
        gc.collect()


if __name__ == '__main__':
    logger.info("Start")
    start = timer()

    # read data
    with jsonlines.open(args.input_file) as reader:
        batch_words, data = [], []
        for obj in reader:
            words = obj.pop("words")
            sent_id = obj.pop("sent_id")
            batch_words.append(words)
            data.append({"sent_id": sent_id})

            # start processing the batch
            if len(data) == args.batch_size:
                # process with ray.io
                results = cow_to_int8(batch_words)
                del batch_words

                # write data
                with jsonlines.open(args.output_file, mode='a') as writer:
                    for obj, res in zip(data, results):
                        obj["feats8"] = res
                        writer.write(obj)

                del results, data
                batch_words, data = [], []

    if len(data) > 0:
        results = cow_to_int8(batch_words)
        with jsonlines.open(args.output_file, mode='a') as writer:
            for obj, res in zip(data, results):
                obj["feats8"] = res
                writer.write(obj)

    # done
    ray.shutdown()
    gc.collect()
    logger.info(f"End: {timer() - start: .6f} sec. elapsed")

