from typing import List
import logging
from timeit import default_timer as timer
import argparse
import gc
import os
import psutil
import jsonlines
import sfst_transduce
import re
import multiprocessing as mp


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
PCT_CPU = float(os.environ.get("SMOR_PCT_CPU", "0.9"))
NUM_CPU = os.environ.get("SMOR_NUM_CPU")
if NUM_CPU is None:
    NUM_CPU = max(1, int(psutil.cpu_count(logical=False) * PCT_CPU))


# path to the SMOR datasets
MODELPATH = os.path.join(args.model_folder, "smor.a")

# Load the pretrained model
model_fst = sfst_transduce.Transducer(MODELPATH)



# Morphemes/Lexemes
def get_morphology_stats(word):
    res = model_fst.analyse(word)
    if len(res) == 0:
        return 0, 0, 0
    variants = {}
    for sinp in res:
        s = re.sub(r'<[^>]*>', '\t', sinp)
        lexemes = [t for t in s.split("\t") if len(t) > 0]
        key = "+".join(lexemes)
        variants[key] = lexemes
    # syntactial ambivalence
    num_usecases = len(res)
    # lexeme ambivalence
    num_splittings = len(variants)
    # working memory for composita comprehension
    max_lexemes = max([len(lexemes) for lexemes in variants.values()])
    return num_usecases, num_splittings, max_lexemes

brackets1 = [1, 2, 4, 8, 16]
brackets2 = [1, 2, 3]
brackets3 = [1, 2, 3]


# util fun
def clip_int8(x):
    return max(-128, min(127, x))


def find_bucket(f, brackets):
    for i, b in enumerate(brackets):
        if f <= b:
            return i
    return len(brackets)



# @ray.remote
def get_morphology_distributions_wrapper(tokens: List[str]):
    # get stats
    num_usecases = []
    num_splittings = []
    max_lexemes = []
    for word in tokens:
        tmp = get_morphology_stats(word)
        num_usecases.append(tmp[0])
        num_splittings.append(tmp[1])
        max_lexemes.append(tmp[2])
    # (A) syntactial ambivalence
    classes = [find_bucket(f, brackets1) for f in num_usecases]
    cnt1 = [0] * 6
    for idx in classes:
        cnt1[idx] += 1
    # (B) lexeme ambivalence
    classes = [find_bucket(f, brackets2) for f in num_splittings]
    cnt2 = [0] * 4
    for idx in classes:
        cnt2[idx] += 1
    # (C) working memory for composita comprehension
    classes = [find_bucket(f, brackets3) for f in max_lexemes]
    cnt3 = [0] * 4
    for idx in classes:
        cnt3[idx] += 1
    # done
    return (
        clip_int8(len(tokens)), 
        *[clip_int8(c) for c in cnt1], 
        *[clip_int8(c) for c in cnt2], 
        *[clip_int8(c) for c in cnt3]
    )

def get_morphology_distributions(batch_words: List[str]):
    try:
        with mp.Pool(NUM_CPU) as pool:
            res = pool.map(get_morphology_distributions_wrapper, batch_words)
        return res
    except Exception as e:
        logger.error(e)
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
                results = get_morphology_distributions(batch_words)
                del batch_words

                # write data
                with jsonlines.open(args.output_file, mode='a') as writer:
                    for obj, res in zip(data, results):
                        obj["feats9"] = res
                        writer.write(obj)

                del results, data
                batch_words, data = [], []

    if len(data) > 0:
        results = get_morphology_distributions(batch_words)
        with jsonlines.open(args.output_file, mode='a') as writer:
            for obj, res in zip(data, results):
                obj["feats9"] = res
                writer.write(obj)

    # done
    gc.collect()
    logger.info(f"End: {timer() - start: .6f} sec. elapsed")

