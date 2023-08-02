import epitran
import ipasymbols
from typing import List
import logging
from timeit import default_timer as timer
import argparse
import ray
import gc
import os
import psutil
import jsonlines



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-file', default="./extracted.jsonl", type=str
)
parser.add_argument(
    '--output-file', default="./feats5.jsonl", type=str
)
parser.add_argument(
    '--batch-size', default=500000, type=int
)
args = parser.parse_args()


# CPU settings
PCT_CPU = float(os.environ.get("CONS_PCT_CPU", "0.9"))
NUM_CPU = os.environ.get("CONS_NUM_CPU")
if NUM_CPU is None:
    NUM_CPU = max(1, int(psutil.cpu_count(logical=False) * PCT_CPU))
ray.init(num_cpus=NUM_CPU)


# Load the pretrained model
model_epi = epitran.Epitran('deu-Latn')


# util fun
def clip_int16(x):
    return max(-32768, min(32767, x))


@ray.remote
def get_consonant_clusters_wrapper(sent: str):
    # convert to IPA
    ipatxt = model_epi.transliterate(sent)
    # identify clusters of 2 and 3, and count consonants
    clusters = ipasymbols.count_clusters(
        ipatxt, query={"type": [
            "pulmonic", "non-pulmonic",
            "affricate", "co-articulated"]},
        phonlen=3, min_cluster_len=1)
    # done
    return (
        clip_int16(len(ipatxt)),
        clip_int16(clusters.get(1, 0)),
        clip_int16(clusters.get(2, 0)),
        clip_int16(clusters.get(3, 0))
    )


def get_consonant_clusters(sentences: List[str]):
    try:
        return ray.get([
            get_consonant_clusters_wrapper.remote(sent)
            for sent in sentences])
    except Exception as e:
        logger.error(e)
        ray.shutdown()
        gc.collect()


if __name__ == '__main__':
    logger.info("Start")
    start = timer()

    # read data
    with jsonlines.open(args.input_file) as reader:
        batch_sentences, data = [], []
        for obj in reader:
            sent = obj.pop("sentence")
            sent_id = obj.pop("sent_id")
            batch_sentences.append(sent)
            data.append({"sent_id": sent_id})

            # start processing the batch
            if len(data) == args.batch_size:
                # process with ray.io
                results = get_consonant_clusters(batch_sentences)
                del batch_sentences

                # write data
                with jsonlines.open(args.output_file, mode='a') as writer:
                    for obj, res in zip(data, results):
                        obj["feats5"] = res
                        writer.write(obj)

                del results, data
                batch_sentences, data = [], []

    if len(data) > 0:
        results = get_consonant_clusters(batch_sentences)
        with jsonlines.open(args.output_file, mode='a') as writer:
            for obj, res in zip(data, results):
                obj["feats5"] = res
                writer.write(obj)

    # done
    ray.shutdown()
    gc.collect()
    logger.info(f"End: {timer() - start: .6f} sec. elapsed")

