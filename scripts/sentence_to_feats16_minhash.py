import kshingle as ks
import datasketch
import struct
import mmh3
import numpy as np
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
    '--output-file', default="./feats16.jsonl", type=str
)
parser.add_argument(
    '--batch-size', default=500000, type=int
)
parser.add_argument(
    '--key-name', default="feats16", type=str
)
args = parser.parse_args()


# CPU settings
PCT_CPU = float(os.environ.get("MHASH_PCT_CPU", "0.9"))
NUM_CPU = os.environ.get("MHASH_NUM_CPU")
if NUM_CPU is None:
    NUM_CPU = max(1, int(psutil.cpu_count(logical=False) * PCT_CPU))
ray.init(num_cpus=NUM_CPU)



# util fun
def hashfunc_mmh3_int32(data: bytes) -> np.uint32:
    return struct.unpack('<I', struct.pack('<l', mmh3.hash(data)))[0]


def uint32_to_int32(i):  # i: np.uint32 -> np.int32
    return struct.unpack('<l', struct.pack('<I', i))[0]


@ray.remote
def get_kshingle_hashes_wrapper(text: str) -> List[np.int32]:
    # k-Shingling: k=6 is approx the average word length in German
    shingled = ks.shingleset_k(text, k=6)
    # build MinHash
    m = datasketch.MinHash(num_perm=32, hashfunc=hashfunc_mmh3_int32)
    for s in shingled:
        m.update(s.encode('utf8'))
    # cast hashvalues from uint64 to uint32
    hv = np.uint32(m.hashvalues)
    # cast to signed int32
    hv = [uint32_to_int32(i) for i in hv]
    return hv




def get_kshingle_hashes(sentences: List[str]):
    try:
        return ray.get([
            get_kshingle_hashes_wrapper.remote(sent)
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
                results = get_kshingle_hashes(batch_sentences)
                del batch_sentences

                # write data
                with jsonlines.open(args.output_file, mode='a') as writer:
                    for obj, res in zip(data, results):
                        obj[args.key_name] = res
                        writer.write(obj)

                del results, data
                batch_sentences, data = [], []

    if len(data) > 0:
        results = get_kshingle_hashes(batch_sentences)
        with jsonlines.open(args.output_file, mode='a') as writer:
            for obj, res in zip(data, results):
                obj[args.key_name] = res
                writer.write(obj)

    # done
    ray.shutdown()
    gc.collect()
    logger.info(f"End: {timer() - start: .6f} sec. elapsed")

