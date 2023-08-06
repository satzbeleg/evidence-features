import treesimi as ts
import datasketch
import struct
import mmh3
import json
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
    '--input-file', default="./adjac.jsonl", type=str
)
parser.add_argument(
    '--output-file', default="./feats16.jsonl", type=str
)
parser.add_argument(
    '--batch-size', default=500000, type=int
)
args = parser.parse_args()


# CPU settings
PCT_CPU = float(os.environ.get("TSIMI_PCT_CPU", "0.9"))
NUM_CPU = os.environ.get("TSIMI_NUM_CPU")
if NUM_CPU is None:
    NUM_CPU = max(1, int(psutil.cpu_count(logical=False) * PCT_CPU))
ray.init(num_cpus=NUM_CPU)



# util fun
def hashfunc_mmh3_int32(data: bytes) -> np.uint32:
    return struct.unpack('<I', struct.pack('<l', mmh3.hash(data)))[0]


@ray.remote
def get_treesimi_hashes_wrapper(adjac) -> List[np.int32]:
    # Convert to Nested Set model
    nested = ts.adjac_to_nested_with_attr(adjac)
    nested = ts.remove_node_ids(nested)
    # shingle subtrees
    shingled = ts.shingleset(nested, use_trunc_leaves=True,
                             use_drop_nodes=False, use_replace_attr=False)
    stringified = [json.dumps(subtree).encode('utf-8')
                   for subtree in shingled]
    # encode with MinHash an MurmurHash3 (mmh3)
    m = datasketch.MinHash(num_perm=32, hashfunc=hashfunc_mmh3_int32)
    for s in stringified:
        m.update(s)
    # cast hashvalues from uint64 to uint32
    hv = np.uint32(m.hashvalues)
    # cast to signed int32
    hv = [ts.uint32_to_int32(i) for i in hv]
    return hv


def get_treesimi_hashes(batch_adjac: List[list]):
    try:
        return ray.get([
            get_treesimi_hashes_wrapper.remote(adjac)
            for adjac in batch_adjac])
    except Exception as e:
        logger.error(e)
        ray.shutdown()
        gc.collect()


if __name__ == '__main__':
    logger.info("Start")
    start = timer()

    # read data
    with jsonlines.open(args.input_file) as reader:
        batch_adjac, data = [], []
        for obj in reader:
            adjac = obj.pop("adjac")
            sent_id = obj.pop("sent_id")
            batch_adjac.append(adjac)
            data.append({"sent_id": sent_id})

            # start processing the batch
            if len(data) == args.batch_size:
                # process with ray.io
                results = get_treesimi_hashes(batch_adjac)
                del batch_adjac

                # write data
                with jsonlines.open(args.output_file, mode='a') as writer:
                    for obj, res in zip(data, results):
                        obj["feats15"] = res
                        writer.write(obj)

                del results, data
                batch_adjac, data = [], []

    if len(data) > 0:
        results = get_treesimi_hashes(batch_adjac)
        with jsonlines.open(args.output_file, mode='a') as writer:
            for obj, res in zip(data, results):
                obj["feats15"] = res
                writer.write(obj)

    # done
    ray.shutdown()
    gc.collect()
    logger.info(f"End: {timer() - start: .6f} sec. elapsed")

