#!/usr/bin/env python
"""
This scripts reads `hashed.jsonl` and compute `feats1.jsonl`
"""
import logging
from timeit import default_timer as timer
import argparse
import bool_to_int8_ray as b2i8
import ray
import gc
import numpy as np
import jsonlines


# start logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-file', default="./edges.jsonl", type=str
)
parser.add_argument(
    '--output-file', default="./feats4.jsonl", type=str
)
parser.add_argument(
    '--batch-size', default=500000, type=int
)
args = parser.parse_args()


if __name__ == '__main__':
    logger.info("Start")
    start = timer()

    # read data
    with jsonlines.open(args.input_file) as reader:
        hashed, data = [], []
        for obj in reader: 
            hashed.append(obj.pop("hashed"))
            data.append(obj)
            # start processing the batch
            if len(hashed) == args.batch_size:
                serialized = b2i8.bool_to_int8_batch(np.array(hashed))
                del hashed
                # save batch results
                with jsonlines.open(args.output_file, mode='a') as writer:
                    for obj, val in zip(data, serialized):
                        writer.write({**obj, "feats1": val.tolist()})
                # reset
                del data, serialized
                hashed, data = [], []

    if len(hashed) > 0:
        # process with ray.io
        serialized = b2i8.bool_to_int8_batch(np.array(hashed))
        del hashed
        # write data
        with jsonlines.open(args.output_file, mode='a') as writer:
            for obj, val in zip(data, serialized):
                writer.write({**obj, "feats1": val.tolist()})
        del data, serialized

    # done
    ray.shutdown()
    gc.collect()
    logger.info(f"End: {timer() - start: .6f} sec. elapsed")
