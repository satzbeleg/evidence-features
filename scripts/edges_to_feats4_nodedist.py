#!/usr/bin/env python
"""
This scripts reads `edges.jsonl` and compute token vs node distances
"""
import logging
from timeit import default_timer as timer
import argparse
import node_distance_ray as ndr
import ray
import gc
import numpy as np
import jsonlines
import itertools

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
args = parser.parse_args()



if __name__ == '__main__':
    logger.info("Start")

    # read data
    with jsonlines.open(args.input_file) as reader:
        batch_edges, batch_numnodes, data = [], [], []
        for obj in reader: 
            tmp = obj.pop("edges")
            batch_edges.append([tmp])
            batch_numnodes.append([max(itertools.chain(*tmp))])
            data.append(obj)

    # process with ray.io
    results = ndr.node_token_distances(
        batch_edges, batch_numnodes, cutoff=25)
    del batch_edges, batch_numnodes
    nodedist = [res[0] for res in results]
    tokendist = [res[1] for res in results]
    del results

    results2 = ndr.tokenvsnode_distribution(
        tokendist, nodedist, xmin=-5, xmax=15)
    del tokendist, nodedist

    # write data
    with jsonlines.open(args.output_file, mode='a') as writer:
        for obj, res in zip(data, results2):
            d = np.minimum(127, np.maximum(-128, res[2])).astype(np.int8).tolist()
            obj["feats4"] = d
            writer.write(obj)

    del results2

    # done
    ray.shutdown()
    gc.collect()
    logger.info("End")
