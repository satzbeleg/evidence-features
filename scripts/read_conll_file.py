#!/usr/bin/env python

import evidence_features as evf
import os
import argparse


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--conll-file', default="./example.conll", type=str
)
parser.add_argument(
    '--output-dir', default="./example", type=str
)
parser.add_argument(
    '--io-batch-size', default=int(1e6), type=int
)  # 1 Mio
args = parser.parse_args()


if __name__ == '__main__':
    os.makedirs(
        args.output_dir, exist_ok=True)
    evf.read_conll_file(
        args.conll_file, args.output_dir, args.io_batch_size)
