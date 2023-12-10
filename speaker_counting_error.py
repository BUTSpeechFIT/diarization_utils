#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
from optparse import OptionParser
import os
from rttm_utils import rttm_to_hard_labels
from types import SimpleNamespace
import argparse
from pathlib import Path


def parse_arguments() -> SimpleNamespace:
    parser = argparse.ArgumentParser(
        description='Compute the mean speaker counting error for the amount of speakers found')
    parser.add_argument('--ref-rttm-dir', type=str, required=True,
                        help='directory with reference rttm files')
    parser.add_argument('--sys-rttm-dir', type=str, required=True,
                        help='directory with system rttm files')
    parser.add_argument('--txt-list', type=str, required=True,
                        help='list of files to process')
    parser.add_argument('--out-file', type=str, required=True,
                        help='output file where results are written')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    list = [line.rstrip() for line in open(args.txt_list, 'r')]
    if not os.path.exists(os.path.dirname(args.out_file)):
        os.makedirs(os.path.dirname(args.out_file))

    total_error = 0

    with open(args.out_file, 'w') as f:
        for line in list:
            key = line
            _, spk_labels = rttm_to_hard_labels(
                Path(args.ref_rttm_dir, F"{key}.rttm"), 1000)
            ref_qty = len(spk_labels)
            _, spk_labels = rttm_to_hard_labels(
                Path(args.sys_rttm_dir, F"{key}.rttm"), 1000)
            sys_qty = len(spk_labels)
            error = np.abs(ref_qty - sys_qty)
            total_error += error
            f.write(f"{line}\t{error}\n")
        f.write(f"Total \t{total_error / len(list)}\n")


if __name__ == "__main__":
    # execute only if run as a script
    main()
