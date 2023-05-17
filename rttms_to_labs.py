#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

from lab_utils import hard_labels_to_lab
from rttm_utils import rttm_to_hard_labels
from types import SimpleNamespace
import argparse
import numpy as np
import os


def parse_arguments() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description='Derive VAD labels from RTTM.')
    parser.add_argument('--rttm-in-dir', type=str, required=True,
                        help='directory with rttm files')
    parser.add_argument('--lab-out-dir', type=str, required=True,
                        help='directory where to save lab files')
    parser.add_argument('--txt-list-file', type=str, required=True,
                        help='file containing list of files to process')
    parser.add_argument('--lengths', type=str, required=True,
                        help='file containing list of lengths per file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    lengths_list = np.loadtxt(args.lengths, dtype=object)
    lengths = {}
    for line in lengths_list:
        name, length = line
        lengths[name] = float(length)

    files_list = np.loadtxt(args.txt_list_file, dtype=object)
    if not os.path.exists(args.lab_out_dir):
        os.makedirs(args.lab_out_dir)
    for name in files_list:
        matrix, _ = rttm_to_hard_labels(
            os.path.join(args.rttm_in_dir, f"{name}.rttm"),
            1000,
            lengths[name])
        speech = matrix.sum(axis=1) > 0
        hard_labels_to_lab(
                    speech,
                    os.path.join(args.lab_out_dir, f"{name}.lab"),
                    1000)
