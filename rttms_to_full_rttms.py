#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

from rttm_utils import rttm_to_hard_labels, hard_labels_to_rttm
from types import SimpleNamespace
import argparse
import numpy as np
import os


def parse_arguments() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description='Convert RTTM to RTTM where overlap and silences are marked.')
    parser.add_argument('--rttm-in-dir', type=str, required=True,
                        help='directory with rttm files')
    parser.add_argument('--rttm-out-dir', type=str, required=True,
                        help='directory where to save full rttm files')
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
    if not os.path.exists(args.rttm_out_dir):
        os.makedirs(args.rttm_out_dir)
    for name in files_list:
        matrix, spk_labels = rttm_to_hard_labels(
            os.path.join(args.rttm_in_dir, f"{name}.rttm"),
            1000,
            lengths[name])
        sil_ov = np.zeros((matrix.shape[0], 2))
        sil_ov[:, 0] = matrix.sum(axis=1) == 0
        sil_ov[:, 1] = matrix.sum(axis=1) > 1
        ov_positions = np.where(matrix.sum(axis=1) > 1)
        matrix[ov_positions] = 0
        matrix = np.concatenate((sil_ov, matrix), axis=1)
        spk_labels = np.concatenate((['SILENCE', 'OVERLAP'], spk_labels))
        hard_labels_to_rttm(
                    matrix, spk_labels, name,
                    os.path.join(args.rttm_out_dir, f"{name}.rttm"),
                    1000)
