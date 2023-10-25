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
        description='Compute the confusion matrix for the amount of speakers found')
    parser.add_argument('--ref-rttm-dir', type=str, required=True,
                        help='directory with reference rttm files')
    parser.add_argument('--sys-rttm-dir', type=str, required=True,
                        help='directory with system rttm files')
    parser.add_argument('--txt-list', type=str, required=True,
                        help='list of files to process')
    parser.add_argument('--out-file', type=str, required=True,
                        help='output file where results are written')
    parser.add_argument('--max-num-speakers', type=int, default=10,
                        help='Maximum number of speakers to consider in matrix')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    list = [line.rstrip() for line in open(args.txt_list, 'r')]
    if not os.path.exists(os.path.dirname(args.out_file)):
        os.makedirs(os.path.dirname(args.out_file))

    counts_dict = {}
    for line in list:
        key = line
        _, spk_labels = rttm_to_hard_labels(
            Path(args.ref_rttm_dir, F"{key}.rttm"), 1000)
        ref_qty = len(spk_labels)
        _, spk_labels = rttm_to_hard_labels(
            Path(args.sys_rttm_dir, F"{key}.rttm"), 1000)
        sys_qty = len(spk_labels)
        if (ref_qty, sys_qty) in counts_dict:
            counts_dict[(ref_qty, sys_qty)] += 1
        else:
            counts_dict[(ref_qty, sys_qty)] = 1

    with open(args.out_file, 'w') as f:
        header_str = "Ref/Sys".ljust(10)
        for i in range(0, args.max_num_speakers + 1):
            header_str += str(i).ljust(4)
        header_str += "\n"
        f.write(header_str)
        for ref in range(0, args.max_num_speakers + 1):
            f.write(str(ref).ljust(10))
            for sys in range(0, args.max_num_speakers + 1):
                if (ref,sys) in counts_dict:
                    qty = counts_dict[(ref,sys)]
                else:
                    qty = 0
                f.write(str(qty).ljust(4))
            f.write("\n")

        

if __name__ == "__main__":
    # execute only if run as a script
    main()
