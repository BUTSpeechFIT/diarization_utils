#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
from optparse import OptionParser
import os
from rttm_utils import rttm_to_hard_labels
from types import SimpleNamespace
import argparse


def parse_arguments() -> SimpleNamespace:
    parser = argparse.ArgumentParser(
        description='Compute stats for silence, speech and overlap from rttms')
    parser.add_argument('--in-rttm-dir', type=str, required=True,
                        help='directory with rttm files')
    parser.add_argument('--out-file', type=str, required=True,
                        help='output file where results are written')
    parser.add_argument('--precision', type=float, required=False, default=1000.0,
                        help='precision used to interpret annotations')
    parser.add_argument('--lengths', type=str, required=True,
                        help='file containing list of lengths per file')
    parser.add_argument('--txt-list', type=str, required=True,
                        help='list of files to process')
    parser.add_argument('--uem-file', type=str, required=False,
                        help='optional uem segments')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    list = [line.rstrip() for line in open(args.txt_list, 'r')]
    if not os.path.exists(os.path.dirname(args.out_file)):
        os.makedirs(os.path.dirname(args.out_file))
    lengths_list = np.loadtxt(args.lengths, dtype=object)
    lengths = {}
    for line in lengths_list:
        name, length = line
        lengths[name] = float(length)
    if args.uem_file is not None:
        uem_list = np.loadtxt(args.uem_file, dtype=object)
        uem_info = {}
        for line in uem_list:
            if line[0] in uem_info.keys():
                uem_info[line[0]].append((float(line[2]), float(line[3])))
            else:
                uem_info[line[0]] = [(float(line[2]), float(line[3]))]

    all_sil = 0.0
    all_1spk = 0.0
    all_2spk = 0.0
    all_3spk = 0.0
    all_4spk = 0.0
    all_seconds = 0.0

    lengths_sil = []
    lengths_1spk = []
    lengths_2spk = []
    lengths_3spk = []
    lengths_4spk = []

    with open(args.out_file, 'w') as f:
        f.write('Name'.ljust(50)+'Silence (s)'.ljust(18) +
                'mean'.ljust(9)+'std'.ljust(13) +
                '1 speaker (s)'.ljust(18)+'mean'.ljust(9)+'std'.ljust(13) +
                '2 speakers (s)'.ljust(18)+'mean'.ljust(9)+'std'.ljust(13) +
                '3 speakers (s)'.ljust(18)+'mean'.ljust(9)+'std'.ljust(13) +
                '>3 speakers (s)'.ljust(18)+'mean'.ljust(9)+'std'.ljust(13)+'\n')
        for line in list:
            key = line
            matrix, _ = rttm_to_hard_labels(
                args.in_rttm_dir+'/'+key+'.rttm', args.precision, lengths[key])
            if args.uem_file is not None:
                # Create corresponding mask
                mask = np.zeros(matrix.shape[0], dtype=int)
                for seg in uem_info[key]:
                    beg = int(seg[0]*args.precision)
                    end = int(seg[1]*args.precision)
                    mask[beg:end] = 1
                matrix = matrix[mask == 1,:]
            classes = np.sum(matrix, axis=1)

            seconds_sil = len(np.where(classes == 0)[0]) / args.precision
            seconds_1spk = len(np.where(classes == 1)[0]) / args.precision
            seconds_2spk = len(np.where(classes == 2)[0]) / args.precision
            seconds_3spk = len(np.where(classes == 3)[0]) / args.precision
            seconds_4spk = len(np.where(classes >= 4)[0]) / args.precision
            seconds = seconds_sil + seconds_1spk + seconds_2spk + seconds_3spk + seconds_4spk

            all_sil += seconds_sil
            all_1spk += seconds_1spk
            all_2spk += seconds_2spk
            all_3spk += seconds_3spk
            all_4spk += seconds_4spk
            all_seconds += seconds

            changes_positions = np.concatenate(([0], np.where(classes[:-1] != classes[1:])[0]+1, [len(classes)]))
            segment_type = classes[changes_positions[:-1]]
            segment_length = changes_positions[1:] - changes_positions[:-1]

            segments_sil = segment_length[np.where(segment_type == 0)[0]]
            segments_1spk = segment_length[np.where(segment_type == 1)[0]]
            segments_2spk = segment_length[np.where(segment_type == 2)[0]]
            segments_3spk = segment_length[np.where(segment_type == 3)[0]]
            segments_4spk = segment_length[np.where(segment_type >= 4)[0]]

            mean_sil = np.mean(segments_sil)
            std_sil = np.std(segments_sil)
            mean_1spk = np.mean(segments_1spk)
            std_1spk = np.std(segments_1spk)
            mean_2spk = np.mean(segments_2spk)
            std_2spk = np.std(segments_2spk)
            mean_3spk = np.mean(segments_3spk)
            std_3spk = np.std(segments_3spk)
            mean_4spk = np.mean(segments_4spk)
            std_4spk = np.std(segments_4spk)

            lengths_sil.extend(segments_sil)
            lengths_1spk.extend(segments_1spk)
            lengths_2spk.extend(segments_2spk)
            lengths_3spk.extend(segments_3spk)
            lengths_4spk.extend(segments_4spk)

            f.write(key.ljust(50) + (str(seconds_sil)).ljust(9) +
                ('('+str(round(100*(seconds_sil/seconds), 2))+'%)').ljust(9) +
                (str(round(mean_sil, 2))).ljust(9) +
                (str(round(std_sil, 2))).ljust(13) +
                (str(seconds_1spk)).ljust(9) +
                ('('+str(round(100*(seconds_1spk/seconds), 2))+'%)').ljust(9) +
                (str(round(mean_1spk, 2))).ljust(9) +
                (str(round(std_1spk, 2))).ljust(13) +
                (str(seconds_2spk)).ljust(9) +
                ('('+str(round(100*(seconds_2spk/seconds), 2))+'%)').ljust(9) +
                (str(round(mean_2spk, 2))).ljust(9) +
                (str(round(std_2spk, 2))).ljust(13) +
                (str(seconds_3spk)).ljust(9) +
                ('('+str(round(100*(seconds_3spk/seconds), 2))+'%)').ljust(9) +
                (str(round(mean_3spk, 2))).ljust(9) +
                (str(round(std_3spk, 2))).ljust(13) +
                (str(seconds_4spk)).ljust(9) +
                ('('+str(round(100*(seconds_4spk/seconds), 2))+'%)').ljust(9) +
                (str(round(mean_4spk, 2))).ljust(9) +
                (str(round(std_4spk, 2))).ljust(13)+'\n')
        f.write('ALL'.ljust(50) + (str(all_sil)).ljust(9) +
                ('('+str(round(100*(all_sil/all_seconds), 2))+'%)').ljust(9) +
                (str(round(np.mean(lengths_sil), 2))).ljust(9) +
                (str(round(np.std(lengths_sil), 2))).ljust(13) +
                (str(all_1spk)).ljust(9) +
                ('('+str(round(100*(all_1spk/all_seconds), 2))+'%)').ljust(9) +
                (str(round(np.mean(lengths_1spk), 2))).ljust(9) +
                (str(round(np.std(lengths_1spk), 2))).ljust(13) +
                (str(all_2spk)).ljust(9) +
                ('('+str(round(100*(all_2spk/all_seconds), 2))+'%)').ljust(9) +
                (str(round(np.mean(lengths_2spk), 2))).ljust(9) +
                (str(round(np.std(lengths_2spk), 2))).ljust(13) +
                (str(all_3spk)).ljust(9) +
                ('('+str(round(100*(all_3spk/all_seconds), 2))+'%)').ljust(9) +
                (str(round(np.mean(lengths_3spk), 2))).ljust(9) +
                (str(round(np.std(lengths_3spk), 2))).ljust(13) +
                (str(all_4spk)).ljust(9) +
                ('('+str(round(100*(all_4spk/all_seconds), 2))+'%)').ljust(9) +
                (str(round(np.mean(lengths_4spk), 2))).ljust(9) +
                (str(round(np.std(lengths_4spk), 2))).ljust(13) + '\n')


if __name__ == "__main__":
    # execute only if run as a script
    main()
