#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import os

def vad_segments_to_bool_vec(labels, min_seg=0, length=0):
    """
    Transform label file into boolean vector representing frame labels
    Inputs:
        labels: array containing start end values (in milliseconds) for all speech segments
        min_seg: minimum segment duration, shorter ones will be discarded
        length: Output vector is truncted or augmented with False values to have this length.
                For negative 'length', it will be only augmented if shorter than '-length'.
                By default (length=0), the vector ends with the last true value.
    Output:
        frames: boolean vector
    """
    min_len, max_len = (length, length) if length > 0 else (-length, None)
    if min_seg:
        labels = labels[labels[:, 1].astype(int)-labels[:, 0].astype(int) > min_seg]
    start, end = np.rint(labels.T[:2].astype(int)).astype(int)
    if not end.size:
        return np.zeros(min_len, dtype=bool)
    frms = np.repeat(np.r_[np.tile([False, True], len(end)), False],
                     np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat,
                     max(0, min_len-end[-1])])
    assert len(frms) >= min_len and np.sum(end-start) == np.sum(frms)
    return frms[:max_len]

def hard_labels_to_lab(
    matrix: np.ndarray,
    lab_path: str,
    precision: float
):
    matrix_extended = np.concatenate((np.asarray([0]), matrix, np.asarray([0])))
    changes_dict = {}
    changes = np.where(matrix_extended[1:] - matrix_extended[:-1]
                           )[0].astype(float)
    if changes.shape[0] > 0:
        if changes[-1] == matrix.shape[0]:
            changes[-1] -= 1  # avoid reading out of array
        beg = changes[:-1]
        end = changes[1:]
        # So far, beg and end include the silences in between
        beg = beg[::2]
        end = end[::2]
        assert beg.shape[0] == end.shape[0], "Amount of beginning and \
                                           end of segments do not match."
        for pos in range(beg.shape[0]):
            time_beg = beg[pos] / precision
            time_length = (end[pos] - beg[pos]) / precision
            changes_dict[time_beg] = f"{round(time_beg, 3)}\t{round(time_beg + time_length, 3)}\tspeech\n"
    changes_dict = OrderedDict(sorted(changes_dict.items()))
    if not os.path.exists(os.path.dirname(lab_path)):
        os.makedirs(os.path.dirname(lab_path))
    with open(lab_path, 'w') as f:
        for k, v in changes_dict.items():
            f.write(v)


def lab_to_hard_labels(
    lab_path: str,
    precision: float,
    overlap: bool,
    length: float = -1
) -> Tuple[np.ndarray, List[str]]:
    """
        reads the lab and returns a Nfx1 matrix encoding the segments in
        which each speaker is present (labels 1/0) at the given precision.
        Nf is the resulting number of frames,
        according to the parameters given.
        Nf might be shorter than the real length of the utterance, as final
        silence parts cannot be recovered from the rttm.
        If length is defined (s), it is to account for that extra silence.
        The function assumes that the lab only contains speaker turns (no
        silence segments).
    """
    # each row is a turn, columns denote beginning (s) and duration (s) of turn
    data = np.loadtxt(lab_path, usecols=[0, 1])
    if data.shape[0] == 2 and len(data.shape) < 2:  # if only one segment
        data = np.asarray([data])
    # length of the file (s) that can be recovered from the lab,
    # there might be extra silence at the end
    if len(data) == 0:
        len_file = 0
    else:
        len_file = data[-1][0]+data[-1][1]
    if length > len_file:
        len_file = length

    # matrix in given precision
    if overlap:
        matrix = np.zeros([int(round(len_file*precision)), 2])
    else:
        matrix = np.zeros([int(round(len_file*precision)), 1])
    if len(data) > 0:
        # ranges to mark each turn
        ranges = np.around((np.array([data[:, 0],
                            data[:, 1]]).T*precision)).astype(int)

        for init_end in ranges: # loop over turns
            matrix[init_end[0]:init_end[1], 0] = 1  # mark the frame
            if overlap:
                matrix[init_end[0]:init_end[1], 1] = 1  # mark the frame

    if overlap:
        labels = ['speech1', 'speech2']
    else:
        labels = ['speech']
    return matrix, labels
