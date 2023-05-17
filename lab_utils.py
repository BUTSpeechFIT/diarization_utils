#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

from collections import OrderedDict
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
