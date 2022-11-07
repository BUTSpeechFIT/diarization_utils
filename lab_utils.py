#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

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
