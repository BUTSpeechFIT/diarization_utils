#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")


from rttm_utils import rttm_to_hard_labels, hard_labels_to_rttm


def test_rttms():
    length = 1113.845375
    matrix, labels = rttm_to_hard_labels('examples/ES2011a.rttm', 1000, length)
    hard_labels_to_rttm(matrix, labels, 'ES2011a', 'examples/newES2011a.rttm', 1000)
    newmatrix, newlabels = rttm_to_hard_labels('examples/ES2011a.rttm', 1000, length)
    assert (matrix == newmatrix).all(), "RTTM matrices differ"
    assert (labels == newlabels).all(), "RTTM labels differ"


if __name__ == '__main__':
    test_rttms()
