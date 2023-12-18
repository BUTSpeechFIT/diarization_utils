#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
from optparse import OptionParser
import os
from types import SimpleNamespace
import argparse
import numpy as np
from sklearn.utils import resample

# From https://github.com/luferrer/ConfidenceIntervals
def get_bootstrap_indices(N, conditions=None):
    """ Method that returns the indices for selecting a bootstrap set.
    - num_samples: number of samples in the original set
    - conditions: integer array indicating the condition of each of those samples (in order)
    If conditions is None, the indices are obtained by sampling an array from 0 to N-1 with 
    replacement. If conditions is not None, the indices are obtained by sampling conditions first
    and then retrieving the sample indices corresponding to the selected conditions.
    """

    indices = np.arange(N)

    if conditions is not None:
        if len(conditions) != N:
            raise Exception("The number of conditions should be equal to N, the first argument")
        unique_conditions = np.unique(conditions)
        bt_conditions = resample(unique_conditions, replace=True, n_samples=len(unique_conditions))
        sel_indices = np.concatenate([indices[np.where(conditions == s)[0]] for s in bt_conditions])
    else:
        sel_indices = resample(indices, replace=True, n_samples=N)
        
    return sel_indices


# From https://github.com/luferrer/ConfidenceIntervals
def get_conf_int(vals, alpha):
    """ Method to obtain the confidence interval from an array of metrics obtained from bootstrapping
    """
    return np.percentile(vals, alpha/2), np.percentile(vals, 100-alpha/2)


def parse_arguments() -> SimpleNamespace:
    parser = argparse.ArgumentParser(
        description='Compute the mean speaker counting error for the amount of speakers found')
    parser.add_argument('--results-file', type=str, required=True,
                        help='file containing the results')
    parser.add_argument('--out-file', type=str, required=True,
                        help='output file where results are written')
    parser.add_argument('--alpha', type=float, required=False,
                        default=5, help="alpha value for confidence interval")
    parser.add_argument('--num-samples', type=float, required=False,
                        default=10000, help="number of samples used to estimate the interval")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.out_file)):
        os.makedirs(os.path.dirname(args.out_file))

    contents = [line.rstrip().split() for line in open(args.results_file, 'r')]
    normalizer = 0
    for i in range(len(contents)):
        # The format expected is either
        # file1 21.27   2311.97
        # file2 18.63   2036.79
        # file3 35.70   2376.45
        # file4 14.08   2011.85
        # for DERs where the second column has the DER value 
        # and the third column has the length of the file.
        # Or the following for metrics that are not weighted by length and
        # where all files weight the same (such as speaker counting error)
        # file1 1
        # file2 1
        # file3 1
        # file4 0
        if len(contents[i]) == 3:
            normalizer += float(contents[i][2])
        else: # just take name and second column as value adding a weight of 1
            contents[i] = [contents[i][0], contents[i][1], 1.0]
            normalizer += 1

    mvals = []
    for nb in np.arange(args.num_samples):
        # conditions are not used as usually for diarization 
        # speakers' true identities are not labeled
        indices = get_bootstrap_indices(len(contents))
        weighted_errors = [float(elem[1])*float(elem[2]) for elem in np.asarray(contents)[indices]]
        mvals.append(sum(weighted_errors) / normalizer)

    error = sum([float(elem[1])*float(elem[2]) for elem in np.asarray(contents)]) / normalizer
    min_limit, max_limit = get_conf_int(mvals, args.alpha)
    with open(args.out_file, 'w') as f:
        f.write("Bootstrapping confidence interval\n")
        f.write(f"Error:  {error:.2f}     Confidence interval: {min_limit:.2f}  {max_limit:.2f}\n")


if __name__ == "__main__":
    # execute only if run as a script
    main()
