#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

from rttm_utils import rttm_to_hard_labels
from types import SimpleNamespace
import argparse
import numpy as np
import os

eps = 0.000001

def parse_arguments() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description='Score overlapped speech detection performance from RTTM.')
    parser.add_argument('--rttm-ref-dir', type=str, required=True,
                        help='directory with rttm files')
    parser.add_argument('--rttm-sys-dir', type=str, required=True,
                        help='directory where to save lab files')
    parser.add_argument('--txt-list-file', type=str, required=True,
                        help='file containing list of files to process')
    parser.add_argument('--lengths', type=str, required=True,
                        help='file containing list of lengths per file')
    parser.add_argument('--out-file', type=str, required=True,
                        help='file where to write output')
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
    
    if not os.path.exists(os.path.dirname(args.out_file)):
        os.makedirs(os.path.dirname(args.out_file))

    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0
    neg_total = 0
    pos_total = 0

    with open(args.out_file, 'w') as f:
        f.write('File'.ljust(10)+'Error%'.ljust(10)+'Miss%'.ljust(10)+'FA%'.ljust(10)+'|'+'F1'.ljust(10)+'Prec.'.ljust(10)+'Reca.'.ljust(10)+'|'+'Acc.%'.ljust(10)+'MissR'.ljust(10)+'FAR'.ljust(10)+'\n')
        f.write('----------------------------------------------------------------------------------------------------\n')

        for name in files_list:
            ref_matrix, _ = rttm_to_hard_labels(
                os.path.join(args.rttm_ref_dir, f"{name}.rttm"),
                1000,
                lengths[name])
            sys_matrix, _ = rttm_to_hard_labels(
                os.path.join(args.rttm_sys_dir, f"{name}.rttm"),
                1000,
                lengths[name])
            ref_matrix = ref_matrix[:min(ref_matrix.shape[0], sys_matrix.shape[0]),:]
            sys_matrix = sys_matrix[:min(ref_matrix.shape[0], sys_matrix.shape[0]),:]


            ref_matrix = (ref_matrix.sum(axis=1) > 1).astype(int)
            sys_matrix = (sys_matrix.sum(axis=1) > 1).astype(int)

            pos_pred = np.where(sys_matrix == 1)[0]
            neg_pred = np.where(sys_matrix != 1)[0]
            pos = (np.where(ref_matrix == 1)[0]).shape[0]
            neg = (np.where(ref_matrix != 1)[0]).shape[0]
            neg_total += neg
            pos_total += pos

            tp = len(np.where(ref_matrix[pos_pred] == 1)[0])
            tp_total += tp
            fp = len(np.where(ref_matrix[pos_pred] != 1)[0])
            fp_total += fp
            fn = len(np.where(ref_matrix[neg_pred] == 1)[0])
            fn_total += fn
            tn = len(np.where(ref_matrix[neg_pred] != 1)[0])
            tn_total += tn

            precision = float(tp) / (float(tp+fp)+eps)
            recall = float(tp) / (float(tp+fn)+eps)
            farate = float(fp) / (float(fp+tn)+eps)
            missrate = float(fn) / (float(tp+fn)+eps)
            f1 = 2.0*tp / (float(2*tp + fp + fn)+eps)
            fa = 100.0 * fp / (float(neg+pos)+eps)
            miss = 100.0 * fn / (float(neg+pos)+eps)
            acc = 100.0 * (tp+tn) / (tp+fp+fn+tn)   

            f.write(name.ljust(10)+str(round(miss+fa, 3)).ljust(10)+str(round(miss, 3)).ljust(10)+str(round(fa, 3)).ljust(10)+'|'+str(round(f1, 3)).ljust(10)+str(round(precision, 3)).ljust(10)+str(round(recall, 3)).ljust(10)+'|'+str(round(acc, 3)).ljust(10)+str(round(missrate, 3)).ljust(10)+str(round(farate, 3)).ljust(10)+'\n')
        
        precision = float(tp_total) / (float(tp_total+fp_total)+eps)
        recall = float(tp_total) / (float(tp_total+fn_total)+eps)
        farate = float(fp_total) / (float(fp_total+tn_total)+eps)
        missrate = float(fn_total) / (float(tp_total+fn_total)+eps)
        f1 = 2.0*tp_total / (float(2*tp_total + fp_total + fn_total)+eps)
        fa = 100.0 * fp_total / (float(neg_total+pos_total)+eps)
        miss = 100.0 * fn_total / (float(neg_total+pos_total)+eps)
        acc = 100.0 * (tp_total+tn_total) / (tp_total+fp_total+fn_total+tn_total)
        f.write('----------------------------------------------------------------------------------------------------\n')
        f.write('File'.ljust(10)+'Error%'.ljust(10)+'Miss%'.ljust(10)+'FA%'.ljust(10)+'|'+'F1'.ljust(10)+'Prec.'.ljust(10)+'Reca.'.ljust(10)+'|'+'Acc.%'.ljust(10)+'MissR'.ljust(10)+'FAR'.ljust(10)+'\n')
        f.write('OVERALL'.ljust(10)+str(round(miss+fa, 3)).ljust(10)+str(round(miss, 3)).ljust(10)+str(round(fa, 3)).ljust(10)+'|'+str(round(f1, 3)).ljust(10)+str(round(precision, 3)).ljust(10)+str(round(recall, 3)).ljust(10)+'|'+str(round(acc, 3)).ljust(10)+str(round(missrate, 3)).ljust(10)+str(round(farate, 3)).ljust(10)+'\n')
