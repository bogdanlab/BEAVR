#!/usr/bin/env python

from optparse import OptionParser
import numpy as np
import scipy.stats as st
import math
import sys
import os
import logging
import pandas as pd

# set up global logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def truncate_eigenvalues(d):
    M = len(d)

    # order evaules in descending order
    d[::-1].sort()

    #running_sum = 0
    d_trun = np.zeros(M)

    # keep only positive evalues
    for i in range(0,M):
        if d[i] > 0:
            # keep evalue
            d_trun[i] = d[i]

    return d_trun


def truncate_matrix_half(V):
    # make V pos-semi-def
    d, Q = np.linalg.eigh(V, UPLO='U')

    # reorder eigenvectors from inc to dec
    idx = d.argsort()[::-1]
    Q[:] = Q[:, idx]
    #d[:] = d.argsort()[::-1]

    # truncate small eigenvalues for stability
    d_trun = truncate_eigenvalues(d)

    d_trun_half = np.sqrt(d_trun)

    # mult decomp back together to get final V_trunc
    M1 = np.matmul(Q, np.diag(d_trun_half))
    V_trun_half = np.matmul(M1, np.matrix.transpose(Q))

    return V_trun_half


def main():
    parser = OptionParser()
    parser.add_option("--ld_file", dest="ld_file")
    parser.add_option("--ld_out", dest="ld_out")

    (options, args) = parser.parse_args()

    ld_file = options.ld_file
    ld_out = options.ld_out

    ld_file_b = ld_file

    try:
        ld_b = np.loadtxt(ld_file_b)
    except: # try npy file instaed
        ld_b = np.load(ld_file_b)

    logging.info("Taking 1/2 power of ld matrix: %s" % os.path.basename(ld_file_b))

    ld_half_b = truncate_matrix_half(ld_b)

    ld_file_base = os.path.basename(ld_file_b)
    ld_string = ld_file_base.split('.')

    ld_half_fname = ld_out

    np.save(ld_half_fname ,ld_half_b)



if __name__== "__main__":
  main()
