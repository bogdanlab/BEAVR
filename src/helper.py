"""
helper.py

Describes helper functions and constants used in main.py and mcmc.py

"""

import sys
import numpy as np


def print_header(id, H_gw, N, ITS, seed, gwas_file, ld_half_file, outdir, f):
    """Prints header to console summarizing user inputs

    Prints a summary of user-inputs to the console.

    Args:
        id: id for each run; can be used to keep track of experiment parameters
        H_gw: genome-wide heritability parameter
        N: sample size
        ITS: number of MCMC iterations
        seed: integer for setting random seed
        gwas_file: txt file containing GWAS effect sizes
        ld_half_file: txt or npy file containing pre-computed 1/2 power of LD matrix
        outdir: path to directory which will hold results
        f: file handler for log file

    Returns:
        does not return
    """

    print_func("- - - - - - - - - - UNITY v3.0 - - - - - - - - -", f)

    print_func("Run id: %s" % id, f)
    print_func("Heritability: %.4f" % H_gw , f)
    print_func("Sample size: %d" % N, f)
    print_func("Iterations: %d" % ITS, f)
    print_func("Seed: %d" % seed, f)
    print_func("Getting effect sizes from: %s" % gwas_file, f)
    print_func("Using ld  from dir: %s" % ld_half_file, f)
    print_func("Outputing simulated GWAS to dir: %s" % outdir, f)

    print_func("- - - - - - - - - - - - - - - - - - - - - - - -", f)

    return


def print_func(line, f):
    """Prints formatted string to both a file handler and to stdout

    Takes a formatted string (C-style formatting) and prints to both
    a specified file handler and stdout.

    Args:
        line: C-style formatted string (i.e. "Animal: %s" % "Platypus")
        f: file-handler for output file

    Returns:
        does not return
    """

    print(line)
    sys.stdout.flush()
    f.write(line)
    f.write('\n')
    return


def smart_start(z, N):
    # concert betas to zscore
    z_thresh = 3.5
    zscores = np.multiply(z, np.sqrt(N))
    causal_inds_list = np.where(zscores >= z_thresh)
    causal_inds = causal_inds_list[0]
    M = len(z)
    c_init = np.zeros(M)
    c_init[causal_inds] = 1

    p_init = np.sum(c_init.flatten()) / float(M)

    return p_init, c_init



