#!/usr/bin/env python

import logging
from optparse import OptionParser
import chr_sim_hess as utils
import pandas as pd
from transform_betas import convert_betas, truncate_matrix_neg_half
from half_ld import truncate_matrix_half
import os
import numpy as np

# set up global logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

def main():
    parser = OptionParser()
    parser.add_option("--chr", type=int)
    parser.add_option("--start", type=int)
    parser.add_option("--stop", type=int)
    parser.add_option("--bim", dest="bim_file")
    parser.add_option("--ld_dir", dest="ld_dir")
    parser.add_option("--outdir", dest="outdir")
    (options, args) = parser.parse_args()

    chr = options.chr
    start = options.start
    stop = options.stop
    bim_file = options.bim_file
    ld_dir = options.ld_dir
    outdir = options.outdir

    ld = utils.LD(ld_dir=ld_dir, chr_i=chr, legend=bim_file)

    logging.info("Reading block -- chr: %d, start: %d, stop: %d" % (chr, start, stop))

    # specify the interested locus position
    loci = pd.Series({'chr': chr, 'start': start, 'stop': stop})

    # ld_loci in (648, 648) shape
    ld_loci = ld.get_locus(loci)

    # LD half power
    ld_half_loci = truncate_matrix_half(ld_loci)
    ld_neg_half_loci = truncate_matrix_neg_half(ld_loci)

    # save files to outdir

    ld_fname = os.path.join(outdir, "chr_%d_start_%d_stop_%d.ld" % (chr, start, stop))
    logging.info("Locus file: %s" % ld_fname)
    np.save(ld_fname, ld_loci)

    ld_half_fname = os.path.join(outdir, "chr_%d_start_%d_stop_%d.half_ld" % (chr, start, stop))
    logging.info("Locus half-LD file: %s" % ld_half_fname)
    np.save(ld_half_fname, ld_half_loci)

if __name__== "__main__":
  main()
