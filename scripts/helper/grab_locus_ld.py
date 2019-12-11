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
    parser.add_option("--sumstats", dest="sumstats_file")
    parser.add_option("--no_ld", action="store_true", dest="no_ld_flag", default=False)
    parser.add_option("--sim_i", type=int, default=None)
    parser.add_option("--ld_dir", dest="ld_dir")
    parser.add_option("--outdir", dest="outdir")
    (options, args) = parser.parse_args()

    chr = options.chr
    start = options.start
    stop = options.stop
    bim_file = options.bim_file
    sumstats_file = options.sumstats_file
    ld_dir = options.ld_dir
    outdir = options.outdir
    no_ld_flag = options.no_ld_flag
    sim_i = options.sim_i

    sumstats = utils.Sumstats(sumstats_path=sumstats_file)
    ld = utils.LD(ld_dir=ld_dir, chr_i=chr, legend=bim_file)

    logging.info("Reading block -- chr: %d, start: %d, stop: %d" % (chr, start, stop))

    # specify the interested locus position
    loci = pd.Series({'chr': chr, 'start': start, 'stop': stop})

    # ld_loci in (648, 648) shape
    ld_loci = ld.get_locus(loci)

    # sumstats_loci is in length 648
    sumstats_loci = sumstats.get_locus(loci)

    # LD half power
    ld_half_loci = truncate_matrix_half(ld_loci)
    ld_neg_half_loci = truncate_matrix_neg_half(ld_loci)

    # convert z to betas
    beta_std = np.divide(sumstats_loci['Z'], np.sqrt(sumstats_loci['N']))
    beta_std_I = convert_betas(beta_std, ld_neg_half_loci)

    # build dataframe
    sumstats_loci['BETA_STD'] = beta_std
    sumstats_loci['BETA_STD_I'] = beta_std_I

    # save files to outdir

    if sim_i is not None:
        loci_fname = os.path.join(outdir, "chr_%d_start_%d_stop_%d_sim_%d.loci" % (chr, start, stop, sim_i))
        logging.info("Locus file: %s" % loci_fname)
        ld_half_fname = os.path.join(outdir, "chr_%d_start_%d_stop_%d_sim_%d.half_ld" % (chr, start, stop, sim_i))
        logging.info("Locus half-LD file: %s" % ld_half_fname)
        np.save(ld_half_fname, ld_half_loci)
    else:
        loci_fname = os.path.join(outdir, "chr_%d_start_%d_stop_%d.loci" % (chr, start, stop))
        logging.info("Locus file: %s" % loci_fname)
        ld_half_fname = os.path.join(outdir, "chr_%d_start_%d_stop_%d.half_ld" % (chr, start, stop))
        logging.info("Locus half-LD file: %s" % ld_half_fname)
        np.save(ld_half_fname, ld_half_loci)

    sumstats_loci.to_csv(loci_fname, sep=' ', index=False)


if __name__== "__main__":
  main()
