#!/usr/bin/env python

from optparse import OptionParser
import numpy as np
import scipy.stats as st
import math
import os
import logging
import pandas as pd


# setup global logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


"""
Simulate GWAS summary statistics from generative model using a pre-specified LD-matrix
p_sim: regional polygenicity parameter
sigma_g: regional genetic variance parameter (h2/M*p)
sigma_e: environmental variance parameter (1-h2/N)
N: sample size
V: LD matrix (numpy array)

beta_hat: simulated GWAS effect sizes
beta_true: true effect sizes 
"""
def simulate_block_LD(p_sim, sigma_g, sigma_e, N, V):

    # get number of SNPs defined by size of LD matrix
    M = V.shape[0]

    # draw casual status vector for the region
    c = st.bernoulli.rvs(p=p_sim, size=M)

    # define slab parameters and true effect sizes
    sd = math.sqrt(sigma_g)
    gamma = st.norm.rvs(loc=0, scale=sd, size=M)
    beta = np.multiply(gamma, c)

    # simulate GWAS effect sizes from true effect sizes 
    mu = np.matmul(V, beta)
    cov = np.multiply(V, sigma_e)
    beta_hat = st.multivariate_normal.rvs(mean=mu, cov=cov)
    beta_true = beta 

    return beta_hat, beta_true



def main():

    parser = OptionParser()
    parser.add_option("--sigma_g", dest="sigma_g", type=float)
    parser.add_option("--h2_gw", dest="h2_gw", type=float)
    parser.add_option("--M_gw", dest="M_gw", type=int, default=500000)
    parser.add_option("--p_sim", dest="p_sim", default=0.05)
    parser.add_option("--N", dest="N", default=100000)
    parser.add_option("--outdir", dest="outdir")
    parser.add_option("--seed", dest="seed", default=100)
    parser.add_option("--ld_file", dest="ld_file")
    parser.add_option("--ld", type=int, dest="ld_flag")
    parser.add_option("--rsid_file", dest="rsid_file")

    (options, args) = parser.parse_args()

    p_sim = float(options.p_sim)
    N = int(options.N)
    outdir = options.outdir
    ld_file = options.ld_file
    ld_flag = options.ld_flag
    rsid_file = options.rsid_file
    seed = int(options.seed)

    try: # npy object 
        V = np.load(ld_file)
    except: # text file 
        V = np.loadtxt(ld_file)

    # number of SNPs defined by size of LD block 
    M = V.shape[0]

    # user specifies genome-wide heritability parameter-- regional heritability will be computed by scaling
    # gw-h2 by the number of SNPs in the region 
    if options.h2_gw is not None: # simulate from genome-wide h2
        logging.info("Simulating with genome-wide heritability...will calculate per-SNP sigma_g")
        h2_gw = options.h2_gw
        M_gw = options.M_gw

        # compute per-SNP h2
        sigma_g = h2_gw/float(M_gw * p_sim)
        h2_region = h2_gw * (M/float(M_gw))
        sigma_e = (1 - h2_region) / float(N)
        gwas_fname = os.path.join(outdir,
        "p_{}_h2_{}_N_{}_ld_{}_M_{}_{}.gwas".format(p_sim, h2_gw, N, ld_flag, M, seed))

    # user specifies per-SNP heritabilty-- regional heritability will be computed by summing the per-variance across SNPs
    elif options.sigma_g is not None:
        logging.info("User provided per-SNP sigma_g")
        sigma_g = options.sigma_g
        
        # compute regional heritability based on number of causal SNPs 
        h2_region = sigma_g * M * p_sim
        sigma_e = (1 - h2_region) / float(N)
        gwas_fname = os.path.join(outdir,
            "p_{}_sigG_{}_N_{}_ld_{}_M_{}_{}.gwas".format(p_sim, sigma_g, N, ld_flag, M, seed))

    else:
        logging.info("User needs to either provide h2_gw or sigma_g! Exiting...")
        exit(1)

    # set seed for replication
    np.random.seed(seed)

    # simulate GWAS sumstats and keep true effect sizes 
    beta_std, beta_true = simulate_block_LD(p_sim, sigma_g, sigma_e, N, V)

    # compute Z scores 
    z = np.multiply(beta_std, np.sqrt(N))

    # add rsid column if pre-specified 
    if rsid_file is not None:
        rsid_series = pd.read_csv(rsid_file, header=None)
        rsid_list = rsid_series.values
        rsid_list = rsid_list.flatten()

    else: # add dummy rsids 
        rsid_list = ['rs000']*M

    # add sample size column 
    N_list = np.repeat(N, M)

    # add all columns to make sumstats file 
    gwas_dict = {'snp': rsid_list, 'BETA_STD': beta_std, 'z': z, 'n': N_list, 'BETA_TRUE': beta_true}
    gwas = pd.DataFrame(gwas_dict)

    # save sumstats file 
    gwas.to_csv(gwas_fname, sep=' ', index=False)

    logging.info("FINISHED simulating")
    logging.info("Simulations can be found at: %s" % gwas_fname)


if __name__== "__main__":
  main()
