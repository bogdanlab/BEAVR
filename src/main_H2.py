from unity_gibbs_h2 import *
from os import listdir
from os.path import isfile, join
import pandas as pd
import logging
from scipy.optimize import minimize
from unity_v3_dp import gibbs_ivar_gw_dp
import cProfile, pstats, StringIO

PROFILE=False

# global logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def neg_log_p_pdf_noLD(p, H_gw, z, M_gw, N):

    sigma_g = H_gw / float(M_gw)
    sigma_e = (1-H_gw)/float(N)

    mu = 0
    sd_1 = np.sqrt(sigma_g + sigma_e)
    sd_0 = np.sqrt(sigma_e)

    log_pdf_0 = st.norm.logpdf(x=z, loc=mu, scale=sd_0) # non-causal
    log_pdf_1 = st.norm.logpdf(x=z, loc=mu, scale=sd_1) # causal

    d_0 = np.add(log_pdf_0, np.log(1-p))
    d_1 = np.add(log_pdf_1, np.log(p))

    snp_pdfs = logsumexp_vector([d_0, d_1])

    log_pdf = np.sum(snp_pdfs)

    log_p = st.beta.logpdf(x=p, a=beta_lam,b=beta_lam)

    neg_log_pdf = -1*(log_pdf + log_p)

    return neg_log_pdf


def initial_estimates(z, H_gw, M_gw, N):

    # hold candidate starting values
    candidates = []
    densities = []

    OPTIMIZATION_ITS = 5

    for it in range(0, OPTIMIZATION_ITS):

        p0 = st.beta.rvs(beta_lam, beta_lam)
        x0 = [p0]

        result = minimize(neg_log_p_pdf_noLD, x0, tol=1e-8, method="L-BFGS-B",
                           args=(H_gw, z, M_gw, N), jac=False,
                           bounds=[(0.00001, 1.0)])

        p_est = result.x

        # calculate density with MAP estimates
        neg_density = neg_log_p_pdf_noLD(p_est, H_gw, z, M_gw, N)
        density = neg_density * (-1)

        logging.info("Candidate starting values (p): %.4g" % (p_est))
        logging.info("Desity at MAP: %.4f" % density)

        candidates.append(p_est)
        densities.append(density)

    # end for-loop through candidate values

    # pick values with greatest density
    max_index = np.argmax(densities)

    # return initialization points with best MAP
    p_est = candidates[max_index]

    return p_est


def smart_start(z_list, N):
    c_init_list = []
    M_gw = 0

    # concert betas to zscore
    for z_b in z_list:
        zscores_b = np.multiply(z_b, np.sqrt(N))
        causal_inds_list_b = np.where(zscores_b >= 5.0)
        causal_inds_b = causal_inds_list_b[0]
        M = len(z_b)
        M_gw += M
        c_init_b = np.zeros(M)
        c_init_b[causal_inds_b] = 1
        c_init_list.append(c_init_b)

    all_c_init = np.asarray(c_init_list)
    p_init = np.sum(all_c_init.flatten()) / float(M_gw)

    return p_init, c_init_list


def print_header(id, H, N, ITS, seed, gwas_dir, ld_half_dir, outdir, f):

    print_func("- - - - - - - - - - UNITY v3.0 - - - - - - - - -", f)

    print_func("Run id: %s" % id, f)
    print_func("Heritability: %.4f" % H , f)
    print_func("Sample size: %d" % N, f)
    print_func("Iterations: %d" % ITS, f)
    print_func("Seed: %d" % seed, f)
    print_func("Getting effect sizes from: %s" % gwas_dir, f)
    print_func("Using ld  from dir: %s" % ld_half_dir, f)
    print_func("Outputing simulated gwas to dir: %s" % outdir, f)

    print_func("- - - - - - - - - - - - - - - - - - - - - - - -", f)

    return


def main():

    # get input options
    parser = OptionParser()
    parser.add_option("--s", "--seed", dest="seed", default="7")
    parser.add_option("--H_snp", "--H_snp", type=float, dest="H_snp")
    parser.add_option("--H_gw", "--H_gw", type=float)
    parser.add_option("--M_gw", type=int)
    parser.add_option("--N", "--N", dest="N", default=1000)
    parser.add_option("--id", "--id", dest="id", default="unique_id")
    parser.add_option("--its", "--ITS", dest="its", default=250)
    parser.add_option("--ld_half_file", dest="ld_half_file")
    parser.add_option("--gwas_file", dest="gwas_file")
    parser.add_option("--outdir", dest="outdir")
    parser.add_option("--dp", dest="DP", default='n')
    parser.add_option("--non_inf_var", default='n')
    parser.add_option("--profile", default='n')
    parser.add_option("--full", dest="full", default='n')

    (options, args) = parser.parse_args()

    # set seed
    seed = int(options.seed)
    np.random.seed(seed)

    # get model settings
    DP = options.DP
    full = options.full
    non_inf_var = options.non_inf_var
    N = int(options.N)

    if full == 'y':
        FULL = True
    else:
        FULL = False

    profile = options.profile
    if profile == 'y':
        global PROFILE
        PROFILE = True

    ITS = int(options.its)
    id = options.id
    ld_half_file = options.ld_half_file
    gwas_file = options.gwas_file
    outdir = options.outdir

    if options.h2_gw is not None:
        h2_gw = options.h2_gw
        M_gw = options.M_gw
        H_snp = None
    elif options.H_snp is not None:
        H_snp = options.H_snp
        h2_gw = None
        M_gw = None
    else:
        logging.info("User needs to either provide h2_gw or sigma_g! Exiting...")
        exit(1)

    if non_inf_var == 'n':
        non_inf_var = False
    else:
        non_inf_var = True
        logging.info("NOTE: Using  non-infinitesimal variance paramerization")

    # get filenames for gwas and ld (assumes full path is given)
    gwas_flist = [gwas_file]
    ld_half_flist = [ld_half_file]

    # find total number of SNPs
    z_list = []
    M_gw = 0
    for gwas_file_b in gwas_flist:
        gwas_b = pd.read_table(gwas_file_b, sep=' ')
        z_b = np.asarray(gwas_b['BETA_STD_I'])
        M_gw += len(z_b)
        z_list.append(z_b)

    blocks = len(gwas_flist)
    logging.info("Found a total of %d blocks" % blocks)
    logging.info("Found a total of %d SNPs accross all files" % M_gw)

    logging.info("Estimating start of chain with zscore cutoff.")
    p_init, c_init_list = smart_start(z_list, N)
    logging.info("Initializing MCMC with starting value: p=%.4g" % p_init)

    """
    if PROFILE:
        pr = cProfile.Profile()
        pr.enable()
    """

    outfile = join(outdir, id +'.' + str(seed) + ".unity_v3.log")
    f = open(outfile, 'w')

    # run experiment
    if FULL:
        logging.info("Using FULL model")
        gamma_init_list = None
        if DP == 'y':
            logging.info("DP SPEEDUP")
            p_est, p_var, sigma_g_est, sigma_g_var, sigma_e_est, sigma_e_var, avg_log_like, var_log_like = gibbs_full_dp(f, z_list, N, ld_half_flist, p_init=p_init, c_init_list=c_init_list, gamma_init_list=gamma_init_list, its=ITS, DP=DP, H_snp=H_snp, H_gw=H_gw, M_gw=M_gw)
        elif DP == 'n':
            logging.info("SLOW VERSION")
            p_est, p_var, sigma_g_est, sigma_g_var, sigma_e_est, sigma_e_var, avg_log_like, var_log_like = gibbs_full_dp(f, z_list, N, ld_half_flist, p_init=p_init, c_init_list=c_init_list, gamma_init_list=gamma_init_list, its=ITS, DP=DP, H_snp=H_snp, H_gw=H_gw, M_gw=M_gw)
        else:
            exit(1)
    else:
        pass

    """
    if PROFILE:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
        print_func(s.getvalue(), f)
    """

    print_func("Estimate p: %.4f" % p_est, f)

    print_func("SD p: %.4g" % math.sqrt(p_var), f)

    print_func("Avg log like: %.6g" % avg_log_like, f)

    print_func("Var log like: %.4g" % math.sqrt(var_log_like), f)

    if FULL:
        print_func("Estimate sigma_g: %.4g" % sigma_g_est, f)
        print_func("SD sigma_g: %.4g" % np.sqrt(sigma_g_var), f)

#        print_func("Estimate sigma_e: %.4g" % sigma_e_est, f)
#        print_func("SD sigma_e: %.4g" % np.sqrt(sigma_e_var), f)

    f.close()

if __name__== "__main__":
  main()
