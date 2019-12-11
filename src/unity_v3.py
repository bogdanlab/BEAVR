from unity_gibbs import *
from os import listdir
from os.path import isfile, join
import pandas as pd
import logging
from scipy.optimize import minimize

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
        causal_inds_list_b = np.where(zscores_b >= 3.0)
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
    parser.add_option("--H", "--H", dest="H")
    parser.add_option("--N", "--N", dest="N", default=1000)
    parser.add_option("--id", "--id", dest="id", default="unique_id")
    parser.add_option("--its", "--ITS", dest="its", default=250)
    parser.add_option("--ld_half_dir", dest="ld_half_dir")
    parser.add_option("--gwas_dir", dest="gwas_dir")
    parser.add_option("--ld_half_ext", dest="ld_half_ext", default="half_ld")
    parser.add_option("--gwas_ext", dest="gwas_ext", default="gwas")
    parser.add_option("--outdir", dest="outdir")
    parser.add_option("--dp", dest="DP", default='y')
    (options, args) = parser.parse_args()

    # set seed
    seed = int(options.seed)
    np.random.seed(seed)

    # get experiment params
    H = float(options.H)
    N = int(options.N)
    ITS = int(options.its)
    id = options.id
    ld_half_dir = options.ld_half_dir
    gwas_dir = options.gwas_dir
    ld_half_ext = options.ld_half_ext
    gwas_ext = options.gwas_ext
    outdir = options.outdir
    DP = options.DP

    outfile = join(outdir, id +'.' + str(seed) + ".log")
    f = open(outfile, 'w')

    print_header(id, H, N, ITS, seed, gwas_dir, ld_half_dir, outdir, f)

    f.close()

    logging.info("Assuming files with %s extension are gwas" % gwas_ext)
    logging.info("Assuming files with %s extension are 1/2 power ld matricies" % ld_half_ext)

    # get filenames for gwas and ld
    gwas_flist = [f for f in listdir(gwas_dir) if f.endswith('.' + gwas_ext)]
    ld_half_flist = [f for f in listdir(ld_half_dir) if f.endswith('.'+ld_half_ext)]

    # append full paths
    gwas_flist[:] = [gwas_dir + '/' + f for f in gwas_flist]
    ld_half_flist[:] = [ld_half_dir + '/' + f for f in ld_half_flist]

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

    # run experiment
    H_snp = H/float(M_gw)
    H_gw = H
    #z_arr = np.array(z_list)
    #z = z_arr.flatten()

    logging.info("Estimating start of chain with zscore cutoff.")
    p_init, c_init_list = smart_start(z_list, N)
    logging.info("Initializing MCMC with starting value: p=%.4g" % p_init)

    gamma_init_list = z_list

    if DP == 'n':
        p_est, p_var, p_list, avg_log_like, var_log_like = gibbs_ivar_gw(z_list, H_snp, H_gw, N, ld_half_flist, p_init=p_init, c_init_list=c_init_list, gamma_init_list=z_list, its=ITS)
    else:
        continue

    # log results to log file
    f = open(outfile, 'w')
    print_func("Estimate p: %.4f" % p_est, f)

    print_func("SD p: %.4g" % math.sqrt(p_var), f)

    print_func("Avg log like: %.6g" % avg_log_like, f)

    print_func("SD log like: %.4g" % math.sqrt(var_log_like), f)

    f.close()

if __name__== "__main__":
  main()
