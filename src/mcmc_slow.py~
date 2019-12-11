from unity_metropolis import *
from auxilary import *
import sys
import scipy.sparse as sp
import pandas as pd
import logging
import cProfile, pstats, StringIO

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


"""
unity_gibbs.py

Describes functions used for sampling from posterior distribution of p.
Models LD in a full Gibbs sampler.

"""
B = 10

def calc_mu_opt(gamma_old, c_old, z, a_matrix, psi, V_half, sigma_g, sigma_e, m):

    M = len(z)
    sum = 0

    """
    # calculate variance term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
    V_m_half = V_half[:, m]

    bottom_sigma_m = 1/float(sigma_g) + (1/float(sigma_e))*(np.matmul(np.transpose(V_m_half), V_m_half))
    sigma_m = 1/float(bottom_sigma_m)

    beta = np.multiply(gamma_old, c_old)

    middle_term = np.matmul(V_half, beta)

    end_term = np.multiply(V_m_half, gamma_old[m])

    r_m = z - middle_term + end_term

    # calculate mean term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
    temp_term = np.matmul(np.transpose(r_m), V_m_half)

    """
	# OPTIMIZATION
    V_m_half = V_half[:, m]
    bottom_sigma_m = 1/(sigma_g) + (1/(sigma_e))*a_matrix[m,m]
    sigma_m = 1/(bottom_sigma_m)
    W_m = V_m_half
    #psi_m = np.dot(z, W_m)
    psi_m = psi[m]
    sum = 0
    M = len(z)

	# find nonzero indicies
    nonzero_inds = np.nonzero(c_old)[0]
    for i in nonzero_inds:
        a_im = a_matrix[i,m]
        if i != m:
            sum += a_im * gamma_old[i] * c_old[i]

    temp_term = psi_m - sum

    mu_m = (sigma_m/(sigma_e))*temp_term
	# end optimization

    return mu_m



def draw_c_gamma_opt(c_old, gamma_old, p_old, sigma_g, sigma_e, V_half, z, a_matrix, psi):

    z = z.reshape(len(z))

    M = len(c_old) # number of SNPs

    # hold new values for c-vector and gamma-vector
    c_t = np.zeros(M)

    gamma_t = np.zeros(M)

    beta_m_old = 100

    # loop through all SNPs
    mu_list = []
    for m in range(0, M):

        # calculate variance term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
        V_m_half = V_half[:, m]

        bottom_sigma_m = 1/(sigma_g) + (1/(sigma_e))*(np.matmul(np.transpose(V_m_half), V_m_half))
        sigma_m = 1/(bottom_sigma_m)

        mu_m = calc_mu_opt(gamma_old, c_old, z, a_matrix, psi, V_half, sigma_g, sigma_e, m)

        # calculate params for posterior of c, where P(c|.) ~ Bern(d_m)
        try:
            var_term = math.sqrt(sigma_m/(sigma_g))
        except:
            print sigma_m
            print sigma_g

        a = 0.50 * 1 / ((sigma_m)) * mu_m * mu_m

        # check for overflow
        if a > EXP_MAX:
            a = EXP_MAX

        # Bernoulli parameter, where P(c|.) ~ Bern(d_m)
        d_m = (p_old*var_term*math.exp(a))/(p_old*var_term*math.exp(a) + (1-p_old))

        # draw c_m
        try:
            c_m = st.bernoulli.rvs(d_m)
        except:
            break

        # draw gamma_m
        if c_m == 0:
            gamma_m = 0
        else:
            gamma_m = st.norm.rvs(mu_m, math.sqrt(sigma_m))

        # update values
        c_t[m] = c_m
        gamma_t[m] = gamma_m

        c_old[m] = c_m
        gamma_old[m] = gamma_m

    #print("mu list")
    #print(mu_list)

    return c_t, gamma_t, mu_m



"""calculates the posterior density of c and gamma in block

    Args:
        c_old: causal status vector from prev iteration
        gamma_old: causal effect sizes
        p_old: proportion p
        sigma_g: genetic variance (h/M)
        sigma_e: environmental variance (1-h/N)
        V_half: LD-matrix to half power
        z: gwas effect sizes

    Returns:
        c_t: new estimate of causal vector
        gamma_t: new estimate of effect sizes

"""
def draw_c_gamma(c_old, gamma_old, p_old, sigma_g, sigma_e, V_half, z):

    M = len(c_old) # number of SNPs

    # hold new values for c-vector and gamma-vector
    c_t = np.zeros(M)

    gamma_t = np.zeros(M)

    beta_m_old = 100


    # loop through all SNPs
    mu_list = []
    for m in range(0, M):

        # calculate variance term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
        V_m_half = V_half[:, m]

        bottom_sigma_m = 1/(sigma_g) + (1/(sigma_e))*(np.matmul(np.transpose(V_m_half), V_m_half))
        try:
            sigma_m = 1/(bottom_sigma_m)
        except:
            print bottom_sigma_m

        beta = np.multiply(gamma_old, c_old)

        middle_term = np.matmul(V_half, beta)

        end_term = np.multiply(V_m_half, gamma_old[m])
        r_m = z - middle_term + end_term

        # calculate mean term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
        temp_term = np.matmul(np.transpose(r_m), V_m_half)

        mu_m = (sigma_m/float(sigma_e))*temp_term

        mu_list.append(mu_m)

        # calculate params for posterior of c, where P(c|.) ~ Bern(d_m)
        try:
            var_term = math.sqrt(sigma_m/(sigma_g))
        except:
            print sigma_m
            print sigma_g

        a = 0.50 * 1 / ((sigma_m)) * mu_m * mu_m

        # check for overflow
        if a > EXP_MAX:
            a = EXP_MAX

        # Bernoulli parameter, where P(c|.) ~ Bern(d_m)
        d_m = (p_old*var_term*math.exp(a))/float(p_old*var_term*math.exp(a) + (1-p_old))

        # draw c_m
        try:
            c_m = st.bernoulli.rvs(d_m)
        except:
            break

        # draw gamma_m
        if c_m == 0:
            gamma_m = 0
        else:
            gamma_m = st.norm.rvs(mu_m, math.sqrt(sigma_m))

        # update values
        c_t[m] = c_m
        gamma_t[m] = gamma_m

        c_old[m] = c_m
        gamma_old[m] = gamma_m

    #print("mu list")
    #print(mu_list)

    return c_t, gamma_t, mu_m


def draw_p_ivar(c):
    M = len(c)
    alpha1 =  beta_lam + np.sum(c)
    alpha2 = beta_lam + (M - np.sum(c))
    p = st.beta.rvs(alpha1, alpha2)

    return p


def gibbs_full_dp(f, z_list, N, ld_half_flist, p_init=None, c_init_list=None, gamma_init_list=None, its=5000, DP='y', H_snp=None, H_gw=None, M_gw=500000):

    # lists to hold esimtates
    p_list = []
    sigma_g_list = []
    sigma_e_list = []
    log_like_list = []
    a_matrix_list = []
    psi_list = []
    mu_list = []

    # lists to hold latent params
    gamma_t_list = []
    c_t_list = []

    B = len(z_list)

    ####################################
    ##          INITIALIZATION        ##
    ####################################

    # initialize p
    if p_init is None:
        p_t = st.beta.rvs(.2, .2)
    else:
        p_t= p_init

    p_list.append(p_t)

    # initialize sigma_g
    z_arr = np.hstack(np.asarray(z_list))

    sigma_g_t = np.var(np.abs(z_arr))
    sigma_g_list.append(sigma_g_t)

    # initialize sigma
    sigma_e_t = (1-sigma_g_t) / float(N)
    sigma_e_list.append(sigma_e_t)

    # initialize c and gamma
    for b in range(0, B):

        # read in betas from gwas file
        z_b = z_list[b]
        M_b = len(z_list[b])

        sd = math.sqrt(sigma_g_t)

        if gamma_init_list is None:
            gamma_t_b = st.norm.rvs(loc=0, scale=sd, size=M_b)
        else:
            gamma_t_b = list(np.multiply(gamma_init_list[b], c_init_list[b]))

        if c_init_list is None:
            c_t_b = st.bernoulli.rvs(p=p_old, size=M_b)
        else:
            c_t_b = list(c_init_list[b])

        # build list of blocks for next iteration
        gamma_t_list.append(gamma_t_b)
        c_t_list.append(c_t_b)

        try:
            V_half = np.loadtxt(ld_half_flist[b])
        except:
            V_half = np.load(ld_half_flist[b])

        # a-matrix initialization
        a_matrix_b = np.empty((M_b, M_b))
        for i in range(0, M_b):
			for m in range(0, M_b):
				V_half_i = V_half[:, i]
				V_half_m = V_half[:, m]
				a_im = np.dot(V_half_i, V_half_m)
				a_matrix_b[i,m] = a_im

        a_matrix_list.append(a_matrix_b)

        # psi list
        psi_b = np.zeros(M_b)
        for m in range(0, M_b):
            V_half_m = V_half[:, m]
            psi_b[m] = np.dot(z_b, V_half_m)

        psi_list.append(psi_b)


    ####################################
    ##            INFERENCE           ##
    ####################################

    # Assuming B is one block

    # get params for block b
    b=0
    z_b = np.asarray(z_list[b])
    M_b = len(z_b)
    M = M_b
    try:
        V_half_b = np.loadtxt(ld_half_flist[b])
    except:
        V_half_b = np.load(ld_half_flist[b])

    a_matrix = a_matrix_list[b]
    psi = psi_list[b]
    gamma_t_b = np.asarray(gamma_t_list[b])
    c_t_b = np.asarray(c_t_list[b])

    # start profiling
    #pr = cProfile.Profile()
    #pr.enable()

    for i in range(0, its):

        # sample c, gamma
        if DP == 'y':
            c_t_b, gamma_t_b, __ = draw_c_gamma_opt(c_t_b, gamma_t_b, p_t, sigma_g_t, sigma_e_t, V_half_b, z_b, a_matrix, psi)
        elif DP =='n':
            c_t_b, gamma_t_b, __ = draw_c_gamma(c_t_b, gamma_t_b, p_t, sigma_g_t, sigma_e_t, V_half_b, z_b)
        else:
            print("ERROR: dp flag is not valid! Exiting...")
            exit(1)

        # sample global params

        # sample p
        p_t = draw_p_ivar_gw(c_t_b)
        p_list.append(p_t)

        if H_gw is not None: # use genome-wide h2
            # set sigma_g
            sigma_g_t = H_gw/float(M_gw * p_t)
            # set sigma_e
            sigma_e_t = (1-H_gw)/float(N)

        elif H_snp is not None:
            sigma_g_t = H_snp
            sigma_g_list.append(sigma_g_t)
            h2 = M*p_t*sigma_g_t
            sigma_e_t = (1-h2)/float(N)

        else: # H_snp and H_gw are none--must infer sigma_g
            # sample sigma_g
            sigma_g_t = draw_sigma_g(gamma_t_b, c_t_b)
            sigma_g_list.append(sigma_g_t)

            # sample sigma_e
            sigma_e_t = draw_sigma_e(z_b, gamma_t_b, c_t_b)
            sigma_e_list.append(sigma_e_t)

        # calculate likelihood
        log_like_t = log_like(z_list, gamma_t_list, c_t_list, sigma_e_t, ld_half_flist)
        log_like_list.append(log_like_t)

        # print debugging-info
        if i <= 10 or i % 10 == 0:
            print("Iteration %d: p: %.4f, sigma_g: %.4g, sigma_e: %.4g, log-like: %4g") % (i, p_t, sigma_g_t, sigma_e_t, log_like_t)
            sys.stdout.flush()

    # end profile
    #pr.disable()
    #s = StringIO.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print s.getvalue()
    #print_func(s.getvalue(), f)

    # compute averages
    start = int(its*burn)
    p_est = np.mean(p_list[start: ])
    p_var = np.var(p_list[start: ])
    sigma_g_est = np.mean(sigma_g_list[start:])
    sigma_g_var = np.var(sigma_g_list[start:])
    #sigma_e_est = np.mean(sigma_e_list[start:])
    #sigma_e_var = np.var(sigma_e_list[start:])
    sigma_e_est = None
    sigma_e_var = None
    avg_log_like = np.mean(log_like_list[start:])
    var_log_like = np.var(log_like_list[start:])

    return p_est, p_var, sigma_g_est, sigma_g_var, sigma_e_est, sigma_e_var, avg_log_like, var_log_like


def draw_p_ivar_gw(c_t_arr):

    #c_t_list = np.asarray(c_t_list)
    #c_t_list = np.hstack(c_t_list)
    M = c_t_arr.size

    alpha1 = beta_lam_1 + np.sum(c_t_arr)
    alpha2 = beta_lam_2 + (M - np.sum(c_t_arr))
    try:
        p = st.beta.rvs(alpha1, alpha2)
    except:
        print c_t_list

    return p


def draw_sigma_g(gamma_t_arr, c_t_arr):
    #c_t_arr = np.hstack(np.asarray(c_t_list))
    M_c = np.sum(c_t_arr)
    alpha_g = alpha_g0 + (0.50)*M_c

    #gamma_t_arr = np.hstack(np.asarray(gamma_t_list))
    gamma_square_sum = np.sum(np.square(gamma_t_arr))
    beta_g = beta_g0 + (0.50)*gamma_square_sum

    sigma_g_t = st.invgamma.rvs(a=alpha_g, scale=beta_g)

    return sigma_g_t


def draw_sigma_e(z_arr, gamma_t_arr, c_t_arr):
    #c_t_arr = np.hstack(np.asarray(c_t_list))
    #gamma_t_arr = np.hstack(np.asarray(gamma_t_list))
    #z_arr = np.hstack(np.asarray(z_list))

    M_gw = len(c_t_arr)
    alpha_e = alpha_e0 + (0.50)*M_gw

    gamma_sum = 0
    nonzero_inds = np.nonzero(c_t_arr)[0]
    for i in nonzero_inds:
        gamma_sum += (z_arr[i] - gamma_t_arr[i]) * (z_arr[i] - gamma_t_arr[i])

    beta_e = beta_e0 + (0.50)*gamma_sum

    sigma_e_t = st.invgamma.rvs(a=alpha_e, scale=beta_e)

    return sigma_e_t


def log_like(z_list, gamma_t_list, c_t_list, sigma_e, V_half_flist):
    total_log_like = 0

    """
    for z, gamma_t, c_t, V_half_fname in zip(z_list, gamma_t_list, c_t_list, V_half_flist):
        V_half = np.loadtxt(V_half_fname)
        M = len(z)
        mu = np.matmul(V_half, np.multiply(gamma_t, c_t))
        cov = np.eye(M)*sigma_e
        log_like = st.multivariate_normal.logpdf(z, mu, cov)
        total_log_like += log_like
    """

    return total_log_like


def log_p_pdf(p_t, gamma_t, c_t, H_gw, M_gw):
    log_p = st.beta.logpdf(p, beta_lam_1, beta_lam_2)

    # find nonzero inds of gamma
    nonzero_inds = np.nonzero(gamma_t)[0]
    nonzero_gamma = np.take(gamma_t, nonzero_inds)
    sd = H_gw/(M_gw * p_t)
    log_gamma = np.sum(st.norm.logpdf(nonzero_gamma, 0, sd))

    log_c = np.sum(st.bernoulli.logpdf(c_t, p_t))

    log_pdf = log_p + log_gamma + log_c

    return log_pdf


def log_q_pdf(p_t):
    alpha = beta_lam_1*B
    beta = beta_lam_2*B
    log_q = st.beta.logpdf(x=p_t, a=alpha, b=beta)

    return log_q

def q_rvs(p):
    alpha = p+beta_lam_1*B
    beta = p+beta_lam_2*B
    q_star = st.beta.rvs(alpha, beta)
    return q_star

def accept_prob(p_old, p_star, gamma_t, c_t, H_gw, M_gw):
    log_q_star = log_q_pdf(p_star)
    log_q_old = log_q_pdf(p_old)
    log_p_star = log_p_pdf(p_star,gamma_t, c_t, H_gw, M_gw)
    log_p_old = log_p_pdf(p_old, gamma_t, c_t, H_gw, M_gw)

    r = (log_p_star - log_p_old) + (log_q_old - log_q_star)

    if r < EXP_MAX: # EXP MAX
        R = math.exp(r)
    else:
        R = 100

    accept = min(1, R)

    return accept


def sample_p_metropolis(p_t, gamma_t, c_t, H_gw, M_gw):
    p_old = p_t
    p_star = q_rvs(p_old)
    accept = accept_prob(p_old, p_star, gamma_t, c_t, H_gw, M_gw)

    u = st.uniform.rvs(size=1)
    if u < accept:
        p_t = p_star
    else:
        p_t = p_old

    return p_t
