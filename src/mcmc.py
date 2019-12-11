"""
mcmc.py

# TODO:

"""
import scipy.stats as st
import numpy as np
import math
import sys
import cProfile
import pstats
import StringIO
import mcmc_slow 

from globals import beta_lam_1, beta_lam_2, EXP_MAX, BURN, metropolis_factor
from helper import smart_start, print_func


def gibbs_sampler_Hgw(z, H_gw, M_gw, N, V_half, its, f, DP_flag, profile_flag):

    ####################################
    ##          INITIALIZATION        ##
    ####################################

    p_list = []
    gamma_t_list = []
    c_t_list = []
    log_like_list = []
    accept_sum = 0

    p_t, c_t = smart_start(z, N)
    if p_t == 0:
        p_t = 0.001
    p_list.append(p_t)

    M = len(z)
    sigma_g = H_gw/(M_gw * p_t)
    sd = np.sqrt(sigma_g)

    gamma_t = st.norm.rvs(loc=0, scale=sd, size=M)
    gamma_t = list(np.multiply(gamma_t, c_t))

    gamma_t_list.append(gamma_t)
    c_t_list.append(c_t)

    a_matrix = np.matmul(V_half, V_half)
    psi = np.zeros(M)
    for m in range(0, M):
        V_half_m = V_half[:, m]
        psi[m] = np.dot(z, V_half_m)

    ####################################
    ##              MCMC              ##
    ####################################

    if profile_flag:
        pr = cProfile.Profile()
        pr.enable()

    for i in range(0, its):

        ### Update sigma_g, sigma_e ###

        sigma_g = H_gw / float(M_gw * p_t)
        sigma_e = (1 - H_gw) / float(N)

        ####### Sample c, gamma ######

        # TODO: go through draw_c_gamma
        if DP_flag:
            c_t, gamma_t = draw_c_gamma_opt(c_t, gamma_t, p_t, sigma_g, sigma_e, V_half, z, a_matrix, psi)
        else:
            # TODO: put slow version here
            c_t, gamma_t = mcmc_slow.draw_c_gamma(c_t, gamma_t, p_t, sigma_g, sigma_e, V_half, z)

        ########## Sample p #########

        p_t, accept = sample_p_metropolis(p_t, gamma_t, c_t, H_gw, M_gw)
        p_list.append(p_t)
        accept_sum += accept

        ######## Likelihood ########

        # TODO: add like function
        log_like_t = log_like(z, gamma_t, c_t, sigma_e, V_half)
        log_like_list.append(log_like_t)

        if i <= 10 or i % 10 == 0:
            # TODO: print accept or not
            print("Iteration %d: , Accept: %d, p: %.4f, log-like: %4g") % (i, accept, p_t, log_like_t)
            sys.stdout.flush() # flush the buffer to print on Hoffman

    ####################################
    ##           Summarize            ##
    ####################################

    if profile_flag:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
        print_func(s.getvalue(), f)

    start = int(its * BURN)
    p_est = np.mean(p_list[start:])
    p_var = np.var(p_list[start:])
    avg_log_like = np.mean(log_like_list[start:])
    var_log_like = np.var(log_like_list[start:])
    accept_percent = accept_sum/float(its)

    return p_est, p_var, avg_log_like, var_log_like, accept_percent


# TODO: add docstrings
def draw_c_gamma_opt(c_old, gamma_old, p_old, sigma_g, sigma_e, V_half, z, a_matrix, psi):

    z = z.reshape(len(z))
    M = len(c_old) # number of SNPs

    # hold new values for c-vector and gamma-vector
    c_t = np.zeros(M)
    gamma_t = np.zeros(M)

    # loop through all SNPs
    for m in range(0, M):

        # calculate variance term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
        V_m_half = V_half[:, m]

        bottom_sigma_m = 1/(sigma_g) + (1/sigma_e)*np.matmul(np.transpose(V_m_half), V_m_half)
        sigma_m = 1/bottom_sigma_m

        mu_m = calc_mu_opt(gamma_old, c_old, a_matrix, psi, sigma_g, sigma_e, m)

        # calculate params for posterior of c, where P(c|.) ~ Bern(d_m)

        var_term = math.sqrt(sigma_m/sigma_g)

        a = 0.50 * 1 / ((sigma_m)) * mu_m * mu_m

        # check for overflow
        if a > EXP_MAX:
            a = EXP_MAX

        # Bernoulli parameter, where P(c|.) ~ Bern(d_m)
        d_m = (p_old*var_term*math.exp(a))/(p_old*var_term*math.exp(a) + (1-p_old))

        # draw c_m
        c_m = st.bernoulli.rvs(d_m)

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

    return c_t, gamma_t


def calc_mu_opt(gamma_old, c_old, a_matrix, psi, sigma_g, sigma_e, m):

    bottom_sigma_m = 1/sigma_g + (1/sigma_e)*a_matrix[m, m]
    sigma_m = 1/bottom_sigma_m
    psi_m = psi[m]
    running_sum = 0

    # find nonzero indicies
    nonzero_inds = np.nonzero(c_old)[0]
    for i in nonzero_inds:
        a_im = a_matrix[i, m]
        if i != m:
            running_sum += a_im * gamma_old[i] * c_old[i]

    temp_term = psi_m - running_sum

    mu_m = (sigma_m/sigma_e)*temp_term

    return mu_m


def log_p_pdf(p_t, gamma_t, c_t, H_gw, M_gw):
    log_p = st.beta.logpdf(p_t, beta_lam_1, beta_lam_2)

    # find nonzero inds of gamma
    nonzero_inds = np.nonzero(gamma_t)[0]
    nonzero_gamma = np.take(gamma_t, nonzero_inds)
    sd = math.sqrt(H_gw/(M_gw * p_t))
    log_gamma = np.sum(st.norm.logpdf(nonzero_gamma, 0, sd))

    log_c = np.sum(st.bernoulli.logpmf(c_t, p_t))

    log_pdf = log_p + log_gamma + log_c

    return log_pdf


def log_q_pdf(p_star, p_old):
    alpha = beta_lam_1 + p_old * metropolis_factor
    beta = beta_lam_2 + p_old * metropolis_factor
    log_q = st.beta.logpdf(x=p_star, a=alpha, b=beta)

    return log_q


def q_rvs(p_old):
    alpha = beta_lam_1 + p_old*metropolis_factor
    beta = beta_lam_2 + p_old*metropolis_factor
    q_star = st.beta.rvs(alpha, beta)
    return q_star


def accept_prob(p_old, p_star, gamma_t, c_t, H_gw, M_gw):
    log_q_star = log_q_pdf(p_star, p_old)
    log_q_old = log_q_pdf(p_old, p_star)
    log_p_star = log_p_pdf(p_star, gamma_t, c_t, H_gw, M_gw)
    log_p_old = log_p_pdf(p_old, gamma_t, c_t, H_gw, M_gw)



    # if any densities are inf/-inf then reject
    if np.inf in [log_q_star, log_q_old, log_p_star, log_p_old]:
        R = 0
    elif -1*(np.inf) in [log_q_star, log_q_old, log_p_star, log_p_old]:
        R = 0
    else:
        r = (log_p_star - log_p_old) + (log_q_old - log_q_star)

    if r < EXP_MAX:
        R = math.exp(r)
    else:
        R = EXP_MAX

    accept = min(1, R)

    return accept


def sample_p_metropolis(p_t, gamma_t, c_t, H_gw, M_gw):
    p_old = p_t
    p_star = q_rvs(p_t)

    if p_star == 0.0 or p_star == 1.0:
        p_t = p_old
        accept = 0
        return p_t, accept

    else:
        accept = accept_prob(p_old, p_star, gamma_t, c_t, H_gw, M_gw)

        u = st.uniform.rvs(size=1)
        if u < accept:
            p_t = p_star
            accept = 1
        else:
            p_t = p_old
            accept = 0

        return p_t, accept


def log_like(z, gamma_t, c_t, sigma_e, V_half):
    """
    M = len(z)
    mu = np.matmul(V_half, np.multiply(gamma_t, c_t))
    cov = np.eye(M)*sigma_e
    total_log_like = st.multivariate_normal.logpdf(z, mu, cov)
    """
    total_log_like = 0
    return total_log_like
