from unity_metropolis import *
from auxilary import *
import sys
import scipy.sparse as sp
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


"""
unity_gibbs_full_backup.py

This contains all of the original functions before cleaning out the deprecated
functions for the Gibbs sampler.
"""

"""calculates the log joint posterior probability

    Args:
        p_est: proportion p
        c_est: casual vector
        gamma_est: causal effect sizes
        h: heritability
        z: gwas effect sizes
        N: number of individuals
        V_half: LD-matrix to half power

    Returns:
        log_pdf: log joint posterior density

"""
def joint_prob(p_est, c_est, gamma_est, h, z, N, V_half):

    M = len(z) # number of SNPs
    sigma_e = (1-h)/float(N) # environmental noise term

    # P(z | c, gamma) - likelihood
    mu_z = np.matmul(V_half, np.multiply(gamma_est, c_est))
    cov_z = np.multiply(np.eye(M), sigma_e)
    log_z_pdf = st.multivariate_normal.logpdf(z, mean=mu_z, cov=cov_z, allow_singular=True)

    # P(gamma | p, h) - causal effects
    mu_gamma = 0
    sigma_gamma = h/float(M*p_est)
    log_gamma = st.norm.logpdf(gamma_est, mu_gamma, np.sqrt(sigma_gamma))
    log_gamma_pdf = np.sum(log_gamma)

    # P(c | p) - causal vector
    log_c = st.bernoulli.logpmf(c_est, p_est)
    log_c_pdf = np.sum(log_c)

    # P(p) - propportion of causal SNPs
    log_p_pdf = st.beta.logpdf(p_est, beta_lam, beta_lam)

    # add log densities
    log_pdf = log_z_pdf + log_gamma_pdf + log_c_pdf + log_p_pdf

    return log_pdf


"""calculates the log joint posterior probability with infinitesimal variance (sigma_g = h/M)

    Args:
        p_est: proportion p
        c_est: casual vector
        gamma_est: causal effect sizes
        h: heritability
        z: gwas effect sizes
        N: number of individuals
        V_half: LD-matrix to half power

    Returns:
        log_pdf: log joint posterior density

"""
def joint_prob_ivar(p_est, c_est, gamma_est, h_est, z, N, V_half):

    M = len(z) # number of SNPs
    sigma_e = (1-h_est)/float(N)

    # P(data | . )
    mu_z = np.matmul(V_half, np.multiply(gamma_est, c_est))
    cov_z = np.multiply(np.eye(M), sigma_e)
    log_z_pdf = st.multivariate_normal.logpdf(z, mean=mu_z, cov=cov_z, allow_singular=True)

    # P(gamma | .)
    mu_gamma = 0
    sigma_gamma = h_est/float(M)
    log_gamma = st.norm.logpdf(gamma_est, mu_gamma, np.sqrt(sigma_gamma))
    log_gamma_pdf = np.sum(log_gamma)

    # P(c | .)
    log_c = st.bernoulli.logpmf(c_est, p_est)
    log_c_pdf = np.sum(log_c)

    # P(p | .)
    log_p_pdf = st.beta.logpdf(p_est, beta_lam, beta_lam)

    # P(h | .)
    log_h_pdf = st.beta.logpdf(h_est, a=sigma_prior_a, b=sigma_prior_b)

    log_pdf = log_z_pdf + log_gamma_pdf + log_c_pdf + log_p_pdf + log_h_pdf

    return log_pdf


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

        bottom_sigma_m = 1/float(sigma_g) + (1/float(sigma_e))*(np.matmul(np.transpose(V_m_half), V_m_half))
        sigma_m = 1/float(bottom_sigma_m)

        beta = np.multiply(gamma_old, c_old)
        #print("Beta:")
        #print(beta)

        if m > 0:
            beta_m = beta[m-1]
        else:
            beta_m = 0

        if beta_m_old != beta_m:
            middle_term = np.matmul(V_half, beta)
            #middle_term = sp.coo_matrix.multiply(V_half, beta)


        end_term = np.multiply(V_m_half, gamma_old[m])
        r_m = z - middle_term + end_term

        if m > 0:
            beta_m_old = beta[m]
        else:
            beta_m_old = 10

        # calculate mean term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
        temp_term = np.matmul(np.transpose(r_m), V_m_half)

        mu_m = (sigma_m/float(sigma_e))*temp_term

        mu_list.append(mu_m)

        # calculate params for posterior of c, where P(c|.) ~ Bern(d_m)
        var_term = math.sqrt(sigma_m/float(sigma_g))

        a = 0.50 * 1 / (float(sigma_m)) * mu_m * mu_m

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

    return c_t, gamma_t



"""calculates the posterior density of p

    Args:
        z: gwas effect sizes
        p: proportion p
        h: heritability
        gamma: causal effect sizes
        N: sample size
        V_half: LD-matrix to half power

    Returns:
        log_p_pdf: log posterior density P(p|.)

"""
def log_p_pdf_ld(z, p, h, c, gamma, N, V_half):

    M = len(c) # number of SNPs

    alpha1 = beta_lam + np.sum(c)
    alpha2 = beta_lam + (M - np.sum(c))

    sigma_g = h/float(M*p)
    gamma_std = math.sqrt(sigma_g)

    log_d_gamma = np.sum(st.norm.logpdf(gamma, 0, gamma_std))

    log_d_p = st.beta.logpdf(x=p, a=alpha1, b=alpha2)

    log_p_pdf = log_d_gamma + log_d_p


    return log_p_pdf


"""calculates the posterior density of p with infinitesimal variance (h/M)

    Args:
        z: gwas effect sizes
        p: proportion p
        h: heritability
        gamma: causal effect sizes
        N: sample size
        V_half: LD-matrix to half power

    Returns:
        log_p_pdf: log posterior density P(p|.) with infinitesimal variance (h/M)

"""
def log_p_pdf_ivar(z, p, h, c, gamma, N, V_half):
    M = len(c)

    alpha1 = beta_lam + np.sum(c)
    alpha2 = beta_lam + (M - np.sum(c))

    log_d_p = st.beta.logpdf(x=p, a=alpha1, b=alpha2)

    log_p_pdf = log_d_p

    return log_p_pdf


def log_h_pdf_ivar(z, p, h, c, gamma, N, V_half):

    M = len(c)
    sigma_g = h / float(M)
    gamma_std = math.sqrt(sigma_g)
    sigma_e = (1-h)/float(N)

    beta_mu = np.matmul(V_half, np.multiply(gamma, c))
    beta_cov = np.multiply(sigma_e, np.eye(M))
    log_d_beta = st.multivariate_normal.logpdf(z, beta_mu, beta_cov)

    # only look at effects where c = 1
    gamma = np.asarray(gamma)
    gamma_causal = gamma[np.nonzero(gamma)]
    log_d_gamma = np.sum(st.norm.logpdf(gamma_causal, 0, gamma_std))
    log_d_sigma = st.beta.logpdf(h, a=sigma_prior_a, b=sigma_prior_b)

    log_pdf = log_d_beta + log_d_gamma + log_d_sigma

    return log_pdf


"""samples p from conditional posterior using Metropolis update

    Args:
        z: gwas effect sizes
        p_old: proportion p
        p_alpha_old:
        h: heritability
        c: causal vector
        N: sample size
        V_half: LD-matrix to half power

    Returns:
        p_t: sample from posterior of p

"""
def draw_p(z, p_old, h, c, gamma, N, V_half):

    # draw candidate p_star
    p_star = q_p_rvs(p_old)

    # acceptance probability
    log_q_star = log_q_pdf(p_star, p_old)

    log_q_old = log_q_pdf(p_old, p_star)

    log_p_star = log_p_pdf_ld(z, p_star, h, c, gamma, N, V_half)

    log_p_old = log_p_pdf_ld(z, p_old, h, c, gamma, N, V_half)

    # compute acceptance ratio
    r = (log_p_star - log_p_old) + (log_q_old - log_q_star)

    # check for exp overflow
    try:
        R = math.exp(r)
    except:  # overflow
        R = 100

    accept_p = min(1, R)

    # draw random number for acceptance
    u = st.uniform.rvs(size=1)

    if u < accept_p:
        p_t = p_star
        #print("ACCEPT")
    else:
        p_t = p_old
        #print("REJECT")

    return p_t


def draw_p_ivar(c):
    M = len(c)
    alpha1 =  beta_lam + np.sum(c)
    alpha2 = beta_lam + (M - np.sum(c))
    p = st.beta.rvs(alpha1, alpha2)

    return p


def draw_h_ivar(z, p, h_old, c, gamma, N, V_half):

    # draw candidate p_star
    h_star = q_h_rvs(h_old)

    # acceptance probability
    log_q_star = log_q_h_pdf(h_star, h_old)

    log_q_old = log_q_h_pdf(h_old, h_star)

    log_p_star = log_h_pdf_ivar(z, p, h_star, c, gamma, N, V_half)

    log_p_old = log_h_pdf_ivar(z, p, h_old, c, gamma, N, V_half)

    # compute acceptance ratio
    r = (log_p_star - log_p_old) + (log_q_old - log_q_star)

    # check for exp overflow
    try:
        R = math.exp(r)
    except:  # overflow
        R = 100

    accept_p = min(1, R)

    # draw random number for acceptance
    u = st.uniform.rvs(size=1)

    if u < accept_p:
        h_t = h_star
        # print("ACCEPT")
    else:
        h_t = h_old
        # print("REJECT")

    return h_t


"""performs Gibbs sampling over a number of iterations

    Args:
        z: gwas effect sizes
        h: heritability
        N: sample size
        M: number of SNPs
        V_half: LD-matrix to half power
        p_init: starting value for p
        c_init: starting values for c
        gamma_init: starting values for gamma
        its: number of iterations

    Returns:
        p_est: posterior mean of p
        gamma_est: posterior mean of gamma vectors
        c_est: posterior mean of c vector
        est-density: posterior density at the estimated parameters
        p_list: all sample of p over each iteration
        densities: all posterior densities over each iteration

"""
def gibbs(z, h, N, M, V_half, p_init=None, c_init=None, gamma_init=None, its=5000):

    # hold values
    p_list = []
    densities = []
    gamma_list = np.zeros(M)
    c_list = np.zeros(M)

    # variance terms
    sigma_e = (1-h)/N

    # initialize for t0 with values from prior if None
    if p_init is None:
        p_old = st.beta.rvs(.2, .2)
    else:
        p_old = p_init
    sd = math.sqrt(h / (M * p_old))
    if gamma_init is None:
        gamma_old = st.norm.rvs(loc=0, scale=sd, size=M)
    else:
        gamma_old = list(np.multiply(gamma_init, c_init))
    if c_init is None:
        c_old = st.bernoulli.rvs(p=p_old, size=M)
    else:
        c_old = list(c_init)

    sigma_g = h/float(M*p_old)

    for i in range(0, its):

        # draw p using MH step
        p_t = draw_p(z, p_old, h, c_old, gamma_old, N, V_half)
        p_density = joint_prob(p_t, c_old, gamma_old, h, z, N, V_half)

        # draw c and gamma
        c_t, gamma_t = draw_c_gamma(c_old, gamma_old, p_t, sigma_g, sigma_e, V_half, z)
        c_density = joint_prob(p_t, c_t, gamma_t, h, z, N, V_half)
        c_prop = np.sum(c_t)/float(M)

        # update variables
        if p_t == p_old:
            accepted = "reject"
        else:
            accepted = "accept"

        p_old = p_t
        c_old[:] = list(c_t)
        gamma_old[:] = list(gamma_t)
        sigma_g = h/float(M*p_old)

        # save values
        p_list.append(p_old)
        densities.append(c_density)

        # print debugging-info
        if i<= 10 or i%1==0:
            print("Iteration %d: sampled p: %.4f, %s, %.6g") % (i, p_old, accepted, p_density)
            print("Iteration %d: sampled c, prop c: %.4f, %.6g") % (i, c_prop, c_density)

            sys.stdout.flush()

        if i >= int(its*burn):
            c_list[:] = np.add(c_list, c_old)
            gamma_list[:] = np.add(gamma_list, gamma_old)

    start = int(its*burn)
    p_est = np.mean(p_list[start: ])

    # take average over c and gamma
    c_est = np.round(np.divide(c_list, its-int(its*burn)))
    gamma_est = np.divide(gamma_list, its-int(its*burn))

    # find estimated density
    est_density = joint_prob(p_est, c_est, gamma_est, h, z, N, V_half)

    return p_est, gamma_est, c_est, est_density, p_list, densities


"""performs Gibbs sampling over a number of iterations assuming infinitesimal variance (h/m)

    Args:
        z: gwas effect sizes
        h: heritability
        N: sample size
        M: number of SNPs
        V_half: LD-matrix to half power
        p_init: starting value for p
        c_init: starting values for c
        gamma_init: starting values for gamma
        its: number of iterations

    Returns:
        p_est: posterior mean of p
        gamma_est: posterior mean of gamma vectors
        c_est: posterior mean of c vector
        est-density: posterior density at the estimated parameters
        p_list: all sample of p over each iteration
        densities: all posterior densities over each iteration

"""
def gibbs_ivar(z, h, N, M, V_half, p_init=None, c_init=None, gamma_init=None, its=5000):

    # hold values
    p_list = []
    maps = []
    gamma_list = np.zeros(M)
    c_list = np.zeros(M)

    # variance terms
    sigma_e = (1-h)/N

    # initialize for t0 with values from prior if None
    if p_init is None:
        p_old = st.beta.rvs(.2, .2)
    else:
        p_old = p_init
    sd = math.sqrt(h / (M))
    if gamma_init is None:
        gamma_old = st.norm.rvs(loc=0, scale=sd, size=M)
    else:
        gamma_old = list(np.multiply(gamma_init, c_init))
    if c_init is None:
        c_old = st.bernoulli.rvs(p=p_old, size=M)
    else:
        c_old = list(c_init)

    sigma_g = h/float(M)

    for i in range(0, its):

        # draw p
        p_t = draw_p_ivar(c_old)

        p_density = joint_prob_ivar(p_t, c_old, gamma_old, h, z, N, V_half)

        # draw c and gamma
        c_t, gamma_t = draw_c_gamma(c_old, gamma_old, p_t, sigma_g, sigma_e, V_half, z)

        c_density = joint_prob_ivar(p_t, c_t, gamma_t, h, z, N, V_half)

        c_prop = np.sum(c_t)/float(M)

        # update variables
        if p_t == p_old:
            accepted = "reject"
        else:
            accepted = "accept"

        p_old = p_t
        c_old[:] = list(c_t)
        gamma_old[:] = list(gamma_t)
        sigma_g = h/float(M)

        # save values
        p_list.append(p_old)
        maps.append(c_density)

        # print debugging-info
        if i<= 10 or i%50==0:
            print("Iteration %d: sampled p: %.4f, %s, %.6g") % (i, p_old, accepted, p_density)
            print("Iteration %d: sampled c, prop c: %.4f, %.6g") % (i, c_prop, c_density)

            sys.stdout.flush()

        if i >= int(its*burn):
            c_list[:] = np.add(c_list, c_old)
            gamma_list[:] = np.add(gamma_list, gamma_old)

    start = int(its*burn)
    p_est = np.mean(p_list[start: ])
    p_var = np.var(p_list[start: ])

    # take average over c and gamma
    c_est = np.round(np.divide(c_list, its-int(its*burn)))
    gamma_est = np.divide(gamma_list, its-int(its*burn))

    # find estimated density
    est_density = joint_prob_ivar(p_est, c_est, gamma_est, h, z, N, V_half)

    return p_est, p_var, gamma_est, c_est, est_density, p_list, maps


def gibbs_ivar_gw(z_list, H_snp, H_gw, N, ld_half_flist, p_init=None, c_init_list=None, gamma_init_list=None, its=5000, non_inf_var=False):

  #  print("c init:")
  #  print(c_init_list)

    # hold samples of p
    p_list = []
    gamma_t_list = []
    c_t_list = []

    log_like_list = []

    # initialize params
    if p_init is None:
        p_t = st.beta.rvs(.2, .2)
    else:
        p_t= p_init

    B = len(z_list) # number of blocks

    # loop through all blocks
    for b in range(0, B):

        # read in betas from gwas file
        z_b = z_list[b]
        M_b = len(z_list[b])

        if non_inf_var:
            sd = math.sqrt(H_gw/float(p_t*M_b))
        else:
            sd = math.sqrt(H_snp)

        # save old value to see see if accepted/rejected
        p_old = p_t

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

    # end loop initializing first iteration

    for i in range(0, its):
        for b in range(0, B):

            # get params for block b
            z_b = z_list[b]
            M_b = len(z_list[b])

            # read in ld directly from file
            V_half_b = np.loadtxt(ld_half_flist[b])

            # get values from prev iteration
            gamma_t_b = gamma_t_list[b]
            c_t_b = c_t_list[b]

            if non_inf_var: # update sigma_g with new p value
                # DEBUGGING
                #if i == 0:
                #    logging.info("WARNING: USING DEBUGGING MODE OF FIXED P FOR VARIANCE")
                #p_t = 0.01
                sigma_g_b = H_gw/float(p_t*M_b)
            else:
                sigma_g_b = H_snp

            #logging.info("Sigma_g: %.4g" % sigma_g_b)

            sigma_e_b = (1 - H_gw) / N

            # sample causal vector and effects for  block b
            c_t_b, gamma_t_b = draw_c_gamma(c_t_b, gamma_t_b, p_t, sigma_g_b, sigma_e_b, V_half_b, z_b)

            # replace in larger lists
            gamma_t_list[b] = gamma_t_b
            c_t_list[b] = c_t_b

#            print("Sampled c:")
#            print(c_t_b)

 #           print("Sampled gamma:")
 #           print(gamma_t_b)

        # end loop over blocks
        p_t = draw_p_ivar_gw(c_t_list)

        sigma_e = (1 - H_gw) / N
        log_like_t = log_like(z_list, gamma_t_list, c_t_list, sigma_e, ld_half_flist)

        # keep running total of log likelihood
        log_like_list.append(log_like_t)

        # add p_t to list
        p_list.append(p_t)

        # print debugging-info
        if i <= 10 or i % 10 == 0:
            print("Iteration %d: sampled p: %.4f, log-like: %4g") % (i, p_t, log_like_t)

    # end loop iterations
    start = int(its*burn)
    p_est = np.mean(p_list[start: ])
    p_var = np.var(p_list[start: ])

    avg_log_like = np.mean(log_like_list[start:])
    var_log_like = np.var(log_like_list[start:])

    return p_est, p_var, p_list, avg_log_like, var_log_like




def gibbs_ivar_full(z, N, M, V_half, p_init=None, h_init=None, c_init=None, gamma_init=None, its=5000):
    # hold values
    p_list = []
    density_list = []
    gamma_list = np.zeros(M)
    c_list = np.zeros(M)
    h_list = []

    # initialize for t0 with values from prior if None
    if p_init is None:
        p_old = st.beta.rvs(beta_lam, beta_lam)
    else:
        p_old = p_init
    if h_init is None:
        h_old = st.beta.rvs(a=sigma_prior_a, b=sigma_prior_b)
    else:
        h_old = h_init
    if gamma_init is None:
        sd = math.sqrt(h_old / (M))
        gamma_old = st.norm.rvs(loc=0, scale=sd, size=M)
    else:
        gamma_old = list(np.multiply(gamma_init, c_init))
    if c_init is None:
        c_old = st.bernoulli.rvs(p=p_old, size=M)
    else:
        c_old = list(c_init)

    for i in range(0, its):

        # draw p
        p_t = draw_p_ivar(c_old)

        p_density = joint_prob_ivar(p_t, c_old, gamma_old, h_old, z, N, V_half)

        # draw h
        h_t = draw_h_ivar(z, p_t, h_old, c_old, gamma_old, N, V_half)
        h_density = joint_prob_ivar(p_t, c_old, gamma_old, h_t, z , N, V_half)

        # draw c and gamma

        # variance terms
        sigma_g = h_t / float(M)
        sigma_e = (1 - h_t) / N
        c_t, gamma_t = draw_c_gamma(c_old, gamma_old, p_t, sigma_g, sigma_e, V_half, z)

        c_density = joint_prob_ivar(p_t, c_t, gamma_t, h_t, z, N, V_half)

        c_prop = np.sum(c_t) / float(M)

        # update variables
        if h_t == h_old:
            accepted = "reject"
        else:
            accepted = "accept"

        p_old = p_t
        h_old = h_t
        c_old[:] = list(c_t)
        gamma_old[:] = list(gamma_t)

        # save values
        p_list.append(p_old)
        h_list.append(h_old)
        density_list.append(c_density)

        # print debugging-info
        if i <= 10 or i % 50 == 0:
            print("Iteration %d: sampled p: %.4f, %.6g") % (i, p_old, p_density)
            print("Iteration %d: sampled h: %.4f, %s, %.6g") % (i, h_old, accepted, h_density)
            print("Iteration %d: sampled c, prop c: %.4f, %.6g") % (i, c_prop, c_density)

            sys.stdout.flush()

        if i >= int(its * burn):
            c_list[:] = np.add(c_list, c_old)
            gamma_list[:] = np.add(gamma_list, gamma_old)

    start = int(its * burn)
    p_est = np.mean(p_list[start:])
    p_var = np.var(p_list[start:])
    h_est = np.mean(h_list[start:])
    h_var = np.var(h_list[start:])

    # take average over c and gamma
    c_est = np.round(np.divide(c_list, its - int(its * burn)))
    gamma_est = np.divide(gamma_list, its - int(its * burn))

    # find estimated density
    est_density = joint_prob_ivar(p_est, c_est, gamma_est, h_est, z, N, V_half)

    return p_est, p_var, h_est, h_var, gamma_est, c_est, est_density, p_list, h_list, density_list



def draw_p_ivar_gw(c_t_list):

    c_t_list = np.asarray(c_t_list)
    c_t_list = np.hstack(c_t_list)
    M = np.asarray(c_t_list).size

    alpha1 = beta_lam + np.sum(c_t_list)
    alpha2 = beta_lam + (M - np.sum(c_t_list))
    try:
        p = st.beta.rvs(alpha1, alpha2)
    except:
        print c_t_list

    return p


def log_like(z_list, gamma_t_list, c_t_list, sigma_e, V_half_flist):
    total_log_like = 0

    for z, gamma_t, c_t, V_half_fname in zip(z_list, gamma_t_list, c_t_list, V_half_flist):
        V_half = np.loadtxt(V_half_fname)
        M = len(z)
        mu = np.matmul(V_half, np.multiply(gamma_t, c_t))
        cov = np.eye(M)*sigma_e
        log_like = st.multivariate_normal.logpdf(z, mu, cov)
        total_log_like += log_like

    return total_log_like
