from unity_metropolis import *
from auxilary import *
import sys
import scipy.sparse as sp
import pandas as pd
import logging
import cProfile, pstats, StringIO


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


    return c_t, gamma_t

