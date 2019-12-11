from auxilary import *


"""
unity_metropolis.py

Describes functions used for sampling from posterior distribution of p using
random-walk Metropolis-Hastings. Also included auxilary functions for general
MH algorithm. Does NOT model LD, assumes c-vector and gamma-vector analytically
integrated out.

"""


"""log likelihood for no-LD case with
   c-vector and gamma-vector analytically integrated out

    Args:
        p: proportion p
        h: heritability
        z: gwas effect sizes
        N: number of individuals
        M: number of SNPs

    Returns:
        log_pdf: log joint posterior density

"""
def log_likelihood_MH(p, h, z, N, M):
    sigma_e = (1 - h) / N
    sigma_g = h / float(M * p)

    # work in vector form
    log_pdf_0= st.norm.logpdf(z, loc=0, scale=math.sqrt(sigma_e))
    log_pdf_1=st.norm.logpdf(z, loc=0, scale=math.sqrt(sigma_e + sigma_g))

    log_d_0 = np.add(log_pdf_0, math.log((1-p)))
    log_d_1 = np.add(log_pdf_1, math.log(p))

    # pdf for each SNP
    snp_pdfs = logsumexp_vector([log_d_0, log_d_1])

    # add together log pdfs
    log_pdf = np.sum(snp_pdfs)

    return log_pdf


""" q-distribution (proposal dist.) for p-vec
    when doing Metropolis hastings step

    Args:
        p_old: estimate of p from previous iteration

    Returns:
        p_star: proposed value for p

"""
def q_p_rvs(p_old):
    B = 10 # constant that controls how much variance in proposal distribution
    p_alpha_star_1 = beta_lam + p_old*B
    p_alpha_star_2 = beta_lam + (1-p_old)*B

    p_star = st.beta.rvs(p_alpha_star_1, p_alpha_star_2)

    return p_star


""" log density of q-distribution (proposal dist.) for p-vec
    when doing Metropolis hastings step

    Args:
        p: proposal for p

    Returns:
        p_old: estimate of p from previous iteration

"""
def log_q_pdf(p, p_old):
    B = 10 # constant that controls how much variance in proposal distribution
    p_alpha_1 = beta_lam + p_old*B
    p_alpha_2 = beta_lam + (1-p_old)*B

    log_d_p = st.beta.logpdf(x=p, a=p_alpha_1, b=p_alpha_2)

    return log_d_p


""" log density of posterior of p w/out LD where
    c-vector and gamma-vector analytically integrated out

    Args:
        p: sample p
        h: heritability
        z: gwas effect sizes
        N: number of individuals
        M: number of SNPs

    Returns:
        log_pdf: log posterior density

"""
def log_p_pdf(p, h, z, N, M):
    # calculate likelihood
    log_like = log_likelihood_MH(p, h, z, N, M)

    # calculate prior
    log_prior = st.beta.logpdf(x=p, a=beta_lam, b=beta_lam)

    # add likelihood and prior
    log_pdf = log_like + log_prior

    return log_pdf



""" calculates acceptance probability in Metropolis-Hastings step

    Args:
        p_old: old value of p
        p_star: proposal value for p
        h: heritability
        z: gwas effect sizes
        N: number of individuals
        M: number of SNPs

    Returns:
        accept_prob: acceptance probability; value in [0,1]

"""
def accept(p_old, p_star, h, z, N, M):

    log_q_star = log_q_pdf(p_star, p_old)

    log_q_old = log_q_pdf(p_old, p_star)

    log_p_star = log_p_pdf(p_star, h, z, N, M)

    log_p_old = log_p_pdf(p_old, h, z, N, M)

    # compute acceptance ratio
    r = (log_p_star - log_p_old) + (log_q_old - log_q_star)

    # check for exp overflow
    try:
        R = math.exp(r)
    except: # overflow
        R = 100

    accept_prob = min(1, R)

    return accept_prob


""" sample p using Metropolis-Hastings step w/out LD
    (c-vector and gamma-vector analytically integrated out)

    Args:
        p_old: old value of p
        h: heritability
        z: gwas effect sizes
        N: number of individuals
        M: number of SNPs

    Returns:
        p_t: sample for p

"""
def draw_p_MH(p_old, h, z, N, M):

    # draw from proposal distribution
    p_star = q_p_rvs(p_old)

    # accept or reject proposal
    accept_p = accept(p_old, p_star, h, z, N, M)

    # draw random number for acceptance
    u = st.uniform.rvs(size=1)

    if u < accept_p:
        p_t = p_star
    else:
        p_t = p_old

    # return accepted params
    return p_t


""" run Metropolis-Hastings algorithm to sample p

    Args:
        z: gwas effect sizes
        h: heritability
        N: number of individuals
        M: number of SNPs
        its: number of iterations
    Returns:
        p_est: posterior mean of p
        est_log_density: posterior density

"""
def metropolis(z, h, N, M, its=5000):
    # hold values
    p_list = []

    # initialize for t0 with values from prior
    p_old = st.beta.rvs(beta_lam, beta_lam)

    for i in range(0, its):

        # draw p
        p_t = draw_p_MH(p_old, h, z, N, M)

        # update variables
        p_old = p_t

        p_list.append(p_old)

    # only keep points after burn-in
    start = int(its*burn)
    p_est = np.mean(p_list[start: ])


    est_log_density = log_p_pdf(p_est,h,z,N,M)

    return p_est, est_log_density


def q_h_rvs(h_old):

    h_star = sigmoid(st.norm.rvs(loc=logit(h_old), scale=1.0))

    return h_star


def log_q_h_pdf(h_star, h_old):

    log_pdf = st.norm.logpdf(x=logit(h_star), loc=logit(h_old), scale=1.0)

    return log_pdf



