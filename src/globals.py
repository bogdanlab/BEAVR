import sys
import math

# Global variables for specifying hyper-parameters for priors

# Hyper-parameter for p, p~Beta(beta_lam_1, beta_lam_2)
beta_lam_1 = .2 # 1, 2, .2
beta_lam_2 = .2 # 1, 2, .2

# Hyper-parameter for sigma_g^2, sigma_g^2~Inv-G(alpha_g0, beta_g0)
alpha_g0=10
beta_g0=0.10

# Hyper-parameter for sigma_e^2, sigma_e^2~Inv_G(alpha_e0, beta_e0)
alpha_e0=10
beta_e0=0.001

# Global constants to check for under/overflow
EXP_MAX = math.log(sys.float_info.max)

# Burn-in for MCMC, (0,1)
BURN = 0.25

# constant contolling proposal distribution for Metropolis step
metropolis_factor = 10
