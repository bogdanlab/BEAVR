from auxilary import *
from unity_gibbs import *
import logging

# global logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def calc_mu_dp(mu_old, delta_c_b, delta_gamma_b, V_half, nu_m, m):

	#print "Delta c"
	#print delta_c_b

	# find all causals
	causal_inds_list = np.where(delta_c_b != 0)
	#print delta_c_b
	causal_inds = causal_inds_list[0]
	sum = 0
	#for i in causal_inds:
	for i in range(0, len(delta_c_b)):
		V_half_m = V_half[:, m]
		V_half_i = V_half[:, i ]

		a_mi = np.dot(V_half_m, V_half_i)
		residual = a_mi * delta_c_b[i] * delta_gamma_b[i]
		delta_c_b_m = delta_c_b[i]
		#print "Delta c_m: %d" % delta_c_b_m
		#print "Delta gamma_m: %.4g" % delta_gamma_b[i]
		#print "Residual: %.4g" % residual
		#if i != m:
		sum += residual*nu_m
		print("residual: %.4g" % residual)
		print("nu m: %.4g" % nu_m)

	#print sum
	mu_m = mu_old - sum

	#print("mu_m: %.4g" % mu_m)

	return mu_m


def calc_mu_opt(gamma_old, c_old, z, a_matrix, psi, V_half, sigma_g, sigma_e, m):

	"""
	M = len(z)
	sum = 0

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
	bottom_sigma_m = 1/float(sigma_g) + (1/float(sigma_e))*a_matrix[m,m]
	sigma_m = 1/float(bottom_sigma_m)
	W_m = V_m_half
	#psi_m = np.dot(z, W_m)
	psi_m = psi[m]
	sum = 0
	M = len(z)

	# find nonzero indicies
	nonzero_inds = np.nonzero(c_old)[0]
	for i in nonzero_inds:
		#W_i = V_half[:,i]
		#a_im = np.dot(W_i, W_m)
		a_im = a_matrix[i,m]
		if i != m:
			sum += a_im * gamma_old[i] * c_old[i]
	"""
	for i in range(0, M):
		W_i = V_half[:,i]
		a_im = np.dot(W_i, W_m)
		if i != m:
			sum += a_im * gamma_old[i] * c_old[i]
	"""
	temp_term = psi_m - sum
	mu_m = (sigma_m/float(sigma_e))*temp_term
	# end optimization

	return mu_m


def draw_c_gamma_dp(c_old, gamma_old, p_old, sigma_g, sigma_e, a_matrix, psi, V_half, z, mu_b, delta_c_b, delta_gamma_b):

	z = z.reshape(len(z))

	M = len(c_old) # number of SNPs

	# hold new values for c-vector and gamma-vector
	c_t = np.zeros(M)

	gamma_t = np.zeros(M)

	# loop through all SNPs
	for m in range(0, M):

		# calculate variance term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
		#V_m_half = V_half[:, m]

		mu_old = mu_b[m]
		c_old_m = c_old[m]
		gamma_old_m = gamma_old[m]

		# calculate params for posterior of c, where P(c|.) ~ Bern(d_m)
		#bottom_sigma_m = 1/float(sigma_g) + (1/float(sigma_e))*(np.matmul(np.transpose(V_m_half), V_m_half))
		bottom_sigma_m = 1/float(sigma_g) + (1/float(sigma_e))*a_matrix[m,m]
		sigma_m = 1/float(bottom_sigma_m)
		var_term = math.sqrt(sigma_m/float(sigma_g))

		#mu_m =  calc_mu_dp(mu_old, delta_c_b, delta_gamma_b, V_half, nu_m, m)
		mu_m = calc_mu_opt(gamma_old, c_old, z, a_matrix, psi, V_half, sigma_g, sigma_e, m)

		#print("mu old: %.4g; mu new: %.4g" % (mu_old, mu_m))
		a = 0.50 * 1 / (float(sigma_m)) * mu_m * mu_m

		# check for overflow
		if a > EXP_MAX:
			a = EXP_MAX

		# Bernoulli parameter, where P(c|.) ~ Bern(d_m)
		d_m = (p_old*var_term*math.exp(a))/float(p_old*var_term*math.exp(a) + (1-p_old))

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

		# update dp params
		mu_b[m] = mu_m
		delta_c_b[m] = c_m - c_old_m
		delta_gamma_b[m] = gamma_m - gamma_old_m

	#print mu_b
	return c_t, gamma_t, mu_b, delta_c_b, delta_gamma_b


def gibbs_ivar_gw_dp(z_list, H_snp, H_gw, N, ld_half_flist, p_init=None, c_init_list=None, gamma_init_list=None, its=5000, non_inf_var=False):

	# hold samples of p
	p_list = []
	gamma_t_list = []
	c_t_list = []
	mu_list = []
	delta_c_list = []
	delta_gamma_list = []
	a_matrix_list = []
	psi_list = []

	log_like_list = []

    # initialize params
	if p_init is None:
		p_t = st.beta.rvs(.2, .2)
	else:
		p_t= p_init

	B = len(z_list) # number of blocks
	logging.info("Found %d blocks" % B)

	logging.info("Initializing first iteration")

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

		delta_gamma_list.append(gamma_t_b)
		delta_c_list.append(c_t_b)

		# long computation for mu

		mu_b = np.zeros(M_b)

		V_half = np.loadtxt(ld_half_flist[b])

		sigma_e = (1-H_gw)/float(N)

		gamma_old = gamma_t_list[b]
		c_old = c_t_list[b]

		gamma_t = np.zeros(M_b)
		c_t = np.zeros(M_b)

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

		if non_inf_var: # update sigma_g with new p value
			sigma_g = H_gw/float(p_t*M_b)
		else:
			sigma_g = H_snp

		logging.info("Long calculation for initialization")
        for m in range(0, M_b):

			# save old values for delta calculation
			gamma_m_old = gamma_old[m]
			c_m_old = c_old[m]

        	# calculate variance term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
			V_m_half = V_half[:, m]

			bottom_sigma_m = 1/float(sigma_g) + (1/float(sigma_e))*(np.matmul(np.transpose(V_m_half), V_m_half))
			sigma_m = 1/float(bottom_sigma_m)

			beta = np.multiply(gamma_old, c_old)

			if m > 0:
				beta_m = beta[m-1]
			else:
				beta_m = 0

			middle_term = np.matmul(V_half, beta)

			end_term = np.multiply(V_m_half, gamma_old[m])
			r_m = z_b - middle_term + end_term

			# calculate mean term of posterior of gamma, where P(gamma|.) ~ N(mu_m, sigma_m)
			temp_term = np.matmul(np.transpose(r_m), V_m_half)

			mu_m = (sigma_m/float(sigma_e))*temp_term

			mu_b[m] = mu_m

			# update c and gamma as well

			# calculate params for posterior of c, where P(c|.) ~ Bern(d_m)
			var_term = math.sqrt(sigma_m/float(sigma_g))

			a = 0.50 * 1 / (float(sigma_m)) * mu_m * mu_m

			# check for overflow
			if a > EXP_MAX:
			    a = EXP_MAX

			# Bernoulli parameter, where P(c|.) ~ Bern(d_m)
			d_m = (p_old*var_term*math.exp(a))/float(p_old*var_term*math.exp(a) + (1-p_old))

			# draw c_m
			c_m = st.bernoulli.rvs(d_m)

			if c_m == -1:
				print("ERROR, -1")
				exit(1)

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

			gamma_t_list[b][m] = gamma_m
			c_t_list[b][m] = c_m

			# change delta values
			#delta_gamma_list[b][m] = gamma_m - gamma_m_old
			#delta_c_list[b][m] = c_m - c_m_old

        mu_list.append(mu_b)

	p_t = draw_p_ivar_gw(c_t_list)

	p_list.append(p_t)

	log_like_t = log_like(z_list, gamma_t_list, c_t_list, sigma_e, ld_half_flist)
	# keep running total of log likelihood
	log_like_list.append(log_like_t)

	# print debugging-info
	if i <= 10 or i % 10 == 0:
		print("Iteration %d: sampled p: %.4f, log-like: %4g") % (i, p_t, log_like_t)

	# end loop through blocks

    # end loop initializing first iteration

	#logging.info("Mu initialization")
	#logging.info(mu_list)

	for i in range(1, its): # start at iteration 1 b/c already drew p above
		for b in range(0, B):

			z_b = z_list[b]
			M_b = len(z_list[b])

			# read in ld directly from file
			V_half_b = np.loadtxt(ld_half_flist[b])

			# get values from prev iteration
			gamma_t_b = gamma_t_list[b]
			c_t_b = c_t_list[b]

			sigma_e_b = (1 - H_gw) / N

			# sample causal vector and effects for  block b
			delta_mu_b = mu_list[b]
			delta_c_b = delta_c_list[b]
			#logging.info("Delta c vec")
			#print delta_c_b
			delta_gamma_b = delta_gamma_list[b]

			#logging.info("Sampling c and gamma")
			a_matrix = a_matrix_list[b]
			psi = psi_list[b]
			c_t_b, gamma_t_b, mu_b, delta_c_b, delta_gamma_b = draw_c_gamma_dp(c_t_b, gamma_t_b, p_t, sigma_g, sigma_e_b, a_matrix, psi, V_half, z_b, delta_mu_b, delta_c_b, delta_gamma_b)

			# update running deltas
			mu_list[b] = mu_b
			#print("mu list")
			#print(mu_list)

			delta_c_list[b] = delta_c_b
			delta_gamma_list[b] = delta_gamma_b

			# replace in larger lists
			gamma_t_list[b] = gamma_t_b
			c_t_list[b] = c_t_b

			#print("Sampled c:")
			#print(c_t_b)

			#print("Sampled gamma:")
			#print(gamma_t_b)

		# end loop over blocks
		p_t = draw_p_ivar_gw(c_t_list)

		if non_inf_var: # update sigma_g with new p value
			sigma_g = H_gw/float(p_t*M_b)
		else:
			sigma_g = H_snp

        # add p_t to list
		p_list.append(p_t)
		if i <= 10 or i % 10 == 0:
			print("Iteration %d: sampled p: %.4f, sigma_g: %.4g, log-like: %4g") % (i, p_t, sigma_g, log_like_t)

    # end loop iterations
	start = int(its*burn)
	p_est = np.mean(p_list[start: ])
	p_var = np.var(p_list[start: ])

	avg_log_like = np.mean(log_like_list[start:])
	var_log_like = np.var(log_like_list[start:])

	return p_est, p_var, p_list, avg_log_like, var_log_like


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
