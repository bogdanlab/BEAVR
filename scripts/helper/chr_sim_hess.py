# single chromosome local h2g simulation study
import dask 
import dask.array as da
import numpy as np
import fire
from scipy import linalg
from os.path import join
import pandas as pd

class Sumstats(object):
    def __init__(self, sumstats_path):
        self.sumstats = pd.read_csv(sumstats_path, sep='\t')
        # print(self.sumstats)

    def get_locus(self, par):
        chr_i = par.get('chr')
        start = par.get('start')
        stop = par.get('stop')
        sumstats_locus = self.sumstats[(self.sumstats.CHR == chr_i) & \
                     (start <= self.sumstats.BP) & \
                     (self.sumstats.BP < stop)].reset_index(drop=True)
        return sumstats_locus
    def get_avg_N(self):
        return np.mean(self.sumstats.N)
class LD(object):
    def __init__(self, ld_dir, chr_i, legend):
        # read legend
        self.legend = pd.read_table(legend, header=None)
        self.legend.columns =  ['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2']

        # read ld
        chr_ld_dir = join(ld_dir, str(chr_i))
        part_info = pd.read_table(join(chr_ld_dir, 'part.info'), header=None, sep='\t', names=['row', 'col'])

        # get last_index to determine shape
        last_ld = np.load(join(chr_ld_dir, 'part_{}.npy'.format(len(part_info))))
        info_end = int(part_info['row'][len(part_info) - 1].split('-')[1])
        index_end = int(part_info['row'][len(part_info) - 1].split('-')[0]) + last_ld.shape[0]
        ld_len = int(np.sqrt(len(part_info)))
        ld = np.zeros([ld_len, ld_len]).tolist()
        
        for part_i, part in part_info.iterrows():
            row_start, row_end = [int(i) for i in part_info['row'][part_i].split('-')]
            col_start, col_end = [int(i) for i in part_info['col'][part_i].split('-')]
            if row_end == info_end:
                row_end = index_end
            if col_end == info_end:
                col_end = index_end
            local_ld = dask.delayed(np.load)(join(chr_ld_dir, 'part_{}.npy'.format(part_i + 1)))
            local_ld = da.from_delayed(local_ld, shape=(row_end - row_start, col_end - col_start), dtype=np.float64)
            ld[int(part_i / ld_len)][part_i % ld_len] = local_ld

        self.ld = da.block(ld)

    def get_locus(self, par):
        chr_i = par.get('chr')
        start = par.get('start')
        stop = par.get('stop')
        legend_chr = self.legend[self.legend.CHR == chr_i].reset_index(drop=True)
        locus_index = np.where((start <= legend_chr.BP) & (legend_chr.BP < stop))[0]
        locus_min = min(locus_index)
        locus_max = max(locus_index) + 1
        return self.ld[locus_min : locus_max, locus_min : locus_max].compute()

# for simulation study
def eigen_decomp(ld_dir, chr_i, legend, partition, out):
    ld = LD(ld_dir, chr_i, legend)
    partition = pd.read_table(partition)
    partition.columns = ['chr', 'start', 'stop']
    partition.chr = partition.chr.apply(lambda x : int(x.strip()[3:]))
    partition = partition[partition.chr == chr_i].reset_index(drop=True)
    eigen_decomp_list = []
    print(partition)
    for par_i, par in partition.iterrows():
        ld_locus = ld.get_locus(par)
        ld_w, ld_v = linalg.eigh(ld_locus)
        eigen_decomp_list.append({'ld_w': ld_w, 'ld_v': ld_v})
    partition['eigen_decomp'] = eigen_decomp_list
    partition.to_pickle(out)

def get_quad_form(sumstats, ld_v, ld_w, max_k):
    # compute \beta^T V^+ \beta
    # where estimation V^+ is controled by max_k
    beta_gwas = sumstats.Z.values / np.sqrt(sumstats.N.values)    
    eig_start_index = (len(ld_w) - max_k) if (len(ld_w) > max_k) else 0

    all_prj = []
    for i in range(eig_start_index, len(ld_w)):
        prj = np.dot(beta_gwas.T, ld_v[:, i])
        all_prj.append(prj ** 2 / ld_w[i])
    quad_form = np.sum(all_prj)
    eig_num = len(ld_w) - eig_start_index
    return quad_form, eig_num

def estimate_batch(eigen, sumstats_prefix, num_sim, out_prefix):
    # use two methods to estimate the local h2g. Because we are doing simulation study, num_indv is consistent
    partition = pd.read_pickle(eigen)
    num_loci = len(partition)
    
    local_h2g_list_loci = []
    # first do loci style estimation
    for sim_i in range(1, num_sim + 1):
        sumstats = Sumstats(sumstats_prefix + str(sim_i) + '.sumstats')
        indv_num = sumstats.get_avg_N()
#         indv_num = np.mean(sumstats.N.values)
        quad_form_list = []
        eig_num_list = []
        for par_i, par in partition.iterrows():
            sumstats_locus = sumstats.get_locus(par)
            # preserve all eigen vectors
            quad_form, eig_num = get_quad_form(sumstats_locus, \
                                par['eigen_decomp']['ld_v'], \
                                par['eigen_decomp']['ld_w'], 9999)
            quad_form_list.append(quad_form)
            eig_num_list.append(eig_num)
        quad_form_list = np.array(quad_form_list)
        eig_num_list = np.array(eig_num_list)
        local_h2g = (indv_num * quad_form_list - eig_num_list) / (indv_num - eig_num_list)
        local_h2g_list_loci.append(local_h2g)
    
    # then do recommended style of estimation
    # set the number of eigen values preserved
    max_k = 50
    local_h2g_list_hess = []
    for sim_i in range(1, num_sim + 1):
        sumstats = Sumstats(sumstats_prefix + str(sim_i) + '.sumstats')
        indv_num = sumstats.get_avg_N()
#         indv_num = np.mean(sumstats.N.values)
        quad_form_list = []
        eig_num_list = []
        for par_i, par in partition.iterrows():
            sumstats_locus = sumstats.get_locus(par)
            # preserve all eigen vectors
            quad_form, eig_num = get_quad_form(sumstats_locus, par['eigen_decomp']['ld_v'], par['eigen_decomp']['ld_w'], max_k)
            quad_form_list.append(quad_form)
            eig_num_list.append(eig_num)
        quad_form_list = np.array(quad_form_list)
        eig_num_list = np.array(eig_num_list)
        # solving the linear equation
        A = np.diag([indv_num for i in range(num_loci)])
        for i in range(num_loci):
            A[i, :] -= eig_num_list[i]
        b = indv_num * quad_form_list - eig_num_list
        rank = np.linalg.matrix_rank(A)
        if rank < num_loci:
            print('rank deficient')
        local_h2g = linalg.solve(A, b)
        local_h2g_list_hess.append(local_h2g)
    # print(local_h2g_list_loci)
    # print(local_h2g_list_hess)
    # print('local_h2g_list_loci')
    # for local_h2g in local_h2g_list_loci:
    #     print(np.sum(local_h2g))
    # print('local_h2g_list_hess')
    # for local_h2g in local_h2g_list_hess:
    #     print(np.sum(local_h2g))
    # print(np.array(local_h2g_list_loci).shape)
    # then save
    np.save(out_prefix + 'loci.npy', local_h2g_list_loci)
    np.save(out_prefix + 'hess.npy', local_h2g_list_hess)

if __name__ == '__main__':
    fire.Fire()
