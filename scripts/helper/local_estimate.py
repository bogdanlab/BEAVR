import dask
import dask.array as da
import numpy as np
import fire
from scipy import linalg
from os.path import join
import pandas as pd

class Sumstats(object):
    def __init__(self, sumstats_path, legend):
        self.legend = pd.read_table(legend, header=None)
        self.legend.columns =  ['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2']
        self.sumstats = pd.read_csv(sumstats_path, names=self.legend.columns, sep='\t')
        #self.sumstats = pd.read_table(sumstats_path)

    def get_locus(self, par):
        chr_i = par.get('chr')
        start = par.get('start')
        stop = par.get('stop')
        sumstats_locus = self.sumstats[(self.sumstats.CHR == chr_i) & \
                     (start <= self.sumstats.BP) & \
                     (self.sumstats.BP < stop)].reset_index(drop=True)
        return sumstats_locus

class LD(object):
    def __init__(self, ld_dir, legend):
        # read legend
        self.legend = pd.read_table(legend, header=None)
        self.legend.columns =  ['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2']

        # read ld
        self.ld_list = []
        for chr_i in range(1,23):
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

            ld = da.block(ld)
            self.ld_list.append(ld)


    def get_locus(self, par):
        chr_i = par.get('chr')
        start = par.get('start')
        stop = par.get('stop')
        legend_chr = self.legend[self.legend.CHR == chr_i].reset_index(drop=True)
        locus_index = np.where((start <= legend_chr.BP) & (legend_chr.BP < stop))[0]
        locus_min = min(locus_index)
        locus_max = max(locus_index) + 1
        return self.ld_list[chr_i - 1][locus_min : locus_max, locus_min : locus_max].compute()



def step1(sumstats, ld_dir, legend, partition, out):
    sumstats = Sumstats(sumstats, legend)
    ld = LD(ld_dir, legend)
    partition = pd.read_table(partition)
    partition.columns = ['chr', 'start', 'stop']
    partition.chr = partition.chr.apply(lambda x : int(x.strip()[3:]))
    eigen_prj_list = []
    snp_num_list = []
    indv_num_list = []
    snp_list = []
    print(partition)
    for par_i, par in partition.iterrows():
        sumstats_locus = sumstats.get_locus(par)
        snp_list.append(sumstats_locus.SNP.values)
        ld_locus = ld.get_locus(par)
        ld_w, all_prj = squared_projection(sumstats_locus, ld_locus)
        eigen_prj_list.append({'prj': all_prj, 'weight': ld_w})
        snp_num = len(sumstats_locus)
        indv_num = sumstats_locus.N.mean()
        quad_form = np.sum(all_prj / ld_w)
        snp_num_list.append(snp_num)
        indv_num_list.append(indv_num)
        print('{}/{}: chr: {}, snp_num: {}, indv_num: {:.2f}, est: {:.6f}'.\
                 format(par_i, len(partition), par.chr, snp_num, indv_num, quad_form))
    partition['snp_num'] = snp_num_list
    partition['indv_num'] = indv_num_list
    partition['eigen_prj'] = eigen_prj_list
    partition.to_pickle(out + '.step1')

def step2(step1, max_k, solve_method, out):
    df = pd.read_pickle(step1)
    num_loci = len(df)
    num_snp = np.sum(df['snp_num'])
    num_indv = np.sum(df['indv_num'] * df['snp_num']) / num_snp
    prj_list = [row['prj'] for row in df['eigen_prj']]
    weight_list = [row['weight'] for row in df['eigen_prj']]
    eig_num_list = []
    reg_quad_form_list = []
    for i in range(num_loci):
        # weight_list (eigen values) are sorted from small to last, only preserve the last max_k one
        eig_start_index = (len(weight_list[i]) - max_k) if (len(weight_list[i]) > max_k) else 0
        eig_num_list.append(len(weight_list[i]) - eig_start_index)
        reg_quad_form = np.sum(np.array(prj_list[i][eig_start_index : ]) / np.array(weight_list[i][eig_start_index : ]))
        reg_quad_form_list.append(reg_quad_form)
    df['quad_form'] = reg_quad_form_list
    df['eig_num'] = eig_num_list

    if solve_method == 'loci':
        # calculating the local h2g without accounting for bias
        local_h2g = (df['indv_num'] * df['quad_form'] - df['eig_num']) / (df['indv_num'] - df['eig_num'])
        df['local_h2g'] = local_h2g
        print(sum(local_h2g))
    elif solve_method == 'gw':
        # calculating the local h2g with accounting for bias (solve the linear system)
        A = np.diag(df['indv_num'])
        for i in range(num_loci):
            A[i, :] -= df['eig_num']
        b = df['indv_num'] * df['quad_form'] - df['eig_num']
        rank = np.linalg.matrix_rank(A)
        if rank < num_loci:
            print('rank deficient')
        local_h2g = linalg.solve(A, b)
        df['local_h2g'] = local_h2g
        print(sum(local_h2g))
    else:
        assert(0)

    # output
    est = df[['chr', 'start', 'stop', 'snp_num', 'indv_num', 'quad_form', 'eig_num', 'local_h2g']]
    est.to_csv(out + '.csv')


def solve_by_loci(df):
    est = pd.DataFrame(df)
    local_h2g = (est['indv_num'] * est['quad_form'] - est['rank']) / (est['indv_num'] - est['rank'])
    return local_h2g.values

def solve_by_gw(est):

    num_loci = len(est)
    num_snp = np.sum(est['snp_num'])
    num_indv = np.sum(est['indv_num'] * est['snp_num']) / num_snp

    A = np.diag(est['indv_num'])
    for i in range(num_loci):
        A[i, :] -= est['rank']
    b = est['indv_num'] * est['quad_form'] - est['rank']
    rank = np.linalg.matrix_rank(A)
    if rank != num_loci:
        print('error: rank deficient')
    local_h2g = linalg.solve(A, b)
    return local_h2g

# estimate squared projection in the first step of HESS
def squared_projection(sumstats, ld):
    ld_w, ld_v = linalg.eigh(ld)
    beta_gwas = sumstats.Z.values / np.sqrt(sumstats.N.values)
    all_prj = []
    for i in range(len(ld_w)):
        prj = np.dot(beta_gwas.T, ld_v[:, i])
        all_prj.append(prj ** 2)
    return ld_w, all_prj


if __name__ == '__main__':
    fire.Fire()
