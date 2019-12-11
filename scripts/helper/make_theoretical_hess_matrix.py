# coding: utf8

import numpy as np
import os
import pandas as pd

# makes nice csv file

# set these params
M_gw = 593300
h2=0.50
loci_dir="/u/home/r/ruthjohn/ruthjohn/unity_v3.0/paper_sims/cau_ratio_0.01_hsq_0.5_maf_0.0_ld_0.0_range_0.0_1.0/6mb"
block='6mb'
prefix1 = 'cau_ratio_0.01_hsq_0.5_maf_0.0_ld_0.0_range_0.0_1.0'
prefix2= 'cau_ratio_0.01_hsq_0.5_maf_0.75_ld_1.0_range_0.0_1.0'
hess_style = 'theoretical'

partition_dir="/u/home/r/ruthjohn/ruthjohn/unity_v3.0/sims_kk_gw/meta_data/partitions"
outdir = "/u/home/r/ruthjohn/ruthjohn/unity_v3.0/misc"

results_df = pd.DataFrame({}, columns=['chr', 'start', 'stop','local_h2g','sim'])
block_bed_file = os.path.join(partition_dir, "{}.bed".format(block))
bed_df = pd.read_csv(block_bed_file, sep='\t')
bed_df.columns = ['chr', 'start', 'stop']

# only look at chr22
if block == 'berisa':
    # weird added space
    bed_df = bed_df.loc[bed_df['chr'] == 'chr22 '].reset_index()
else:
    bed_df = bed_df.loc[bed_df['chr'] == 'chr22'].reset_index()

for index, row in bed_df.iterrows():

    chr_str = row['chr']
    chr = int(chr_str.split("chr")[1])
    start = row['start']
    stop = row['stop']
    temp_df = pd.DataFrame({}, columns=['chr', 'start', 'stop','local_h2g','sim'])

    # get M_b
    sim_i = 1
    loci_file = "chr_{}_start_{}_stop_{}_sim_{}.loci".format(chr, start, stop, sim_i)
    loci_df = pd.read_csv(os.path.join(loci_dir, loci_file), sep=' ')
    M_b = len(loci_df.index)
    h2_b = (M_b/float(M_gw))*h2

    temp_df['chr'] = [chr]*100
    temp_df['start'] = [start]*100
    temp_df['stop'] = [stop]*100
    temp_df['local_h2g'] = [h2_b]*100
    temp_df['sim'] = np.arange(1,100+1)

    # add to overall df
    results_df = pd.concat([results_df, temp_df])

# save resulting to a csv file
out_csv = os.path.join(outdir, "{}.{}.{}.csv".format(prefix1, block, hess_style))
print "Saving results to %s" % out_csv
results_df.to_csv(out_csv, index=False, sep=',')

out_csv = os.path.join(outdir, "{}.{}.{}.csv".format(prefix2, block, hess_style))
print "Saving results to %s" % out_csv
results_df.to_csv(out_csv, index=False, sep=',')


