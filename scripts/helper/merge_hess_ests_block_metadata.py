# coding: utf8

import numpy as np
import os
import pandas as pd

# makes nice csv file

partition_dir="/u/home/r/ruthjohn/ruthjohn/unity_v3.0/sims_kk_gw/meta_data/partitions"
hess_results_dir="/u/project/pasaniuc/kangchen/projects/ukbb_local_h2g/out/chr_sim_estimate/local_h2g"
outdir = "/u/home/r/ruthjohn/ruthjohn/unity_v3.0/misc"

block_list = ['berisa', '6mb', '12mb', '24mb', '48mb']
#block_list = ['berisa']
prefix_list = ["cau_ratio_0.01_hsq_0.5_maf_0.0_ld_0.0_range_0.0_1.0", "cau_ratio_0.01_hsq_0.5_maf_0.75_ld_1.0_range_0.0_1.0"]
hess_style_list = ['hess', 'loci']

for prefix in prefix_list:
    for hess_style in hess_style_list:
        for block in block_list:
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

            # read in local_h2g estimates
            h2g_arr = np.load(os.path.join(hess_results_dir, "{}.{}.{}.npy".format(prefix, block, hess_style))) # 100 x n_blocks

            for index, row in bed_df.iterrows():
                chr_str = row['chr']
                chr = int(chr_str.split("chr")[1])
                start = row['start']
                stop = row['stop']
                temp_df = pd.DataFrame({}, columns=['chr', 'start', 'stop','local_h2g','sim'])

                temp_df['chr'] = [chr]*100
                temp_df['start'] = [start]*100
                temp_df['stop'] = [stop]*100
                temp_df['local_h2g'] = h2g_arr[:, index]
                temp_df['sim'] = np.arange(1,100+1)

                # add to overall df
                results_df = pd.concat([results_df, temp_df])

            # save resulting to a csv file
            out_csv = os.path.join(outdir, "{}.{}.{}.csv".format(prefix, block, hess_style))
            print "Saving results to %s" % out_csv
            results_df.to_csv(out_csv, index=False, sep=',')
