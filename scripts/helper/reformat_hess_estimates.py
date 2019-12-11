import pandas as pd
import numpy as np
import os
import re

# partition files path
parition_dir = "/u/home/r/ruthjohn/ruthjohn/unity_v3.0/sims_kk_gw/meta_data/partitions"

# h2 matrix dir
h2_matrix_dir = "/u/home/r/ruthjohn/ruthjohn/unity_v3.0/sims_kk_gw/meta_data/chr22_hess_estimates"

# output dir
outdir="/u/home/r/ruthjohn/ruthjohn/unity_v3.0/sims_kk_gw/meta_data/local_h2g_ests"

for trait in ["cau_ratio_0.01_hsq_0.25_maf_0.0_ld_0.0_range_0.0_1.0", "cau_ratio_0.01_hsq_0.25_maf_0.75_ld_1.0_range_0.0_1.0"]:

    # loop through blocks
    for block in ["6mb", "12mb", "24mb", "48mb"]:

        out_csv = os.path.join(outdir, "{}.{}.csv".format(trait, block))
        f = open(out_csv, 'w')
        f.write("chr,start,stop,local_h2g,sim\n")

        partition_bed_file=os.path.join(parition_dir, "{}.bed".format(block))
        paritition_bed = pd.read_csv(partition_bed_file, sep='\t')
        bed_chr22 = paritition_bed.loc[paritition_bed["chr"] == "chr22"]
        bed_chr22 = bed_chr22.reset_index(drop=True)

        h2_matrix_file=os.path.join(h2_matrix_dir, "{}.{}.loci.npy".format(trait, block))
        #h2_matrix_file=os.path.join(h2_matrix_dir, "{}.{}.hess.npy".format(trait, block))
        h2_matrix = np.load(h2_matrix_file)

        # loop through regions for a block size
        for index, row in bed_chr22.iterrows():
            start = row["start"]
            stop = row["stop"]
            chr = re.sub("\D", "", row["chr"])

            for sim_i in range(0, 100):
                local_h2g=h2_matrix[sim_i, index]

                # print to csv file
                f.write("%s,%s,%s,%.6g,%d\n" % (chr, start, stop, local_h2g, sim_i +1))
