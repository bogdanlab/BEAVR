#!/usr/bin/env sh

MASTER_PATH=/u/home/r/ruthjohn/BEAVR
SCRIPT_DIR=${MASTER_PATH}/scripts
LOCI_DIR=${MASTER_PATH}/ukbb_sample/sumstats
SRC_DIR=${MASTER_PATH}/src

LD_HALF_FILE=${MASTER_PATH}/ukbb_sample/ld/ukbb.1000.half_ld.npy
LD_NEG_HALF_FILE=${MASTER_PATH}/ukbb_sample/ld/ukbb.1000.ld.neg_half_ld.npy

    H2=0.001
    N=500000
    M=1000
    SEED=2019

        # run!
        LOCUS_FILE=${LOCI_DIR}/p_0.01_h2_0.5_N_500000_ld_1_M_1000_1.gwas

	PREFIX="ukbb_test"

        # preprocess betas
        python ${SCRIPT_DIR}/helper/transform_betas.py  \
          --gwas_file $LOCUS_FILE  \
          --ld_neg_half $LD_NEG_HALF_FILE

	# file suffix
        ITS=1000
        python ${SRC_DIR}/main_new.py \
              --seed $SEED \
              --N $N \
              --id $PREFIX \
              --its $ITS \
              --ld_half_file $LD_HALF_FILE \
              --gwas_file $LOCUS_FILE  \
              --outdir $LOCI_DIR \
  	      --H_gw $H2 \
              --M_gw $M \
 	      --dp

