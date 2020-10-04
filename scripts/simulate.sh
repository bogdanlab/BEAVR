#!/usr/bin/env sh
#$ -cwd
#$ -j y

source /u/local/Modules/default/init/modules.sh
module load python/2.7

H2=0.50
M_GW=500000
P_SIM=0.01
N=500000

# CHANGE THIS DIRECTORY TO YOUR OWN 
MASTER_DIR=/u/home/r/ruthjohn/BEAVR

SCRIPT_DIR=${MASTER_DIR}/scripts/helper
SIM_DIR=${MASTER_DIR}/1kg_sample/sumstats
LD_FILE=${MASTER_DIR}/1kg_sample/ld/1KG.npy

for i in {1..1}
do

    python $SCRIPT_DIR/simulate.py \
          --h2_gw $H2 \
          --M_gw $M_GW \
          --p_sim $P_SIM \
          --N $N \
          --outdir $SIM_DIR \
          --seed $i \
          --ld_file $LD_FILE 

done 

for i in {1..1}
do
 
    python $SCRIPT_DIR/simulate.py \
	  --sigma_g 0.0001 \
          --p_sim $P_SIM \
          --N $N \
          --outdir $SIM_DIR \
          --seed $i \
          --ld_file $LD_FILE 

done