# BEAVR

BEAVR is a Bayesian framework designed to perform local polygenicity estimation within a region. 

## Basic dependencies

- Python2.7 (we recommend using anaconda)
- numpy
- scikit-learn

## Installation

```
git clone https://github.com/bogdanlab/BEAVR.git
```

## Running with example data

```
cd BEAVR/
cd scripts/
```

Change the `MASTER_PATH` varible to reflect the directory where you installed BEAVR. See below:

```
#!/usr/bin/env sh

MASTER_PATH=/u/home/r/ruthjohn/BEAVR
SCRIPT_DIR=${MASTER_PATH}/scripts
LOCI_DIR=${MASTER_PATH}/data
SRC_DIR=${MASTER_PATH}/src
```

Simply, run the test script:
```
./run_beavr.sh
```

# Running with your own data

## Data

As input, BEAVR requires a file with SNP-SNP correlations, or LD (no header). Additionally, we require a file with standardized effects from gwas. Each region requires its own set of files. For now, we require that the header of the standardized effect sizes be labeled as `BETA_TRUE` (we are currently working to allow users to specify their own header); all other columns will be ignored.

For details on how to standardize the effects from GWAS, please see: https://huwenboshi.github.io/data%20management/2017/11/23/tips-for-formatting-gwas-summary-stats.html

The first few lines of each file are shown below:

### .ld file

```
1.000000000000000000e+00 0.200000000000000000e+00 0.300000000000000
0.300000000000000000e+00 1.000000000000000000e+00 0.100000000000000
0.500000000000000000e+00 0.000500000000000000e+00 1.000000000000000
```

### .gwas file

```
BETA_STD BETA_TRUE n snp z BETA_STD_I
0.000692574347539 0.0 500000 rs116911124 0.489724017621 0.000692574347539
-0.000478672356416 -0.0 500000 rs131533 -0.338472469188 -0.000478672356416
```

## Step 1: Inverse transform gwas effects and LD

To avoid additional computations, we do an inverse transformation on the GWAS effects and LD matrix. Since this only needs to be performed once, we do these computations in a separate step. We provide scripts below:

```
# transform LD
python ${SCRIPT_DIR}/helper/half_ld.py.py \
       --ld_file $LD_FILE
       --ld_out $LD_NEG_HALF_FILE

# transform betas
python ${SCRIPT_DIR}/helper/transform_betas.py  \
          --gwas_file $LOCUS_FILE  \
          --ld_neg_half $LD_NEG_HALF_FILE
```

## Step 2: Estimate local polygenicity

```
        python ${SRC_DIR}/main_new.py \
              --seed $SEED \
              --N $N \
              --id $PREFIX \
              --its $ITS \
              --ld_half_file $LD_HALF_FILE \
              --gwas_file $LOCUS_FILE  \
              --outdir $LOCI_DIR \
  	          --H_gw $H2 \
              --M_gw $M_GW \
 	            --dp
``` 
