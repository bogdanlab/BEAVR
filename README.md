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
