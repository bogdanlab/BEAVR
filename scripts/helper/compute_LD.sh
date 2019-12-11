#!/bin/sh

# full path to snplist 
RSID_FILE=/u/flashscratch/r/ruthjohn/ukbb_height_exp/gwas/assoc_chr22.assoc.linear.snplist
RSID_FILE_CLEAN=/u/flashscratch/r/ruthjohn/ukbb_height_exp/gwas/assoc_chr22.assoc.linear.clean.snplist
GWAS=/u/flashscratch/r/ruthjohn/ukbb_height_exp/gwas/assoc_chr22.assoc.linear.clean

# get snplist 
tail -n +2 $GWAS | awk '{print $2}' > $RSID_FILE 

NAME_PREFIX=assoc_chr22.assoc.linear.clean

# full path to plink executable
PLINK=/u/home/r/ruthjohn/pasaniucdata/software/plink
OUTDIR=/u/flashscratch/r/ruthjohn/ukbb_height_exp/gwas
MAF=0.05

# get chromosome number 
CHR=22

# ref panel 
BFILE=/u/project/pasaniuc/pasaniucdata/DATA/UKBiobank/array/allchr.unrelatedbritishqced.mafhwe

# filter SNPs 
$PLINK --allow-no-sex --biallelic-only --bfile $BFILE --maf $MAF --chr $CHR --extract $RSID_FILE --out ${OUTDIR}/${NAME_PREFIX} --write-snplist

# compute LD 
$PLINK --allow-no-sex --biallelic-only --bfile $BFILE --maf $MAF --chr $CHR --extract $RSID_FILE_CLEAN --r --matrix --out ${OUTDIR}/${NAME_PREFIX}

# filter remaining SNPs 
Rscript filter_by_snps.R $GWAS $RSID_FILE_CLEAN 
