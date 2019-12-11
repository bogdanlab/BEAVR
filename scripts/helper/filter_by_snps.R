#!/usr/bin/env Rscript

library(dplyr)

args = commandArgs(trailingOnly=TRUE)

gwas_file<-args[1]
snp_file<-args[2]
out_file<-args[3]

gwas_df <- read.table(gwas_file, header=T)
snp_df <- read.table(snp_file, header=F)

snp_df$SNP <- snp_df$V1

df_filter <- inner_join(gwas_df, snp_df, by="SNP")

write.table(df_filter, out_file, quote=F, row.names=F, col.names=T)