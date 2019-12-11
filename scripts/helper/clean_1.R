#!/usr/bin/env Rscript

#$ -cwd
#$ -j y
#$ -l h_data=5G,h_rt=1:00:00,highp
#$ -e clean_1.log
#$ -o clean_1.log 

library(data.table)

args = commandArgs(trailingOnly=TRUE)

filename <- args[1]
out <- args[2]

# sample size 
N <- 337205

df <- fread(filename, header=T, stringsAsFactors=F)

df$Z <-df$BETA /df$SE
df$N <- N

write.table(df, out, row.names=FALSE, quote=FALSE)