#!/usr/bin/env Rscript

library(data.table)


args = commandArgs(trailingOnly=TRUE)

# assumed to be .gz file 
trait_file <- args[1]

gwas <-fread(trait_file, header=TRUE, showProgress=FALSE, fill=T)

# assumed to have standard headings from All_summary_statistics folder 
SE<-1/sqrt(gwas$N)

# standardized effect sizes and standard errors  
# SE = 1/sqrt(N)
# Z = beta/SE(beta)
# beta = Z*SE(beta)

gwas$BETA_STD <- gwas$Z*SE 
gwas$SE_STD

# output to data dir 
out_dir <- dirname(trait_file)
trait_prefix <- strsplit(basename(trait_file),".txt")
outfile <- paste(out_dir, paste(trait_prefix, "clean",sep='.'), sep='/')
print(outfile)
write.table(gwas, file=outfile, quote=F, row.names=F)