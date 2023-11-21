#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH -p bii-largemem
#SBATCH -A nssac_covid19

source activate biocomplexity

python ukr_mim_mdm_idp_refugee.py "$1" $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21}


