#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 03:00:00
#SBATCH -p bii
#SBATCH -A nssac_covid19
#SBATCH --job-name=fd_method

source activate abm_env

python disease_burden_method.py "$1" $2 $3 $4 $5


