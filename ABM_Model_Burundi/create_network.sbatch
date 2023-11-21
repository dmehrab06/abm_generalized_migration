#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000
#SBATCH -t 08:00:00
#SBATCH -p bii
#SBATCH -A nssac_covid19

source activate biocomplexity

python create_network_of_households.py "$1"


