#!/bin/bash --login
#SBATCH -J ganexp1
#SBATCH -o %x.o.%J
#SBATCH -e %x.e.%J
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH -p dev
#SBATCH --time=01:00:00
#SBATCH --account=


venv activate ~/torch/bin/
module purge
module load python/3.6.9-intel2019u5
python3 master.py

#-p htc
