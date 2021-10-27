#!/bin/bash --login
#SBATCH -J ganexp1
#SBATCH -o %x.o.%J
#SBATCH -e %x.e.%J
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH -p dev
#SBATCH --time=01:00:00
#SBATCH --account=


pyhton3 -m venv activate ~/torch/bin/
cd ~/exp/
pyhton3 master.py

#-p htc
