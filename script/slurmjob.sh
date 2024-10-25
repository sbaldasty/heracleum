#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=01-23:59:59
#SBATCH --job-name=heracleum
#SBATCH --mail-user=steven.baldasty@uvm.edu,yeonhee.yeh@uvm.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/gpfs2/scratch/sbaldast/heracleum/logs/outs/%A.out 

conda env update heracleum/environment.yaml
python src/_poisoneffect.py
