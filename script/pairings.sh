#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=01-23:59:59
#SBATCH --job-name=heracleum
#SBATCH --mail-user=steven.baldasty@uvm.edu,yeonhee.yeh@uvm.edu
#SBATCH --mail-type=ALL
#SBATCH --output=%A.out 

cd heracleum
source script/initconda.sh
mkdir -p out/pairings
rm out/pairings/*
python src/_pairings.py
