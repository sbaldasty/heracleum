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
#SBATCH --output=/gpfs2/scratch/sbaldast/heracleum/logs/outs/%A.out 

cd heracleum

# Make sure we have the right conda loaded
module purge
module load python3.11-anaconda/2023.09-0
source ${ANACONDA_ROOT}/etc/profile.d/conda.sh

# Create the environment if we need to
if ! conda env list | grep heracleum >/dev/null 2>&1; then
    yes | conda create --name heracleum python=3.11
fi

# Update the environment with latest project dependencies
conda env update --file environment.yaml --prune
conda activate heracleum

# Run the experiments
python -m pip install -e .
python src/_poisoneffect.py
