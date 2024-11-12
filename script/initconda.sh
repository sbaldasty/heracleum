#!/bin/bash

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

# Install ourself locally
python -m pip install -e .
