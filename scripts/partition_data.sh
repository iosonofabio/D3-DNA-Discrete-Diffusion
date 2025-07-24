#!/bin/bash

#SBATCH --job-name=d3-dna-diffusion-frac
#SBATCH --output=d3-dna-diffusion-frac-%A_%a.out
#SBATCH --error=d3-dna-diffusion-frac-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --qos=bio_ai
#SBATCH --array=0

source ~/.bashrc
mamba activate d3

cd ~/scratch/D3-DNA-Discrete-Diffusion

# Run the training script with the current fraction_data value
python scripts/partition_data.py 
