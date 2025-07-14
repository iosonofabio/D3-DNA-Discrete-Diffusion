#!/bin/bash

#SBATCH --job-name=d3-dna-diffusion-frac
#SBATCH --output=d3-dna-diffusion-frac-%A_%a.out
#SBATCH --error=d3-dna-diffusion-frac-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH --qos=bio_ai
#SBATCH --array=0-8

source ~/.bashrc
mamba activate d3

cd ~/scratch/D3-DNA-Discrete-Diffusion

# Define array of fraction_data values
fraction_values=(0.25 0.25 0.5 0.5 0.75 0.75 1.0 1.0)

# Get the fraction value for this job array task
fraction_data=${fraction_values[$SLURM_ARRAY_TASK_ID]}

echo "Running job array task $SLURM_ARRAY_TASK_ID with fraction_data=$fraction_data"

# Run the training script with the current fraction_data value
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran \
    --gpus 1 \
    --fraction_data $fraction_data

echo "Completed job array task $SLURM_ARRAY_TASK_ID with fraction_data=$fraction_data"