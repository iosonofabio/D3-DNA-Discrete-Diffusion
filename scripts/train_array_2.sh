#!/bin/bash

#SBATCH --job-name=d3-dna-diffusion-frac2
#SBATCH --output=d3-dna-diffusion-frac2-%A_%a.out
#SBATCH --error=d3-dna-diffusion-frac2-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH --qos=slow_nice
#SBATCH --array=0-2

source ~/.bashrc
mamba activate d3

cd ~/scratch/D3-DNA-Discrete-Diffusion

# Define array of fraction_data values
fraction_values=(0.1 0.1 0.1)

# Get the fraction value for this job array task
fraction_data=${fraction_values[$SLURM_ARRAY_TASK_ID]}

# Set a unique seed for each job using SLURM_ARRAY_TASK_ID
SEED=$(( 43 + SLURM_ARRAY_TASK_ID ))

# Set max_steps based on fraction_data
if [[ "$fraction_data" == "1.0" ]]; then
    max_steps=1000000
elif [[ "$fraction_data" == "0.75" ]]; then
    max_steps=750000
elif [[ "$fraction_data" == "0.5" ]]; then
    max_steps=500000
elif [[ "$fraction_data" == "0.25" ]]; then
    max_steps=250000
elif [[ "$fraction_data" == "0.1" ]]; then
    max_steps=100000
else
    max_steps=50000  # Default/fallback value
fi

echo "Running job array task $SLURM_ARRAY_TASK_ID with fraction_data=$fraction_data"

# Run the training script with the current fraction_data value
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran \
    --gpus 1 \
    --fraction_data $fraction_data \
    --seed $SEED \
    --max_steps $max_steps

echo "Completed job array task $SLURM_ARRAY_TASK_ID with fraction_data=$fraction_data"