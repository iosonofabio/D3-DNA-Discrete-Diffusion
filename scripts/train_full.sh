#!/bin/bash

#SBATCH --job-name=d3-dna-diffusion-frac
#SBATCH --output=full-%A_%a.out
#SBATCH --error=full-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=64GB
#SBATCH --qos=slow_nice
#SBATCH --array=0-2

source ~/.bashrc
mamba activate d3-old

cd ~/D3-DNA-Discrete-Diffusion

SEEDS=(0 42 123)
DATAFILES=("/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/data_files/DeepSTARR_data_0.h5" "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/data_files/DeepSTARR_data_42.h5" "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/data_files/DeepSTARR_data_123.h5")

i=$SLURM_ARRAY_TASK_ID
data_file="${DATAFILES[$i]}"
seed="${SEEDS[$i]}"

echo "Running training for $data_file with seed $seed"

python model_zoo/deepstarr/train.py \
    --architecture transformer \
    --wandb_project d3-ablation-2 \
    --wandb_name full-${seed} \
    --seed $seed \
    --paths.data_file $data_file \
    --optim.weight_decay 0 \
    --ngpus 2 \
    --model.dropout 0.1

echo "Terminated training run for data: $data_file"