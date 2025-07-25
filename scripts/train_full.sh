#!/bin/bash

#SBATCH --job-name=d3-dna-diffusion-frac
#SBATCH --output=full-%A_%a.out
#SBATCH --error=d3-dna-diffusion-frac-%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=64GB
#SBATCH --qos=slow_nice
#SBATCH --array=0-2

source ~/.bashrc
mamba activate d3-old

cd ~/scratch/D3-DNA-Discrete-Diffusion

SEEDS=(0 42 123)
DATAFILES=("/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/data_files/DeepSTARR_data_0.h5" "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/data_files/DeepSTARR_data_42.h5" "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/data_files/DeepSTARR_data_123.h5")

echo "Training on all data: full"

for i in "${!DATAFILES[@]}"; do
    data_file="${DATAFILES[$i]}"
    seed="${SEEDS[$i]}"

    echo "Running training for $data_file with seed $SEED"

    # command line args can override any keyword in the config file
    python scripts/train.py \
        --architecture transformer \
        --wandb_project d3-ablation-2 \
        --wandb_name full-${seed} \
        --seed $seed \
        --paths.data_file $data_file \
        --optim.weight_decay 0 \
        --ngpus 2 \

        # --resume_from  \
done

echo "Completed all training runs for data: full"