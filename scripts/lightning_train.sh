#!/bin/bash

#SBATCH --job-name=d3-dna-diffusion
#SBATCH --output=d3-dna-diffusion.out
#SBATCH --error=d3-dna-diffusion.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:4
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=4
# #SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --qos=bio_ai

source ~/.bashrc
mamba activate d3

cd ~/scratch/D3-DNA-Discrete-Diffusion

# srun python scripts/train_lightning.py --dataset deepstarr --arch Tran # --config model_zoo/deepstarr/config/Tran/hydra/config.yaml

python scripts/train_lightning.py --dataset deepstarr --arch Tran # --config model_zoo/deepstarr/config/Tran/hydra/config.yaml