#!/bin/bash

#SBATCH --job-name=d3-dna-diffusion_transfer_learn
#SBATCH --output=transfer_learn.out
#SBATCH --error=transfer_learn.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH --qos=slow_nice

source ~/.bashrc
mamba activate d3

cd ~/scratch/D3-DNA-Discrete-Diffusion

# srun python scripts/train_lightning.py --dataset deepstarr --arch Tran # --config model_zoo/deepstarr/config/Tran/hydra/config.yaml

# srun --partition=gpuq --qos=bio_ai --mem=64G --time=48:00:00 --gres=gpu:h100:2 --pty /bin/bash

python scripts/train_lightning.py --dataset atacseq --arch Tran --gpus 1  # --config model_zoo/deepstarr/config/Tran/hydra/config.yaml