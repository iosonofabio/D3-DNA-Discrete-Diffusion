#!/bin/bash

###SETUP ENVIRONMENT####
# # Load modules
module load cuda12.3/toolkit/12.3.2
mamba init
source ~/.bashrc

# Activate/create the 'd3' environment
if mamba env list | grep -q 'd3'; then
    echo "Environment 'd3' found. Activating it..."
    mamba activate d3
else
    echo "Environment 'd3' not found. Creating it..."
    mamba create -n d3 python=3.11 -y
    mamba activate d3
fi

#Try to install packages to ensure that everything is up to date
mamba install pytorch=2.5 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia --yes
pip install wandb dm-tree pytorch-lightning transformers hydra-core omegaconf hydra-submitit-launcher scikit-learn
mamba install ipykernel ipywidgets biotite matplotlib seaborn h5py --yes # lightning
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -U tensorboard tensorboardX