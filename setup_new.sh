#!/bin/bash

###SETUP ENVIRONMENT####
# Load modules
module load cuda11.8/toolkit/11.8.0
mamba init
source ~/.bashrc

# Activate/create the 'd3-old' environment
if mamba env list | grep -q 'd3-old'; then
    echo "Environment 'd3-old' found. Activating it..."
    mamba activate d3-old
else
    echo "Environment 'd3-old' not found. Creating it..."
    mamba create -n d3-old python=3.9 -y
    mamba activate d3-old
fi

#Try to install packages to ensure that everything is up to date
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install wandb dm-tree pytorch-lightning==2.3 transformers hydra-core omegaconf hydra-submitit-launcher scikit-learn
mamba install ipykernel ipywidgets biotite matplotlib seaborn h5py --yes # lightning
python -m pip -v install flash-attn==2.6.3 --no-build-isolation
pip install -U tensorboard tensorboardX
