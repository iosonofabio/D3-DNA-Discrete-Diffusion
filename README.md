# Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion

This repo contains a PyTorch implementation for the paper "Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion". The training and sampling part of the code is inspired by [Score entropy discrete diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

## Design Choices

This codebase is built modularly to promote future research (as opposed to a more compact framework, which would be better for applications). The primary files are 

1. ```noise_lib.py```: the noise schedule
2. ```graph_lib```: the forward diffusion process
3. ```sampling.py```: the sampling strategies
4. ```model/```: the model architecture

## Installation

All the training and sampling related codes for D3 are in ```train_n_sample``` folder. Please navigate there and simply run

```
conda env create -f environment.yml
```

which will create a ```d3``` environment with packages installed (please provide your server username in place of ```<username>```). Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. Activate ```d3 ``` and install torch with below command

Please install other packages as required (may not have installed from ```environment.yml```).

## New Model Zoo Structure

This codebase has been refactored to organize dataset-specific components into a clean `model_zoo` structure:

```
model_zoo/
├── deepstarr/
│   ├── config/Conv/hydra/config.yaml    # Convolutional architecture config
│   ├── config/Tran/hydra/config.yaml    # Transformer architecture config
│   ├── deepstarr.py                     # PyTorch Lightning training module
│   ├── checkpoints/                     # Model checkpoints
│   └── oracle_models/                   # Oracle model files
├── mpra/
│   ├── config/Conv/hydra/config.yaml
│   ├── config/Tran/hydra/config.yaml
│   ├── transformer_mpra.py              # MPRA-specific model (3D embedding, 3 classes)
│   ├── mpra.py                          # PyTorch Lightning training module
│   ├── checkpoints/
│   └── oracle_models/
└── promoter/
    ├── config/Conv/hydra/config.yaml
    ├── config/Tran/hydra/config.yaml
    ├── transformer_promoter.py          # Promoter-specific model (1D embedding)
    ├── checkpoints/
    └── oracle_models/
```

## Quick Start with Unified Scripts

The new unified scripts support **automatic architecture selection** - no more manual code editing!

### Training (with Automatic Architecture Selection)
```bash
# Train DeepSTARR with Transformer architecture
python scripts/run_train_unified.py --dataset deepstarr --arch Tran

# Train MPRA with Convolutional architecture  
python scripts/run_train_unified.py --dataset mpra --arch Conv

# Train Promoter with Transformer architecture
python scripts/run_train_unified.py --dataset promoter --arch Tran
```

**New Features:**
- ✅ **Config-driven architecture selection** (no manual code commenting!)
- ✅ **Automatic path resolution** for configs and data files  
- ✅ **Architecture validation** between command line and config files
- ✅ **Unified interface** across all datasets and architectures

### Evaluation (with Oracle Models)
```bash
# Evaluate DeepSTARR model
python scripts/run_evaluate_unified.py --dataset deepstarr --arch Tran --model_path path/to/model

# Evaluate MPRA model
python scripts/run_evaluate_unified.py --dataset mpra --arch Conv --model_path path/to/model
```

### Sampling
```bash
# Generate 1000 samples for DeepSTARR
python scripts/run_sampling_unified.py --dataset deepstarr --arch Tran --model_path path/to/model --num_samples 1000

# Generate samples with conditioning
python scripts/run_sampling_unified.py --dataset mpra --arch Conv --model_path path/to/model --conditioning test
```

## Setup Requirements

1. **For Promoter dataset**: Follow setup from [Dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching) and [Dirichlet diffusion score model](https://github.com/jzhoulab/ddsm), then uncomment the promoter import in `utils/data.py`.

2. **Dataset files**: Place your dataset files in the project root:
   - `DeepSTARR_data.h5` for DeepSTARR
   - `mpra_data.h5` for MPRA  
   - Promoter dataset files as per external setup

3. **Oracle models**: Download and place oracle models in respective `model_zoo/[dataset]/oracle_models/` folders:
   - [DeepSTARR oracle](https://huggingface.co/anonymous-3E42/DeepSTARR_oracle)
   - [MPRA oracle](https://huggingface.co/anonymous-3E42/MPRA_oracle)

## Legacy Training (Original Method)

For backward compatibility, the original training method is still available:

```bash
python scripts/train.py noise.type=geometric graph.type=uniform model=small model.scale_by_sigma=False
```

This creates a new directory `direc=model_zoo/[dataset]/checkpoints/DATE/TIME` with the following structure:
```
├── direc
│   ├── hydra
│   │   ├── config.yaml
│   │   ├── ...
│   ├── checkpoints
│   │   ├── checkpoint_*.pth
│   ├── checkpoints-meta
│   │   ├── checkpoint.pth
│   ├── samples
│   │   ├── iter_*
│   │   │   ├── sample_*.txt
│   ├── logs
```
Here, `checkpoints-meta` is used for reloading the run following interruptions, `samples` contains generated sequences as the run progresses, and `logs` contains the run output. Arguments can be added with `ARG_NAME=ARG_VALUE`, with important ones being:
```
ngpus                     the number of gpus to use in training (using pytorch DDP)
noise.type                geometric
graph.type                uniform
model                     small
model.scale_by_sigma      False
```
## File Organization

### Dataset Files
Place your dataset files in the project root:
- `DeepSTARR_data.h5` - DeepSTARR dataset
- `mpra_data.h5` - MPRA dataset  
- Promoter dataset files (follow external setup instructions)

### Model Checkpoints
Trained model checkpoints are automatically saved to:
- `model_zoo/[dataset]/checkpoints/YYYY.MM.DD/HHMMSS/`

For pre-trained models, download and place in:
- `model_zoo/[dataset]/checkpoints/` (any subdirectory structure)

### Oracle Models  
Download oracle models and place in:
- `model_zoo/deepstarr/oracle_models/oracle_DeepSTARR_DeepSTARR_data.ckpt`
- `model_zoo/mpra/oracle_models/oracle_mpra_mpra_data.ckpt`
- `model_zoo/promoter/oracle_models/best.sei.model.pth.tar` (SEI model)

### Legacy Sampling (Original Method)

The original sampling scripts are still available:
- `scripts/run_sample.py` - For DeepSTARR/MPRA (requires manual dataset switching)
- `scripts/run_sample_promoter.py` - For Promoter dataset

**Note**: Use the new unified scripts (`run_evaluate_unified.py`, `run_sampling_unified.py`) for better experience.

We can run sampling using a command 

```
python run_sample.py --model_path MODEL_PATH --steps STEPS
```
The ```model_path``` argument should point to ```exp_local/"dataset"/"arch"/``` folder. If you trained a D3 model, the folder should be ```exp_local/"dataset"/${now:%Y.%m.%d}/${now:%H%M%S}```, which should already be created during training.
In any case, this will generate samples for all the true test activity levels and store them in the model path. Also it will calculate the mse (between true test vs generated) through the oracle predictions. If you face any key mismatch issue with the pretrained D3 models, please consider un/commenting related variables from model architecture details to solve them.

### Datasets and Oracles

We provide preprocessed datasets for [DeepSTARR](https://huggingface.co/datasets/anonymous-3E42/DeepSTARR_preprocessed), [MPRA](https://huggingface.co/datasets/anonymous-3E42/MPRA_preprocessed) and oracle models at [DeepSTARR](https://huggingface.co/anonymous-3E42/DeepSTARR_oracle), [MPRA](https://huggingface.co/anonymous-3E42/MPRA_oracle).

### Pretrained Models

We provide pretrained models for Promoter, DeepSTARR and MPRA datasets below, each with transformer and convolution architectures.

1. [Promoter with transformer](https://huggingface.co/anonymous-3E42/Promoter_D3_Tran_model)
2. [Promoter with convolution](https://huggingface.co/anonymous-3E42/Promoter_D3_Conv_model)
3. [DeepSTARR with transformer](https://huggingface.co/anonymous-3E42/DeepSTARR_D3_Tran_model)
4. [DeepSTARR with convolution](https://huggingface.co/anonymous-3E42/DeepSTARR_D3_Conv_model)
5. [MPRA with transformer](https://huggingface.co/anonymous-3E42/MPRA_D3_Tran_model)
6. [MPRA with convolution](https://huggingface.co/anonymous-3E42/MPRA_D3_Conv_model)

### Sample generated data

We generate data points conditioned on the same activity levels for every dataset, where we only used test splits. Please find below the links to the generated data sets where D3 trained with transformer and convolution architectures.

1.[Promoter generated samples with D3 transformer](https://huggingface.co/datasets/anonymous-3E42/Promoter_sample_generated_D3_Tran)

2.[Promoter generated samples with D3 convolution](https://huggingface.co/datasets/anonymous-3E42/Promoter_sample_generated_D3_Conv)

3.[DeepSTARR generated samples with D3 transformer](https://huggingface.co/datasets/anonymous-3E42/DeepSTARR_sample_generated_D3_Tran)

4.[DeepSTARR generated samples with D3 convolution](https://huggingface.co/datasets/anonymous-3E42/DeepSTARR_sample_generated_D3_Conv)

5.[MPRA generated samples with D3 transformer](https://huggingface.co/datasets/anonymous-3E42/MPRA_sample_generated_D3_Tran)

6.[MPRA generated samples with D3 convolution](https://huggingface.co/datasets/anonymous-3E42/MPRA_sample_generated_D3_Conv)
