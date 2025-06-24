# D3-DNA Model Zoo Usage Guide

This guide provides comprehensive instructions for training, evaluating, and sampling with the D3-DNA model zoo structure.

## 📁 Directory Structure

```
model_zoo/
├── deepstarr/                           # DeepSTARR dataset components
│   ├── config/
│   │   ├── Conv/hydra/config.yaml      # Convolutional architecture config
│   │   └── Tran/hydra/config.yaml      # Transformer architecture config
│   ├── deepstarr.py                    # PyTorch Lightning training module
│   ├── checkpoints/                    # Model checkpoints (auto-organized by date/time)
│   └── oracle_models/                  # Oracle model for evaluation
│       └── oracle_DeepSTARR_DeepSTARR_data.ckpt
├── mpra/                               # MPRA dataset components  
│   ├── config/
│   │   ├── Conv/hydra/config.yaml
│   │   └── Tran/hydra/config.yaml
│   ├── transformer_mpra.py             # MPRA-specific model (3D embedding, 3 classes)
│   ├── mpra.py                         # PyTorch Lightning training module
│   ├── checkpoints/
│   └── oracle_models/
│       └── oracle_mpra_mpra_data.ckpt
└── promoter/                           # Promoter dataset components
    ├── config/
    │   ├── Conv/hydra/config.yaml
    │   └── Tran/hydra/config.yaml
    ├── transformer_promoter.py         # Promoter-specific model (1D embedding)
    ├── checkpoints/
    └── oracle_models/
        ├── best.sei.model.pth.tar      # SEI model
        └── target.sei.names
```

## 🚀 Quick Start Commands

### Training Models

```bash
# Train DeepSTARR with Transformer architecture
python scripts/run_train_unified.py --dataset deepstarr --arch Tran

# Train DeepSTARR with Convolutional architecture
python scripts/run_train_unified.py --dataset deepstarr --arch Conv

# Train MPRA with Transformer architecture
python scripts/run_train_unified.py --dataset mpra --arch Tran

# Train MPRA with Convolutional architecture  
python scripts/run_train_unified.py --dataset mpra --arch Conv

# Train Promoter with Transformer architecture
python scripts/run_train_unified.py --dataset promoter --arch Tran

# Train Promoter with Convolutional architecture
python scripts/run_train_unified.py --dataset promoter --arch Conv
```

**Optional Parameters:**
- `--config_path`: Custom config file path (auto-resolved if not provided)

### Evaluating Models (with Oracle Models)

```bash
# Evaluate DeepSTARR model (auto-resolves oracle and data paths)
python scripts/run_evaluate_unified.py \
  --dataset deepstarr \
  --arch Tran \
  --model_path model_zoo/deepstarr/checkpoints/2024.01.15/123456/checkpoints/checkpoint_10.pth

# Evaluate MPRA model with custom paths
python scripts/run_evaluate_unified.py \
  --dataset mpra \
  --arch Conv \
  --model_path path/to/mpra/model \
  --oracle_path model_zoo/mpra/oracle_models/oracle_mpra_mpra_data.ckpt \
  --data_path mpra_data.h5

# Evaluate with custom batch size and steps
python scripts/run_evaluate_unified.py \
  --dataset deepstarr \
  --arch Tran \
  --model_path path/to/model \
  --batch_size 128 \
  --steps 500
```

**Auto-resolved Paths:**
- Oracle models: `model_zoo/[dataset]/oracle_models/[default_oracle_file]`
- Data files: `[dataset_name]_data.h5` in project root

### Sampling Sequences

```bash
# Generate 1000 unconditional samples
python scripts/run_sampling_unified.py \
  --dataset deepstarr \
  --arch Tran \
  --model_path path/to/model \
  --num_samples 1000 \
  --unconditional

# Generate samples with test set conditioning
python scripts/run_sampling_unified.py \
  --dataset mpra \
  --arch Conv \
  --model_path path/to/model \
  --num_samples 500 \
  --conditioning test \
  --data_path mpra_data.h5

# Generate samples with custom parameters
python scripts/run_sampling_unified.py \
  --dataset promoter \
  --arch Tran \
  --model_path path/to/model \
  --num_samples 2000 \
  --batch_size 64 \
  --steps 256 \
  --output_dir ./results/
```

**Conditioning Options:**
- `--unconditional`: Generate without conditioning
- `--conditioning test`: Use test set labels for conditioning
- `--conditioning valid`: Use validation set labels for conditioning  
- `--conditioning train`: Use training set labels for conditioning

## 📂 File Organization

### 1. Dataset Files (Project Root)
```
├── DeepSTARR_data.h5     # DeepSTARR dataset
├── mpra_data.h5          # MPRA dataset
└── [promoter files]      # Promoter dataset (external setup required)
```

### 2. Model Checkpoints
**Training Output:**
```
model_zoo/[dataset]/checkpoints/YYYY.MM.DD/HHMMSS/
├── checkpoints/
│   ├── checkpoint_10.pth
│   ├── checkpoint_20.pth
│   └── ...
├── checkpoints-meta/
│   └── checkpoint.pth    # For resuming training
├── samples/              # Generated samples during training
└── logs/                 # Training logs
```

**For Pre-trained Models:**
Place downloaded checkpoints anywhere under `model_zoo/[dataset]/checkpoints/`

### 3. Oracle Models
Download and place oracle models in respective folders:

**DeepSTARR:**
```bash
# Download from: https://huggingface.co/anonymous-3E42/DeepSTARR_oracle
# Place at: model_zoo/deepstarr/oracle_models/oracle_DeepSTARR_DeepSTARR_data.ckpt
```

**MPRA:**
```bash
# Download from: https://huggingface.co/anonymous-3E42/MPRA_oracle  
# Place at: model_zoo/mpra/oracle_models/oracle_mpra_mpra_data.ckpt
```

**Promoter:**
```bash
# Follow setup from: https://github.com/HannesStark/dirichlet-flow-matching
# and: https://github.com/jzhoulab/ddsm
# Place SEI model at: model_zoo/promoter/oracle_models/best.sei.model.pth.tar
```

## ⚙️ Architecture Differences

### Convolutional vs Transformer
- **Conv**: Uses convolutional layers, typically faster training
- **Tran**: Uses transformer architecture, potentially better performance

### Dataset-Specific Models
- **DeepSTARR**: Uses base transformer (2D signal embedding, 4 classes)
- **MPRA**: Custom transformer (3D signal embedding, 3 classes)  
- **Promoter**: Custom transformer (1D signal embedding, no label embedding)

## 🔧 Advanced Usage

### Custom Configuration
```bash
# Use custom config file
python scripts/run_train_unified.py \
  --dataset deepstarr \
  --arch Tran \
  --config_path path/to/custom/config.yaml
```

### Multi-GPU Training
Modify the config files to set `ngpus: N` for multi-GPU training.

### Resuming Training
Training automatically resumes from `checkpoints-meta/checkpoint.pth` if it exists.

### Custom Data Paths
```bash
# Use custom data file location
python scripts/run_evaluate_unified.py \
  --dataset deepstarr \
  --arch Tran \
  --model_path path/to/model \
  --data_path /custom/path/to/DeepSTARR_data.h5
```

## 📊 Output Files

### Training Outputs
- **Checkpoints**: `model_zoo/[dataset]/checkpoints/[timestamp]/checkpoints/checkpoint_*.pth`
- **Logs**: `model_zoo/[dataset]/checkpoints/[timestamp]/logs/`
- **Config**: `model_zoo/[dataset]/checkpoints/[timestamp]/hydra/config.yaml`

### Evaluation Outputs
- **Results**: `evaluation_results_[dataset].npz` containing:
  - `generated_sequences`: Generated sequence samples
  - `oracle_scores`: Oracle model predictions on test data
  - `predicted_scores`: Oracle model predictions on generated data
  - `mse`: Mean squared error between oracle and predicted scores

### Sampling Outputs
- **Samples**: `samples_[dataset]_[conditioning].npz` containing:
  - `generated_sequences`: Generated sequence samples  
  - `conditioning_values`: Conditioning labels used (if applicable)
  - `dataset`, `num_samples`, `sequence_length`: Metadata

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and PYTHONPATH includes project root
2. **CUDA Errors**: Check GPU availability and CUDA version compatibility  
3. **Config Not Found**: Verify dataset and architecture names are correct
4. **Oracle Model Missing**: Download oracle models to correct locations
5. **Data File Missing**: Place dataset files in project root with correct names

### Environment Setup
```bash
# Create environment
conda env create -f environment.yml
conda activate d3

# Install additional packages if needed
pip install [missing_packages]
```

### Promoter Setup
For promoter dataset, additional setup is required:
1. Follow [Dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching) setup
2. Follow [Dirichlet diffusion score model](https://github.com/jzhoulab/ddsm) setup  
3. Uncomment promoter import in `utils/data.py`

## 📝 Legacy Scripts

Original scripts are preserved for backward compatibility:
- `scripts/train.py`: Original training script
- `scripts/run_sample.py`: Original sampling for DeepSTARR/MPRA
- `scripts/run_sample_promoter.py`: Original promoter sampling

**Recommendation**: Use the new unified scripts for better experience and features.