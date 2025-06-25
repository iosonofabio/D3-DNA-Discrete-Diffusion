# PyTorch Lightning Integration for D3-DNA-Discrete-Diffusion

This document describes the PyTorch Lightning integration for the D3-DNA-Discrete-Diffusion project. The Lightning integration simplifies training while maintaining full compatibility with existing checkpoints and functionality.

## 🔥 Key Features

- **Simplified Training**: Replace manual training loops with Lightning's robust training system
- **Checkpoint Compatibility**: Full backward compatibility with existing `.pth` checkpoints
- **Multi-GPU Training**: Seamless distributed training with DDP
- **Enhanced Logging**: Built-in support for TensorBoard, WandB, and other loggers
- **Flexible Configuration**: Compatible with existing Hydra configs + Lightning-specific options

## 📁 New Files Added

```
scripts/
├── lightning_trainer.py      # Lightning module and data module
├── train_lightning.py        # Lightning training script
├── convert_checkpoints.py    # Checkpoint conversion utility
└── test_checkpoint_compatibility.py  # Compatibility tests

utils/
└── checkpoint_utils.py       # Checkpoint handling utilities

configs/
└── lightning_config.yaml     # Lightning-optimized configuration
```

## 🚀 Quick Start

### 1. Train with Lightning (New Method)

```bash
# Train DeepSTARR with Transformer architecture
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran \
    --gpus 1

# Train MPRA with Convolutional architecture
python scripts/train_lightning.py \
    --dataset mpra \
    --arch Conv \
    --gpus 2

# Resume from existing checkpoint (works with both .pth and .ckpt)
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran \
    --resume_from /path/to/checkpoint.pth \
    --gpus 1
```

### 2. Use Original Training (Still Works)

```bash
# Original training method continues to work
python scripts/run_train_unified.py \
    --dataset deepstarr \
    --arch Tran
```

## 🔄 Checkpoint Compatibility

### Automatic Compatibility

The system automatically detects and handles both checkpoint formats:

- **Original `.pth` format**: `{model, ema, optimizer, step}`
- **Lightning `.ckpt` format**: Full Lightning checkpoint with state_dict

### Loading Checkpoints

```python
# This works with both formats automatically
from utils.load_model import load_model_local

model, graph, noise = load_model_local("/path/to/checkpoint_dir", device)
```

### Converting Checkpoints

Convert existing `.pth` checkpoints to Lightning format:

```bash
# Convert single checkpoint
python scripts/convert_checkpoints.py /path/to/model_dir

# Convert all checkpoints recursively
python scripts/convert_checkpoints.py /path/to/model_zoo --recursive
```

## ⚙️ Configuration

### Lightning-Specific Config

Use the optimized Lightning configuration:

```yaml
# configs/lightning_config.yaml
lightning:
  precision: bf16-mixed
  checkpoint:
    save_top_k: 3
    monitor: val_loss
  logging:
    log_every_n_steps: 1000
  performance:
    sync_batchnorm: true
```

### WandB Integration

Enable WandB logging:

```yaml
wandb:
  enabled: true
  project: d3-dna-diffusion
  entity: your-wandb-entity
```

## 🎯 Training Examples

### Basic Training

```bash
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran
```

### Advanced Training

```bash
python scripts/train_lightning.py \
    --dataset mpra \
    --arch Conv \
    --gpus 4 \
    --max_steps 100000 \
    --val_check_interval 2000 \
    --work_dir ./my_experiment
```

### Resume Training

```bash
# Resume from original checkpoint
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran \
    --resume_from model_zoo/deepstarr/checkpoints/checkpoint.pth

# Resume from Lightning checkpoint  
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran \
    --resume_from ./lightning_runs/last.ckpt
```

## 🧪 Testing

Test checkpoint compatibility:

```bash
python scripts/test_checkpoint_compatibility.py
```

This will verify:
- ✅ Checkpoint format detection
- ✅ Original checkpoint loading
- ✅ Checkpoint conversion
- ✅ Lightning checkpoint loading
- ✅ Model output consistency

## 📊 Key Differences from Original

| Feature | Original | Lightning |
|---------|----------|-----------|
| Training Loop | Manual (~200 lines) | Automatic |
| Multi-GPU | Manual DDP setup | `devices=N` |
| Checkpointing | Manual save/load | Automatic |
| Logging | Custom logger | Multiple loggers |
| Validation | Manual scheduling | Automatic |
| Early Stopping | Not available | Built-in |
| Mixed Precision | Manual AMP | `precision='bf16-mixed'` |

## 🔧 Advanced Usage

### Custom Callbacks

```python
from pytorch_lightning.callbacks import Callback

class CustomCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Custom logic here
        pass

# Add to trainer
callbacks = [CustomCallback()]
```

### Custom Logger

```python
from pytorch_lightning.loggers import MLFlowLogger

logger = MLFlowLogger(
    experiment_name="d3-experiments",
    tracking_uri="file:./ml-runs"
)
```

## 🔍 Debugging

### Fast Development Run

```bash
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran \
    --fast_dev_run
```

### Dry Run (Setup Only)

```bash
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran \
    --dry_run
```

## 🛠️ Architecture Overview

### Lightning Module Structure

```python
class D3LightningModule(pl.LightningModule):
    def __init__(self, cfg, dataset_name):
        # Initialize SEDD model, EMA, graph, noise
        
    def training_step(self, batch, batch_idx):
        # Compute loss with accumulation
        
    def validation_step(self, batch, batch_idx):  
        # Validation with EMA weights
        
    def configure_optimizers(self):
        # Setup optimizer and scheduler
        
    def load_from_original_checkpoint(self, path):
        # Handle .pth format loading
```

### Checkpoint Compatibility Layer

```python
# Automatic format detection
if is_original_checkpoint(path):
    load_original_format()
else:
    load_lightning_format()
```

## 📈 Performance Benefits

- **Simplified Code**: ~200 lines of training loop code eliminated
- **Better Monitoring**: Rich progress bars and logging
- **Robust Training**: Built-in error handling and recovery
- **Easy Scaling**: Seamless multi-GPU and multi-node training
- **Professional Workflow**: Industry-standard training pipeline

## 🔗 Migration Path

1. **Gradual Migration**: Both training methods work side-by-side
2. **Test Compatibility**: Use `test_checkpoint_compatibility.py`
3. **Convert Checkpoints**: Use `convert_checkpoints.py` if needed
4. **Start Training**: Use `train_lightning.py` for new experiments
5. **Full Migration**: Gradually move all experiments to Lightning

## 🆘 Troubleshooting

### Common Issues

**Q: Lightning training fails with CUDA out of memory**
```bash
# Reduce batch size or use gradient accumulation
python scripts/train_lightning.py --dataset deepstarr --arch Tran \
    --config_path configs/lightning_config.yaml
```

**Q: Can't load old checkpoint**
```bash
# Verify checkpoint format
python -c "from utils.checkpoint_utils import is_original_checkpoint; print(is_original_checkpoint('path/to/checkpoint.pth'))"
```

**Q: Multi-GPU training not working**
```bash
# Ensure proper strategy
python scripts/train_lightning.py --dataset deepstarr --arch Tran --gpus 2
```

### Getting Help

1. Run compatibility tests: `python scripts/test_checkpoint_compatibility.py`
2. Check logs in `work_dir/lightning_logs/`
3. Use `--fast_dev_run` for quick debugging
4. Verify config with `--dry_run`

---

## 🎉 Summary

The Lightning integration provides a modern, robust training pipeline while maintaining 100% compatibility with existing D3 checkpoints and workflows. This enables:

- **Immediate Benefits**: Use Lightning for new experiments
- **Gradual Migration**: Continue using existing checkpoints
- **Enhanced Features**: Better logging, monitoring, and multi-GPU support
- **Future-Proof**: Built on industry-standard framework

Start using Lightning today while keeping all your existing work intact!