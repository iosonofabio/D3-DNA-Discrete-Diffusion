# PyTorch Lightning Phase 2 - Complete Model Zoo Integration

This document describes Phase 2 of the Lightning integration, which adds full support for all dataset-specific model implementations in the model_zoo.

## 🎯 What's New in Phase 2

### Dataset-Specific Model Support
- **Promoter Dataset**: Uses `model_zoo/promoter/transformer_promoter.py` SEDD implementation
- **MPRA Dataset**: Uses `model_zoo/mpra/transformer_mpra.py` SEDD implementation  
- **DeepSTARR Dataset**: Uses generic `model/transformer.py` SEDD implementation
- **Automatic Routing**: Factory pattern automatically selects correct implementation

### Enhanced Lightning Modules
- `PromoterD3LightningModule`: Specialized for promoter dataset with sequence length 1024
- `MPRAD3LightningModule`: Specialized for MPRA dataset with sequence length 200
- `D3LightningModule`: Generic module for DeepSTARR and other datasets

## 🏗️ Architecture Overview

### Factory Pattern
```python
def create_lightning_module(cfg, dataset_name):
    if dataset_name == 'promoter':
        return PromoterD3LightningModule(cfg)
    elif dataset_name == 'mpra':
        return MPRAD3LightningModule(cfg)
    else:
        return D3LightningModule(cfg, dataset_name)
```

### Model Class Routing
```python
def get_model_class_for_dataset(dataset):
    if dataset == 'promoter':
        from model_zoo.promoter.transformer_promoter import SEDD
        return SEDD
    elif dataset == 'mpra':
        from model_zoo.mpra.transformer_mpra import SEDD
        return SEDD
    else:
        from model.transformer import SEDD
        return SEDD
```

## 🚀 Usage Examples

### Training with Dataset-Specific Models

```bash
# Train Promoter with Transformer - uses promoter-specific SEDD
python scripts/train_lightning.py \
    --dataset promoter \
    --arch Tran \
    --gpus 1

# Train MPRA with Convolutional - uses MPRA-specific SEDD  
python scripts/train_lightning.py \
    --dataset mpra \
    --arch Conv \
    --gpus 1

# Train DeepSTARR with Transformer - uses generic SEDD
python scripts/train_lightning.py \
    --dataset deepstarr \
    --arch Tran \
    --gpus 1
```

### All Supported Combinations
- ✅ `deepstarr/Conv` - Generic SEDD, convolutional architecture, 249 length
- ✅ `deepstarr/Tran` - Generic SEDD, transformer architecture, 249 length
- ✅ `mpra/Conv` - MPRA-specific SEDD, convolutional architecture, 200 length
- ✅ `mpra/Tran` - MPRA-specific SEDD, transformer architecture, 200 length
- ✅ `promoter/Conv` - Promoter-specific SEDD, convolutional architecture, 1024 length
- ✅ `promoter/Tran` - Promoter-specific SEDD, transformer architecture, 1024 length

## 🔄 Checkpoint Compatibility

### Enhanced Load Functions
The `load_model_local()` function now automatically detects the correct model implementation:

```python
# Automatically uses correct SEDD implementation
model, graph, noise = load_model_local("/path/to/checkpoint_dir", device)
```

### Conversion Support
```bash
# Convert any dataset-specific checkpoint to Lightning format
python scripts/convert_checkpoints.py model_zoo/promoter/checkpoints/
python scripts/convert_checkpoints.py model_zoo/mpra/checkpoints/
```

## 📊 Evaluation Compatibility

The existing `run_evaluate_unified.py` continues to work unchanged because:

1. **Enhanced `load_model_local()`**: Automatically detects and loads correct model implementation
2. **Identical Interface**: Same sampling and evaluation logic
3. **Oracle Model Preservation**: Uses existing oracle models (`PL_DeepSTARR`, `PL_mpra`)
4. **Results Consistency**: Identical MSE calculations and output formats

```bash
# Works with both original and Lightning checkpoints
python scripts/run_evaluate_unified.py \
    --dataset promoter \
    --arch Tran \
    --model_path /path/to/lightning/checkpoint
```

## 🧪 Testing

### Comprehensive Test Suite
```bash
# Test all dataset/architecture combinations
python scripts/test_lightning_integration.py
```

Tests include:
- ✅ Model class routing verification
- ✅ Lightning module factory functionality  
- ✅ Config loading for all combinations
- ✅ Forward pass validation
- ✅ Error handling and edge cases

### Checkpoint Compatibility Testing
```bash
# Test checkpoint compatibility across formats
python scripts/test_checkpoint_compatibility.py
```

## 🔧 Technical Details

### Lightning Module Inheritance
```python
class PromoterD3LightningModule(D3LightningModule):
    def __init__(self, cfg):
        super().__init__(cfg, dataset_name='promoter')
        
        # Use promoter-specific SEDD model
        from model_zoo.promoter.transformer_promoter import SEDD
        self.score_model = SEDD(cfg)
        
        # Update EMA for new model
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(), 
            decay=cfg.training.ema
        )
```

### Config Auto-Resolution
The training script automatically resolves configs:
```
--dataset promoter --arch Tran
→ model_zoo/promoter/config/Tran/hydra/config.yaml
→ PromoterD3LightningModule with transformer_promoter.SEDD
```

### Path Safety
All model imports use safe path manipulation:
```python
sys.path.insert(0, 'model_zoo/promoter')
try:
    from transformer_promoter import SEDD
    return SEDD
finally:
    sys.path.pop(0)  # Always clean up
```

## ⚡ Performance Benefits

- **Zero Overhead**: Factory pattern adds no computational cost
- **Memory Efficient**: Only loads required model implementation
- **Fast Routing**: O(1) lookup time for model selection
- **Lazy Loading**: Models imported only when needed

## 🎯 Migration Guide

### For Existing Users
1. **No Changes Needed**: All existing checkpoints work automatically
2. **Enhanced Features**: Get dataset-specific optimizations for free
3. **Same Commands**: All training/evaluation commands unchanged

### For New Projects
1. **Use Lightning Training**: `python scripts/train_lightning.py`
2. **Specify Dataset**: `--dataset promoter|mpra|deepstarr`
3. **Pick Architecture**: `--arch Conv|Tran`
4. **Everything Else Automatic**: Factory pattern handles the rest

## 🚦 Status Summary

| Dataset | Architecture | Status | Model Implementation |
|---------|-------------|--------|---------------------|
| deepstarr | Conv | ✅ Ready | `model.transformer.SEDD` |
| deepstarr | Tran | ✅ Ready | `model.transformer.SEDD` |
| mpra | Conv | ✅ Ready | `model_zoo.mpra.transformer_mpra.SEDD` |
| mpra | Tran | ✅ Ready | `model_zoo.mpra.transformer_mpra.SEDD` |
| promoter | Conv | ✅ Ready | `model_zoo.promoter.transformer_promoter.SEDD` |
| promoter | Tran | ✅ Ready | `model_zoo.promoter.transformer_promoter.SEDD` |

## 🔮 What's Next

The Lightning integration is now **complete** with:
- ✅ Full model_zoo support
- ✅ All dataset/architecture combinations working
- ✅ Complete checkpoint compatibility  
- ✅ Identical evaluation results
- ✅ Comprehensive testing suite

You can now use PyTorch Lightning for all your D3-DNA training while maintaining 100% compatibility with existing workflows and checkpoints!