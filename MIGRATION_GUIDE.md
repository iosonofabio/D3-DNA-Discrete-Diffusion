# Migration Guide: Dataset-Independent Model Loading

This guide explains the changes made to make the D3-DNA codebase fully dataset-independent outside of `model_zoo/` folders.

## What Changed

### 1. No More Dataset Name Guessing
**Before**: Model loading functions guessed dataset names from directory paths or used hardcoded defaults.

**After**: All model loading now requires explicit configuration files that specify the dataset.

### 2. Explicit Checkpoint Paths
**Before**: Functions searched for checkpoints using hardcoded patterns like `last.ckpt`, `checkpoint.pth`.

**After**: Users must provide explicit checkpoint paths. No more assumptions about checkpoint names.

### 3. Generic Model Interface
**Before**: Dataset-specific factory loading with internal dataset logic.

**After**: Generic `ModelLoader` class that works with any properly configured dataset.

## New API Usage

### Basic Model Loading

```python
from utils.model_interface import load_model_from_config_and_checkpoint
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load('path/to/config.yaml')

# Load model with explicit paths
model, graph, noise = load_model_from_config_and_checkpoint(
    config='path/to/config.yaml',
    checkpoint_path='path/to/checkpoint.ckpt',
    device='cuda'
)
```

### Model Loading with EMA

```python
from utils.model_interface import load_model_with_ema_from_config_and_checkpoint

model, graph, noise, ema = load_model_with_ema_from_config_and_checkpoint(
    config='path/to/config.yaml',
    checkpoint_path='path/to/checkpoint.ckpt',
    device='cuda'
)
```

### Using ModelLoader Class

```python
from utils.model_interface import ModelLoader
from omegaconf import OmegaConf

# Create loader
loader = ModelLoader(device='cuda')

# Load configuration
config = OmegaConf.load('path/to/config.yaml')

# Load model
model, graph, noise = loader.load_model_from_config(config, 'path/to/checkpoint.ckpt')

# Or load with EMA
model, graph, noise, ema = loader.load_model_with_ema(config, 'path/to/checkpoint.ckpt')
```

### Lightning Module Loading

```python
from utils.model_interface import load_lightning_module_from_checkpoint

# Load Lightning module directly (dataset type inferred from checkpoint)
lightning_model = load_lightning_module_from_checkpoint(
    checkpoint_path='path/to/lightning_checkpoint.ckpt',
    device='cuda'
)

# Extract components
model = lightning_model.score_model
graph = lightning_model.graph  
noise = lightning_model.noise
```

## Updated Evaluation Usage

### Command Line
```bash
# Now requires explicit config file
python model_zoo/deepstarr/evaluate.py \
    --architecture transformer \
    --config path/to/config.yaml \
    --checkpoint path/to/model.ckpt

# With oracle evaluation
python model_zoo/deepstarr/evaluate.py \
    --architecture transformer \
    --config path/to/config.yaml \
    --checkpoint path/to/model.ckpt \
    --use_oracle \
    --oracle_checkpoint path/to/oracle.ckpt
```

## Updated Sampling Usage

### Command Line
```bash
# Now requires explicit config file
python model_zoo/deepstarr/sample.py \
    --architecture transformer \
    --config path/to/config.yaml \
    --checkpoint path/to/model.ckpt \
    --num_samples 1000

# With specific sampling method and steps
python model_zoo/deepstarr/sample.py \
    --architecture transformer \
    --config path/to/config.yaml \
    --checkpoint path/to/model.ckpt \
    --num_samples 1000 \
    --method ddim \
    --num_steps 50 \
    --eta 0.0
```

### Programmatic Usage

#### Evaluation
```python
from scripts.evaluate import BaseEvaluator
from omegaconf import OmegaConf

# Create evaluator (dataset-specific)
evaluator = MyDatasetEvaluator('my_dataset')

# Load config explicitly
config = OmegaConf.load('path/to/config.yaml')

# Run evaluation with explicit paths
results = evaluator.evaluate(
    checkpoint_path='path/to/checkpoint.ckpt',
    config=config,
    architecture='transformer',
    split='test'
)
```

#### Sampling
```python
from scripts.sample import BaseSampler
from omegaconf import OmegaConf

# Create sampler (dataset-specific)
sampler = MyDatasetSampler('my_dataset')

# Load config explicitly
config = OmegaConf.load('path/to/config.yaml')

# Generate samples with explicit paths
sequences = sampler.sample(
    checkpoint_path='path/to/checkpoint.ckpt',
    config=config,
    architecture='transformer',
    num_samples=1000,
    method='ddpm',
    num_steps=128
)
```

## Configuration Requirements

Your configuration files must now include the dataset specification:

```yaml
dataset:
  name: "deepstarr"  # or "mpra", "promoter"
  # ... other dataset config

model:
  architecture: "transformer"  # or "convolutional"
  # ... other model config

training:
  ema: 0.999
  # ... other training config
```

## Breaking Changes

### 1. `load_model()` Function
**Before**:
```python
model, graph, noise = load_model(directory, device)
```

**After**:
```python
# Option 1: Use new generic function
model, graph, noise = load_model_from_config_and_checkpoint(
    config='path/to/config.yaml',
    checkpoint_path='path/to/checkpoint.ckpt',
    device=device
)

# Option 2: Use legacy function with explicit config
model, graph, noise = load_model_hf(directory, device)  # Still works if directory has config.yaml
```

### 2. `load_model_local()` Function
**Before**:
```python
model, graph, noise = load_model_local(directory, device)
```

**After**:
```python
# Now requires explicit checkpoint path
model, graph, noise = load_model_local(
    root_dir=directory, 
    device=device,
    checkpoint_path='path/to/specific/checkpoint.ckpt'  # Required!
)

# Or use new interface
model, graph, noise = load_model_from_config_and_checkpoint(
    config='path/to/config.yaml',
    checkpoint_path='path/to/checkpoint.ckpt',
    device=device
)
```

### 3. Evaluation Scripts
**Before**:
```bash
python model_zoo/deepstarr/evaluate.py --architecture transformer --checkpoint model.ckpt
```

**After**:
```bash
# Must provide config
python model_zoo/deepstarr/evaluate.py \
    --architecture transformer \
    --config path/to/config.yaml \
    --checkpoint path/to/model.ckpt
```

## Checkpoint Path Discovery

If you need to find checkpoints in a directory, use the new utility:

```python
from utils.load_model import find_checkpoint_in_directory

# Find checkpoint using common patterns
checkpoint_path = find_checkpoint_in_directory('/path/to/model/directory')

# Then load explicitly
model, graph, noise = load_model_from_config_and_checkpoint(
    config='path/to/config.yaml',
    checkpoint_path=checkpoint_path,
    device='cuda'
)
```

## Benefits of New Approach

1. **No Dataset Dependencies**: Code outside `model_zoo/` folders works with any dataset
2. **Explicit Configuration**: No more guessing dataset types from directory names
3. **Flexible Checkpoint Paths**: Users specify exact checkpoint files
4. **Better Error Messages**: Clear errors when files are missing
5. **Consistent Interface**: Same API works for all datasets
6. **Future-Proof**: Easy to add new datasets without changing core code

## Backward Compatibility

Most legacy functions still work but may issue deprecation warnings. The new approach is recommended for all new code.

## Troubleshooting

### "Config file is required" Error
**Solution**: Always provide a `--config` argument or config file path.

### "Checkpoint not found" Error  
**Solution**: Use absolute paths or check that the checkpoint file exists at the specified location.

### "Required configuration field missing" Error
**Solution**: Ensure your config includes `dataset.name`, `model.architecture`, and `training.ema` fields.

### Import Errors with Lightning Modules
**Solution**: Ensure the dataset-specific train.py files are in the correct `model_zoo/{dataset}/` locations.