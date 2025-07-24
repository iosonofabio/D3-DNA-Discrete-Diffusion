# Simple Dataset-Independent Architecture

The codebase has been refactored for simplicity and clarity. Here's the new, clean architecture:

## Key Principle

**Each dataset handles its own models directly - no complex generic interfaces.**

## What Changed

### 1. Clean Dataset Structure
- Each dataset in `model_zoo/{dataset}/` has its own complete `evaluate.py` and `sample.py`
- These scripts import and use their dataset-specific models directly
- No more hunting for the right model class - it's explicit in each dataset's script

### 2. Simplified Utils
- `utils/model_interface.py`: Only handles basic model loading from configs (base models only)  
- `utils/checkpoint_utils.py`: Simple checkpoint utilities, no dataset logic
- Removed complex generic scripts from `scripts/`

### 3. Explicit Checkpoint Paths
- No more guessing `last.ckpt` or searching directories
- Users must provide explicit checkpoint paths
- Clear error messages when files don't exist

## New Usage

### Evaluation
```bash
# Each dataset has its own evaluate.py
python model_zoo/deepstarr/evaluate.py \
    --architecture transformer \
    --checkpoint /path/to/model.ckpt \
    --config /path/to/config.yaml

# With oracle evaluation
python model_zoo/deepstarr/evaluate.py \
    --architecture transformer \
    --checkpoint /path/to/model.ckpt \
    --config /path/to/config.yaml \
    --use_oracle \
    --oracle_checkpoint /path/to/oracle.ckpt
```

### Sampling
```bash
# Each dataset has its own sample.py
python model_zoo/deepstarr/sample.py \
    --architecture transformer \
    --checkpoint /path/to/model.ckpt \
    --config /path/to/config.yaml \
    --num_samples 1000

# With specific conditioning
python model_zoo/deepstarr/sample.py \
    --architecture transformer \
    --checkpoint /path/to/model.ckpt \
    --config /path/to/config.yaml \
    --num_samples 1000 \
    --dev_activity 2.0 \
    --hk_activity 1.5
```

## Directory Structure

```
model_zoo/
├── deepstarr/
│   ├── evaluate.py          # Complete evaluation script
│   ├── sample.py            # Complete sampling script
│   ├── models.py            # DeepSTARR-specific models
│   ├── data.py              # DeepSTARR data loading
│   └── configs/             # DeepSTARR configurations
│       ├── transformer.yaml
│       └── convolutional.yaml
├── mpra/
│   ├── evaluate.py          # MPRA evaluation script
│   ├── sample.py            # MPRA sampling script
│   └── ...
└── promoter/
    ├── evaluate.py          # Promoter evaluation script
    ├── sample.py            # Promoter sampling script
    └── ...

utils/
├── model_interface.py       # Simple base model loading only
├── checkpoint_utils.py      # Basic checkpoint utilities
└── load_model.py           # Legacy compatibility functions
```

## Benefits of This Approach

1. **No Complexity**: Each script is self-contained and easy to understand
2. **No Redundancy**: No searching for models or complex dispatch logic
3. **Clear Ownership**: Each dataset owns its evaluation and sampling logic
4. **Easy Extension**: Adding a new dataset just requires copying and modifying one dataset's folder
5. **Direct Control**: Full control over model loading, data handling, and evaluation metrics

## For New Datasets

To add a new dataset:

1. Create `model_zoo/my_dataset/` folder
2. Copy `model_zoo/deepstarr/evaluate.py` and `sample.py`  
3. Update imports to use your dataset's models and data
4. Modify model creation, data loading, and evaluation logic as needed
5. Add your configs to `model_zoo/my_dataset/configs/`

That's it! No changes needed to `utils/` or any other part of the codebase.

## Migration from Complex Version

If you were using the previous complex generic interface:

**Before**:
```python
from utils.model_interface import load_model_from_config_and_checkpoint
model, graph, noise = load_model_from_config_and_checkpoint(config, checkpoint, device)
```

**After**:  
Just use the dataset-specific script directly:
```bash
python model_zoo/your_dataset/evaluate.py --checkpoint /path/to/checkpoint.ckpt --config /path/to/config.yaml
```

## Utils are Now Simple

The `utils/` folder is now clean and focused:

- `model_interface.py`: Basic model loading for base TransformerModel/ConvolutionalModel only
- `checkpoint_utils.py`: Simple checkpoint format handling  
- `load_model.py`: Legacy compatibility functions

No more dataset detection, dynamic imports, or complex generic interfaces!