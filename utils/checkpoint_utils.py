"""
Utilities for handling checkpoint compatibility between original D3 format and PyTorch Lightning format.
"""

import os
import torch
from typing import Dict, Any, Optional
from pathlib import Path


def is_original_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint is in original D3 format (.pth with model, ema, optimizer, step)."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Original format has these specific keys
        required_keys = {'model', 'ema', 'step'}
        return all(key in checkpoint for key in required_keys)
    except:
        return False


def is_lightning_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint is in Lightning format."""
    return checkpoint_path.endswith('.ckpt') and os.path.exists(checkpoint_path)


def convert_original_to_lightning_state(original_checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Convert original D3 checkpoint format to Lightning state dict format."""
    
    lightning_state = {
        'state_dict': {},
        'lr_schedulers': [],
        'epoch': 0,
        'global_step': original_checkpoint.get('step', 0),
        'pytorch-lightning_version': '2.0.0',
        'hyper_parameters': {},
    }
    
    # Convert model weights
    model_state = original_checkpoint.get('model', {})
    for key, value in model_state.items():
        lightning_state['state_dict'][f'score_model.{key}'] = value
    
    # Convert EMA weights - store as separate state
    if 'ema' in original_checkpoint:
        ema_state = original_checkpoint['ema']
        for key, value in ema_state.items():
            lightning_state['state_dict'][f'ema.{key}'] = value
    
    # Store optimizer state if available (though Lightning will reinitialize)
    if 'optimizer' in original_checkpoint:
        lightning_state['optimizer_states'] = [original_checkpoint['optimizer']]
    
    return lightning_state


def convert_pth_to_ckpt(pth_path: str, ckpt_path: str, cfg: Optional[Any] = None) -> str:
    """Convert original .pth checkpoint to Lightning .ckpt format."""
    
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Original checkpoint not found: {pth_path}")
    
    if not is_original_checkpoint(pth_path):
        raise ValueError(f"File {pth_path} is not in original D3 checkpoint format")
    
    print(f"Converting {pth_path} to Lightning format...")
    
    # Load original checkpoint
    original_checkpoint = torch.load(pth_path, map_location='cpu')
    
    # Convert to Lightning format
    lightning_state = convert_original_to_lightning_state(original_checkpoint)
    
    # Add configuration if provided
    if cfg is not None:
        lightning_state['hyper_parameters'] = {
            'cfg': cfg,
            'original_step': original_checkpoint.get('step', 0)
        }
    
    # Save as Lightning checkpoint
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(lightning_state, ckpt_path)
    
    print(f"✓ Converted checkpoint saved to: {ckpt_path}")
    print(f"✓ Original step: {original_checkpoint.get('step', 0)}")
    
    return ckpt_path


def load_weights_from_original_checkpoint(model, ema, checkpoint_path: str, device: str = 'cpu') -> int:
    """Load only model and EMA weights from original checkpoint, return step count."""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
        print("✓ Loaded model weights from original checkpoint")
    
    # Load EMA weights
    if 'ema' in checkpoint and ema is not None:
        ema.load_state_dict(checkpoint['ema'])
        print("✓ Loaded EMA weights from original checkpoint")
    
    step = checkpoint.get('step', 0)
    print(f"✓ Checkpoint was at step: {step}")
    
    return step


def get_model_class_for_checkpoint(root_dir: str, dataset_name: str = None):
    """Determine the correct model class based on config and dataset."""
    try:
        from utils.utils import load_hydra_config_from_run
        cfg = load_hydra_config_from_run(root_dir)
        
        # Try to get dataset from config
        if dataset_name is None and hasattr(cfg, 'data') and hasattr(cfg.data, 'train'):
            dataset_name = cfg.data.train
        
        # Use dataset factory to get model creation function
        from utils.dataset_factory import get_factory
        factory = get_factory()
        
        # Return a factory function that creates the model
        def create_model_fn(cfg):
            architecture = getattr(cfg.model, 'architecture', 'transformer')
            return factory.create_model(dataset_name, cfg, architecture)
        
        return create_model_fn
            
    except Exception as e:
        print(f"Warning: Could not determine dataset-specific model, using default: {e}")
        # Return default factory function
        from utils.dataset_factory import get_factory
        factory = get_factory()
        
        def create_model_fn(cfg):
            architecture = getattr(cfg.model, 'architecture', 'transformer')
            return factory.create_model('deepstarr', cfg, architecture)  # Default to deepstarr
        
        return create_model_fn


def update_load_model_local(root_dir: str, device: str):
    """Enhanced version of load_model_local that handles both checkpoint formats."""
    from utils.utils import load_hydra_config_from_run
    from model.ema import ExponentialMovingAverage
    from utils import graph_lib, noise_lib
    
    # Load config
    cfg = load_hydra_config_from_run(root_dir)
    
    # Initialize components
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    
    # Get appropriate model class
    SEDD_class = get_model_class_for_checkpoint(root_dir)
    score_model = SEDD_class(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)
    
    # Try to find checkpoint
    checkpoint_paths = [
        os.path.join(root_dir, "checkpoint.pth"),
        os.path.join(root_dir, "lightning_checkpoint.ckpt"),
        os.path.join(root_dir, "last.ckpt"),
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {root_dir}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if is_original_checkpoint(checkpoint_path):
        # Load original format
        step = load_weights_from_original_checkpoint(score_model, ema, checkpoint_path, device)
    else:
        # Assume Lightning format
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model weights from Lightning state_dict
        model_state = {}
        ema_state = {}
        
        for key, value in checkpoint.get('state_dict', {}).items():
            if key.startswith('score_model.'):
                model_key = key.replace('score_model.', '')
                model_state[model_key] = value
            elif key.startswith('ema.'):
                ema_key = key.replace('ema.', '')
                ema_state[ema_key] = value
        
        # Load states
        if model_state:
            score_model.load_state_dict(model_state, strict=False)
            print("✓ Loaded model weights from Lightning checkpoint")
        
        if ema_state:
            ema.load_state_dict(ema_state)
            print("✓ Loaded EMA weights from Lightning checkpoint")
        
        step = checkpoint.get('global_step', 0)
        print(f"✓ Lightning checkpoint was at step: {step}")
    
    # Apply EMA weights to model
    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    
    return score_model, graph, noise
