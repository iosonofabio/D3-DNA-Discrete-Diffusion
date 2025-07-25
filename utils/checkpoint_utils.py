"""
Utilities for handling checkpoint compatibility between original D3 format and PyTorch Lightning format.
Now provides dataset-agnostic checkpoint loading functions.
"""

import os
import torch
from typing import Dict, Any, Optional


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


def create_model_from_config(cfg, device: str = 'cpu'):
    """Create model from configuration using base models only."""
    architecture = getattr(cfg.model, 'architecture', 'transformer').lower()
    
    if architecture == 'transformer':
        from model.transformer import TransformerModel
        model = TransformerModel(cfg)
    elif architecture == 'convolutional':
        from model.cnn import ConvolutionalModel
        model = ConvolutionalModel(cfg)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: 'transformer', 'convolutional'")
    
    return model.to(device)


def load_model_from_checkpoint_path(config, checkpoint_path: str, device: str = 'cpu'):
    """Load model from explicit checkpoint path with config."""
    from model.ema import ExponentialMovingAverage
    from utils import graph_lib, noise_lib
    
    # Initialize components
    graph = graph_lib.get_graph(config, device)
    noise = noise_lib.get_noise(config).to(device)
    
    # Create model
    score_model = create_model_from_config(config, device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.training.ema)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
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
