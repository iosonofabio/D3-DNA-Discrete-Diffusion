import os
import torch
from model import SEDD
from utils.utils import load_hydra_config_from_run
from model.ema import ExponentialMovingAverage
from utils import graph_lib
from utils import noise_lib

from omegaconf import OmegaConf

def load_model_hf(dir, device):
    print (dir)
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device):
    """Enhanced version that handles both original .pth and Lightning .ckpt formats."""
    from utils.checkpoint_utils import (
        is_original_checkpoint, 
        load_weights_from_original_checkpoint,
        get_model_class_for_checkpoint
    )
    
    cfg = load_hydra_config_from_run(root_dir)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    
    # Get appropriate model class based on dataset
    SEDD_class = get_model_class_for_checkpoint(root_dir)
    score_model = SEDD_class(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    # Try to find checkpoint in multiple locations
    checkpoint_paths = [
        os.path.join(root_dir, "checkpoint.pth"),
        os.path.join(root_dir, "lightning_checkpoints", "last.ckpt"),
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
        # Handle Lightning format
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


def load_model_from_lightning(checkpoint_path, device):
    """Load model specifically from Lightning checkpoint."""
    # Try to determine dataset from checkpoint or path
    dataset_name = None
    if 'promoter' in checkpoint_path.lower():
        dataset_name = 'promoter'
    elif 'mpra' in checkpoint_path.lower():
        dataset_name = 'mpra'
    elif 'deepstarr' in checkpoint_path.lower():
        dataset_name = 'deepstarr'
    
    # Load checkpoint to get config and determine correct module type
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get dataset from hyperparameters
    if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
        cfg = checkpoint['hyper_parameters']['cfg']
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'train'):
            dataset_name = cfg.data.train
    
    # Import appropriate Lightning module class
    if dataset_name == 'promoter':
        from scripts.lightning_trainer import PromoterD3LightningModule
        lightning_model = PromoterD3LightningModule.load_from_checkpoint(checkpoint_path)
    elif dataset_name == 'mpra':
        from scripts.lightning_trainer import MPRAD3LightningModule
        lightning_model = MPRAD3LightningModule.load_from_checkpoint(checkpoint_path)
    else:
        from scripts.lightning_trainer import D3LightningModule
        lightning_model = D3LightningModule.load_from_checkpoint(checkpoint_path)
    
    lightning_model = lightning_model.to(device)
    lightning_model.eval()
    
    # Extract components
    score_model = lightning_model.score_model
    graph = lightning_model.graph
    noise = lightning_model.noise
    
    # Apply EMA weights
    lightning_model.ema.store(score_model.parameters())
    lightning_model.ema.copy_to(score_model.parameters())
    
    return score_model, graph, noise


def load_model(root_dir, device):
    return load_model_hf(root_dir, device)
