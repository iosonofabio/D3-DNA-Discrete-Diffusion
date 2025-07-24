import os
import torch
from pathlib import Path
from typing import Tuple, Any, Union, Optional
from omegaconf import OmegaConf, DictConfig

from utils.utils import load_hydra_config_from_run
from utils.model_interface import ModelLoader
from model.ema import ExponentialMovingAverage
from utils import graph_lib
from utils import noise_lib

def load_model_hf(dir: str, device: str) -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load model from HuggingFace-style directory with config and weights.
    
    Args:
        dir: Directory containing config.yaml and pytorch_model.bin
        device: Device to load model on
        
    Returns:
        Tuple of (model, graph, noise)
        
    Raises:
        FileNotFoundError: If config.yaml is not found
        ValueError: If config is invalid
    """
    print(f"Loading model from HF directory: {dir}")
    
    # Load config (required)
    config_path = os.path.join(dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {dir}")
    
    config = OmegaConf.load(config_path)
    
    # Find model weights file
    model_path = os.path.join(dir, 'pytorch_model.bin')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"pytorch_model.bin not found in {dir}")
    
    # Use generic model loader
    loader = ModelLoader(device)
    return loader.load_model_from_config(config, model_path)


def load_model_local(root_dir: str, device: str, checkpoint_path: Optional[str] = None) -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load model from local directory with Hydra config.
    
    Args:
        root_dir: Directory containing Hydra config
        device: Device to load model on
        checkpoint_path: Explicit path to checkpoint file. If None, searches for common checkpoint files.
        
    Returns:
        Tuple of (model, graph, noise)
        
    Raises:
        FileNotFoundError: If config or checkpoint not found
    """
    print(f"Loading model from local directory: {root_dir}")
    
    # Load Hydra config
    cfg = load_hydra_config_from_run(root_dir)
    
    # Find checkpoint if not explicitly provided
    if checkpoint_path is None:
        checkpoint_path = find_checkpoint_in_directory(root_dir)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Use generic model loader with EMA
    loader = ModelLoader(device)
    model, graph, noise, ema = loader.load_model_with_ema(cfg, checkpoint_path)
    
    return model, graph, noise


def load_model_from_lightning(checkpoint_path: str, device: str) -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load model directly from Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, graph, noise)
        
    Raises:
        FileNotFoundError: If checkpoint not found
        ValueError: If config not found in checkpoint
    """
    print(f"Loading model from Lightning checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint to extract config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config from hyperparameters
    if 'hyper_parameters' not in checkpoint or 'cfg' not in checkpoint['hyper_parameters']:
        raise ValueError(f"Config not found in Lightning checkpoint: {checkpoint_path}")
    
    cfg = checkpoint['hyper_parameters']['cfg']
    
    # Use generic model loader
    loader = ModelLoader(device)
    model, graph, noise, ema = loader.load_model_with_ema(cfg, checkpoint_path)
    
    return model, graph, noise


def find_checkpoint_in_directory(root_dir: str) -> str:
    """
    Find checkpoint file in directory using common patterns.
    
    Args:
        root_dir: Directory to search in
        
    Returns:
        Path to found checkpoint
        
    Raises:
        FileNotFoundError: If no checkpoint found
    """
    # Try common checkpoint locations (in order of preference)
    checkpoint_patterns = [
        "checkpoint.pth",  # Original D3 format
        "pytorch_model.bin",  # HuggingFace format
        "model.ckpt",  # Common Lightning format
        "last.ckpt",  # Lightning last checkpoint
        "lightning_checkpoints/last.ckpt",  # Lightning in subdirectory
        "best.ckpt",  # Lightning best checkpoint
    ]
    
    for pattern in checkpoint_patterns:
        checkpoint_path = os.path.join(root_dir, pattern)
        if os.path.exists(checkpoint_path):
            return checkpoint_path
    
    raise FileNotFoundError(f"No checkpoint found in {root_dir}. Tried: {checkpoint_patterns}")


def load_model_from_config_and_checkpoint(config: Union[str, DictConfig], checkpoint_path: str, 
                                        device: str = 'auto') -> Tuple[torch.nn.Module, Any, Any]:
    """
    Generic model loading function that works with explicit config and checkpoint.
    
    Args:
        config: Configuration file path or DictConfig object
        checkpoint_path: Explicit path to checkpoint file
        device: Device to load on ('cuda', 'cpu', or 'auto')
        
    Returns:
        Tuple of (model, graph, noise)
    """
    from utils.model_interface import load_model_from_config_and_checkpoint
    return load_model_from_config_and_checkpoint(config, checkpoint_path, device)


# Legacy function with new implementation
def load_model(root_dir: str, device: str) -> Tuple[torch.nn.Module, Any, Any]:
    """
    Legacy function for backward compatibility.
    Now uses HuggingFace-style loading as default.
    
    Args:
        root_dir: Directory containing model files
        device: Device to load on
        
    Returns:
        Tuple of (model, graph, noise)
    """
    return load_model_hf(root_dir, device)
