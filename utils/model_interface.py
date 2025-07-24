"""
Generic Model Interface for D3-DNA Discrete Diffusion

This module provides a dataset-agnostic interface for loading and working with D3 models.
It replaces the dataset-specific loading logic with a generic approach that works with
any properly configured dataset.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from omegaconf import OmegaConf, DictConfig
from abc import ABC, abstractmethod

from model.ema import ExponentialMovingAverage
from utils import graph_lib, noise_lib


class ModelLoader:
    """
    Generic model loader that works with any dataset configuration.
    
    This class provides a dataset-agnostic interface for loading D3 models,
    replacing the previous dataset-specific factory approach.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the model loader.
        
        Args:
            device: Device to load models on ('cuda', 'cpu', or 'auto')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
    
    def load_model_from_config(self, config: DictConfig, checkpoint_path: str) -> Tuple[torch.nn.Module, Any, Any]:
        """
        Load model from configuration and checkpoint path.
        
        Args:
            config: Configuration object with model and dataset specifications
            checkpoint_path: Explicit path to checkpoint file
            
        Returns:
            Tuple of (model, graph, noise)
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If config is missing required fields
        """
        # Validate inputs
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self._validate_config(config)
        
        # Create model using dataset factory
        model = self._create_model_from_config(config)
        model.to(self.device)
        
        # Create graph and noise components
        graph = graph_lib.get_graph(config, self.device)
        noise = noise_lib.get_noise(config).to(self.device)
        
        # Load checkpoint
        self._load_checkpoint(model, checkpoint_path, config)
        
        return model, graph, noise
    
    def load_model_with_ema(self, config: DictConfig, checkpoint_path: str) -> Tuple[torch.nn.Module, Any, Any, ExponentialMovingAverage]:
        """
        Load model with EMA weights from checkpoint.
        
        Args:
            config: Configuration object
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (model, graph, noise, ema)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self._validate_config(config)
        
        # Create model and components
        model = self._create_model_from_config(config)
        model.to(self.device)
        
        graph = graph_lib.get_graph(config, self.device)
        noise = noise_lib.get_noise(config).to(self.device)
        
        # Create EMA
        ema = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)
        
        # Load checkpoint with EMA
        self._load_checkpoint_with_ema(model, ema, checkpoint_path, config)
        
        # Apply EMA weights to model
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        
        return model, graph, noise, ema
    
    def _create_model_from_config(self, config: DictConfig) -> torch.nn.Module:
        """
        Create model instance from configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            Model instance
        """
        # Get dataset name and architecture from config
        dataset_name = config.dataset.name
        architecture = config.model.architecture
        
        # Import the dataset factory
        from utils.dataset_factory import get_factory
        factory = get_factory()
        
        # Create model using factory
        model = factory.create_model(dataset_name, config, architecture)
        
        return model
    
    def _validate_config(self, config: DictConfig) -> None:
        """
        Validate that configuration has required fields.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = [
            'dataset.name',
            'model.architecture',
            'training.ema'
        ]
        
        for field in required_fields:
            if not OmegaConf.select(config, field):
                raise ValueError(f"Required configuration field missing: {field}")
    
    def _load_checkpoint(self, model: torch.nn.Module, checkpoint_path: str, config: DictConfig) -> None:
        """
        Load checkpoint weights into model.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
            config: Configuration object
        """
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if checkpoint_path.endswith('.ckpt'):
            # Lightning checkpoint format
            if 'state_dict' in checkpoint:
                state_dict = self._extract_model_state_from_lightning(checkpoint['state_dict'])
            else:
                state_dict = checkpoint
        elif 'model' in checkpoint:
            # Original D3 checkpoint format
            state_dict = checkpoint['model']
        elif 'pytorch_model.bin' in checkpoint_path or checkpoint_path.endswith('.bin'):
            # HuggingFace format
            state_dict = checkpoint
        else:
            # Assume direct state dict
            state_dict = checkpoint
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
        print("✓ Model weights loaded successfully")
    
    def _load_checkpoint_with_ema(self, model: torch.nn.Module, ema: ExponentialMovingAverage, 
                                 checkpoint_path: str, config: DictConfig) -> None:
        """
        Load checkpoint with EMA weights.
        
        Args:
            model: Model to load weights into
            ema: EMA object to load weights into
            checkpoint_path: Path to checkpoint file
            config: Configuration object
        """
        print(f"Loading checkpoint with EMA: {checkpoint_path}")
        
        if checkpoint_path.endswith('.pth') and self._is_original_checkpoint(checkpoint_path):
            # Original D3 checkpoint format
            from utils.checkpoint_utils import load_weights_from_original_checkpoint
            step = load_weights_from_original_checkpoint(model, ema, checkpoint_path, self.device)
            print(f"✓ Original checkpoint loaded at step: {step}")
        else:
            # Lightning or other format
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                # Lightning format
                model_state, ema_state = self._extract_states_from_lightning(checkpoint['state_dict'])
                
                if model_state:
                    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                    if missing_keys:
                        print(f"Warning: Missing keys in model state: {missing_keys}")
                    if unexpected_keys:
                        print(f"Warning: Unexpected keys in model state: {unexpected_keys}")
                    print("✓ Model weights loaded from Lightning checkpoint")
                
                if ema_state:
                    try:
                        ema.load_state_dict(ema_state)
                        print("✓ EMA weights loaded from Lightning checkpoint")
                    except Exception as e:
                        print(f"Warning: Could not load EMA state: {e}")
                
                step = checkpoint.get('global_step', 0)
                print(f"✓ Lightning checkpoint was at step: {step}")
            else:
                # Direct state dict or other format
                if 'model' in checkpoint and 'ema' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                    ema.load_state_dict(checkpoint['ema'])
                    print("✓ Model and EMA weights loaded")
                else:
                    # Fall back to model-only loading
                    print("Warning: EMA weights not found, loading model weights only")
                    self._load_checkpoint(model, checkpoint_path, config)
    
    def _extract_model_state_from_lightning(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract model state from Lightning checkpoint state_dict."""
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith('score_model.'):
                model_key = key.replace('score_model.', '')
                model_state[model_key] = value
            elif key.startswith('model.'):
                model_key = key.replace('model.', '')
                model_state[model_key] = value
        return model_state
    
    def _extract_states_from_lightning(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Extract both model and EMA states from Lightning checkpoint."""
        model_state = {}
        ema_state = {}
        
        for key, value in state_dict.items():
            if key.startswith('score_model.'):
                model_key = key.replace('score_model.', '')
                model_state[model_key] = value
            elif key.startswith('model.'):
                model_key = key.replace('model.', '')
                model_state[model_key] = value
            elif key.startswith('ema.'):
                ema_key = key.replace('ema.', '')
                ema_state[ema_key] = value
        
        return model_state, ema_state
    
    def _is_original_checkpoint(self, checkpoint_path: str) -> bool:
        """Check if checkpoint is in original D3 format."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return 'model' in checkpoint and 'ema' in checkpoint and 'step' in checkpoint
        except:
            return False


def load_model_from_config_and_checkpoint(config: Union[str, DictConfig], checkpoint_path: str, 
                                        device: str = 'auto') -> Tuple[torch.nn.Module, Any, Any]:
    """
    Convenience function to load model from config file/object and checkpoint.
    
    Args:
        config: Configuration file path or DictConfig object
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Tuple of (model, graph, noise)
    """
    if isinstance(config, str):
        config = OmegaConf.load(config)
    
    loader = ModelLoader(device)
    return loader.load_model_from_config(config, checkpoint_path)


def load_model_with_ema_from_config_and_checkpoint(config: Union[str, DictConfig], checkpoint_path: str, 
                                                  device: str = 'auto') -> Tuple[torch.nn.Module, Any, Any, ExponentialMovingAverage]:
    """
    Convenience function to load model with EMA from config and checkpoint.
    
    Args:
        config: Configuration file path or DictConfig object
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Tuple of (model, graph, noise, ema)
    """
    if isinstance(config, str):
        config = OmegaConf.load(config)
    
    loader = ModelLoader(device)
    return loader.load_model_with_ema(config, checkpoint_path)


def load_lightning_module_from_checkpoint(checkpoint_path: str, device: str = 'auto'):
    """
    Load Lightning module directly from checkpoint, inferring the correct module type from config.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint
        device: Device to load on
        
    Returns:
        Lightning module instance
        
    Raises:
        FileNotFoundError: If checkpoint not found
        ValueError: If config not found or dataset not supported
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint to extract config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'hyper_parameters' not in checkpoint or 'cfg' not in checkpoint['hyper_parameters']:
        raise ValueError(f"Config not found in Lightning checkpoint: {checkpoint_path}")
    
    cfg = checkpoint['hyper_parameters']['cfg']
    dataset_name = cfg.dataset.name.lower()
    
    # Import the appropriate Lightning module class based on dataset
    if dataset_name == 'promoter':
        try:
            # Try to import from dataset-specific location
            import sys
            import importlib.util
            
            module_path = Path(__file__).parent.parent / 'model_zoo' / 'promoter' / 'train.py'
            if module_path.exists():
                spec = importlib.util.spec_from_file_location("promoter_train", module_path)
                promoter_train = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(promoter_train)
                lightning_model = promoter_train.PromoterD3LightningModule.load_from_checkpoint(checkpoint_path)
            else:
                raise ImportError("Promoter Lightning module not found")
        except ImportError:
            raise ValueError(f"Could not load Promoter Lightning module from {checkpoint_path}")
    
    elif dataset_name == 'mpra':
        try:
            import sys
            import importlib.util
            
            module_path = Path(__file__).parent.parent / 'model_zoo' / 'mpra' / 'train.py'
            if module_path.exists():
                spec = importlib.util.spec_from_file_location("mpra_train", module_path)
                mpra_train = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mpra_train)
                lightning_model = mpra_train.MPRAD3LightningModule.load_from_checkpoint(checkpoint_path)
            else:
                raise ImportError("MPRA Lightning module not found")
        except ImportError:
            raise ValueError(f"Could not load MPRA Lightning module from {checkpoint_path}")
    
    elif dataset_name == 'deepstarr':
        try:
            import sys
            import importlib.util
            
            module_path = Path(__file__).parent.parent / 'model_zoo' / 'deepstarr' / 'train.py'
            if module_path.exists():
                spec = importlib.util.spec_from_file_location("deepstarr_train", module_path)
                deepstarr_train = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(deepstarr_train)
                lightning_model = deepstarr_train.D3LightningModule.load_from_checkpoint(checkpoint_path)
            else:
                raise ImportError("DeepSTARR Lightning module not found")
        except ImportError:
            raise ValueError(f"Could not load DeepSTARR Lightning module from {checkpoint_path}")
    
    else:
        raise ValueError(f"Unsupported dataset for Lightning loading: {dataset_name}")
    
    lightning_model = lightning_model.to(device)
    lightning_model.eval()
    
    return lightning_model