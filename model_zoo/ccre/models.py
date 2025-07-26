"""
cCRE Dataset-Specific Models

This module provides cCRE-specific wrappers around the core architectures.
These handle the dataset-specific preprocessing and model instantiation for
unlabeled 512bp sequences.
"""

import os
import torch
from omegaconf import DictConfig
from typing import Tuple, Any
from model.transformer import TransformerModel, create_transformer_model
from model.cnn import ConvolutionalModel, create_convolutional_model


class cCREVocabEmbedding(torch.nn.Module):
    """Vocabulary embedding without signal conditioning for cCRE."""
    
    def __init__(self, vocab_dim: int, dim: int):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=torch.sqrt(torch.tensor(5.0)))
    
    def forward(self, x, y):
        # Only return vocabulary embedding, ignore y (labels)
        return self.embedding[x]


class cCRETransformerModel(TransformerModel):
    """cCRE-specific transformer model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'ccre'
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 512
        config.dataset.signal_dim = 0  # No labels for unlabeled data
        
        super().__init__(config)
        
        # Replace the vocab_embed with our custom one that doesn't need signals
        vocab_size = config.dataset.num_classes
        self.vocab_embed = cCREVocabEmbedding(
            vocab_dim=vocab_size,
            dim=config.model.hidden_size
        )
    
    def forward(self, indices: torch.Tensor, labels: torch.Tensor, 
                train: bool, sigma: torch.Tensor) -> torch.Tensor:
        """
        cCRE-specific forward pass for unlabeled data.
        
        Completely ignores labels and does unconditional generation.
        """
        # Can now call super with any labels since our vocab_embed ignores them
        return super().forward(indices, labels, train, sigma)


class cCREConvolutionalModel(ConvolutionalModel):
    """cCRE-specific convolutional model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'ccre'
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 512
        config.dataset.signal_dim = 0  # No labels for unlabeled data
        
        super().__init__(config)
    
    def forward(self, indices: torch.Tensor, labels: torch.Tensor, 
                train: bool, sigma: torch.Tensor) -> torch.Tensor:
        """
        cCRE-specific forward pass for unlabeled data.
        
        Completely ignores labels and does unconditional generation.
        """
        # Create dummy labels with shape (batch_size, 0) for unconditional generation
        batch_size = indices.shape[0]
        dummy_labels = torch.zeros(batch_size, 0, device=indices.device, dtype=torch.float32)
        return super().forward(indices, dummy_labels, train, sigma)


def create_model(config: DictConfig, architecture: str):
    """
    Factory function to create cCRE models.
    
    Args:
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        
    Returns:
        Model instance
    """
    if architecture.lower() == 'transformer':
        return cCRETransformerModel(config)
    elif architecture.lower() == 'convolutional':
        return cCREConvolutionalModel(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_trained_model(checkpoint_path: str, config: DictConfig, architecture: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load trained cCRE model from specific checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        device: Device to load model on
        
    Returns:
        Tuple of (model, graph, noise) ready for sampling/evaluation
    """
    # Import required components
    from model.ema import ExponentialMovingAverage
    from utils import graph_lib, noise_lib
    
    print(f"Loading cCRE {architecture} model from {checkpoint_path}")
    
    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    if architecture.lower() == 'transformer':
        model = cCRETransformerModel(config)
    elif architecture.lower() == 'convolutional':
        model = cCREConvolutionalModel(config)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    model.to(device)
    
    # Initialize graph and noise
    graph = graph_lib.get_graph(config, device)
    noise = noise_lib.get_noise(config).to(device)
    
    # Initialize EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if checkpoint_path.endswith('.ckpt'):
        # Lightning checkpoint
        if 'state_dict' in checkpoint:
            model_state = {}
            ema_state = {}
            
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('score_model.'):
                    model_key = key.replace('score_model.', '')
                    model_state[model_key] = value
                elif key.startswith('model.'):
                    model_key = key.replace('model.', '')
                    model_state[model_key] = value
                elif key.startswith('ema.'):
                    ema_key = key.replace('ema.', '')
                    ema_state[ema_key] = value
            
            # Load model weights
            if model_state:
                model.load_state_dict(model_state, strict=False)
                print("✓ Loaded model weights from Lightning checkpoint")
            
            # Load EMA weights
            if ema_state:
                ema.load_state_dict(ema_state)
                print("✓ Loaded EMA weights from Lightning checkpoint")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("✓ Loaded model weights from Lightning checkpoint")
    else:
        # Original D3 format
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            print("✓ Loaded model weights from original checkpoint")
        
        if 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])
            print("✓ Loaded EMA weights from original checkpoint")
    
    # Apply EMA weights to model
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    
    print(f"✓ cCRE {architecture} model loaded successfully")
    
    return model, graph, noise


# Legacy compatibility
def create_ccre_model(config: DictConfig, architecture: str = None):
    """Legacy function for backward compatibility."""
    if architecture is None:
        architecture = config.model.architecture
    return create_model(config, architecture)