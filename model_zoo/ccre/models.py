"""
cCRE Dataset-Specific Models

This module provides cCRE-specific models that completely remove all label conditioning
for true unconditional generation on unlabeled 512bp sequences.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, Any, Optional
from einops import rearrange

from model import rotary
from model.layers import LayerNorm, get_bias_dropout_scale


class cCREVocabEmbedding(nn.Module):
    """Simple vocabulary embedding for unconditional generation."""
    
    def __init__(self, vocab_dim: int, dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
    
    def forward(self, x):
        return self.embedding[x]


class cCRETransformerBlock(nn.Module):
    """Transformer block without conditioning for cCRE."""
    
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout

        # Attention layers
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward layers
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

    def _get_bias_dropout_scale(self):
        return get_bias_dropout_scale()(self.training)

    def forward(self, x: torch.Tensor, rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Self-attention with rotary position encoding
        bias, dropout_p, scale = self._get_bias_dropout_scale()
        
        # Pre-norm for attention
        x_norm = self.norm1(x)
        qkv = self.attn_qkv(x_norm)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        # Apply rotary position encoding
        cos, sin = rotary_cos_sin
        qkv = rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
        
        # Flash attention or regular attention
        try:
            from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
            attn_output = flash_attn_qkvpacked_func(qkv, dropout_p=dropout_p if self.training else 0.0)
            attn_output = rearrange(attn_output, 'b s h d -> b s (h d)')
        except ImportError:
            # Fallback to regular attention
            q, k, v = qkv.unbind(dim=2)
            q, k, v = map(lambda t: rearrange(t, 'b s h d -> b h s d'), (q, k, v))
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p if self.training else 0.0)
            attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        
        attn_output = self.attn_out(attn_output)
        attn_output = self.dropout1(attn_output)
        
        # Residual connection
        x = x + attn_output
        
        # Pre-norm for MLP
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        mlp_output = self.dropout2(mlp_output)
        
        # Residual connection
        x = x + mlp_output
        
        return x


class cCRETransformerModel(nn.Module):
    """Unconditional transformer model for cCRE sequences."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        if isinstance(config, dict):
            config = OmegaConf.create(config)
            
        # Model configuration
        vocab_size = config.dataset.num_classes
        
        # Core components (no conditioning)
        self.vocab_embed = cCREVocabEmbedding(vocab_size, config.model.hidden_size)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)
        
        # Transformer blocks (no conditioning)
        self.blocks = nn.ModuleList([
            cCRETransformerBlock(
                dim=config.model.hidden_size, 
                n_heads=config.model.n_heads, 
                dropout=config.model.dropout
            ) 
            for _ in range(config.model.n_blocks)
        ])
        
        # Output layer (no conditioning)
        self.norm_final = LayerNorm(config.model.hidden_size)
        self.output_layer = nn.Linear(config.model.hidden_size, vocab_size)
        self.output_layer.weight.data.zero_()
        self.output_layer.bias.data.zero_()

    def forward(self, indices: torch.Tensor, labels: torch.Tensor, 
                train: bool, sigma: torch.Tensor) -> torch.Tensor:
        """Forward pass ignoring all conditioning."""
        # Embedding (no labels)
        x = self.vocab_embed(indices)
        
        # Rotary position encoding
        rotary_cos_sin = self.rotary_emb(x)

        # Forward through transformer blocks (no conditioning)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin)
            
            x = self.norm_final(x)
            x = self.output_layer(x)

        # Mask out the input tokens
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        
        return x


class cCREConvolutionalModel(nn.Module):
    """Unconditional convolutional model for cCRE sequences."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        if isinstance(config, dict):
            config = OmegaConf.create(config)
            
        # Model configuration
        vocab_size = config.dataset.num_classes
        conv_channels = getattr(config.model, 'conv_channels', 256)
        
        # Input layer
        self.linear = nn.Conv1d(vocab_size, conv_channels, 1)
        self.act = nn.SiLU()
        
        # Conv blocks (no conditioning)
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(conv_channels, conv_channels, 3, padding=1)
            for _ in range(config.model.n_blocks)
        ])
        
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, conv_channels)
            for _ in range(config.model.n_blocks)
        ])
        
        # Output layer
        self.final = nn.Conv1d(conv_channels, vocab_size, 1)
        self.final.weight.data.zero_()
        self.final.bias.data.zero_()

    def forward(self, indices: torch.Tensor, labels: torch.Tensor, 
                train: bool, sigma: torch.Tensor) -> torch.Tensor:
        """Forward pass ignoring all conditioning."""
        # Convert indices to one-hot
        x = F.one_hot(indices, num_classes=4).float()
        x = x.permute(0, 2, 1)  # (batch_size, 4, seq_length)
        
        # Apply initial convolution
        out = self.act(self.linear(x))
        
        # Apply conv blocks (no conditioning)
        for block, norm in zip(self.conv_blocks, self.norms):
            h = self.act(block(norm(out)))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h
        
        # Final output
        x = self.final(out)
        x = x.permute(0, 2, 1)
        
        return x


def create_model(config: DictConfig, architecture: str):
    """
    Factory function to create cCRE models.
    
    Args:
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        
    Returns:
        Model instance
    """
    # Ensure dataset-specific config is set
    if not hasattr(config, 'dataset'):
        config.dataset = {}
    config.dataset.name = 'ccre'
    config.dataset.num_classes = 4
    config.dataset.sequence_length = 512
    config.dataset.signal_dim = 0  # No labels for unlabeled data
    
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
    model = create_model(config, architecture)
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