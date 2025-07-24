"""
Promoter Dataset-Specific Models

This module provides Promoter-specific wrappers around the core architectures.
These handle the dataset-specific signal preprocessing and model instantiation.
"""

import torch
from omegaconf import DictConfig
from model.transformer import TransformerModel, create_transformer_model
from model.cnn import ConvolutionalModel, create_convolutional_model


class PromoterTransformerModel(TransformerModel):
    """Promoter-specific transformer model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'promoter'
        config.dataset.signal_dim = 1
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 1024
        
        super().__init__(config)


class PromoterConvolutionalModel(ConvolutionalModel):
    """Promoter-specific convolutional model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'promoter'
        config.dataset.signal_dim = 1
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 1024
        
        super().__init__(config)
    
    def forward(self, indices: torch.Tensor, labels: torch.Tensor, 
                train: bool, sigma: torch.Tensor) -> torch.Tensor:
        """
        Promoter-specific forward pass with label preprocessing.
        
        Promoter uses labels directly without additional embedding.
        """
        # Convert indices to one-hot
        x = torch.nn.functional.one_hot(indices, num_classes=4).float()
        
        # Promoter uses labels directly - no additional processing needed
        # Concatenate sequence and labels directly
        x = torch.cat([x, labels], dim=-1)
        x = x.permute(0, 2, 1)
        
        # Apply initial convolution
        out = self.act(self.linear(x))
        
        # Time conditioning
        c = torch.nn.functional.silu(self.sigma_map(sigma))
        
        # Apply conv blocks
        for block, dense, norm in zip(self.conv_blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(c)[:, :, None])))
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
    Factory function to create Promoter models.
    
    Args:
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        
    Returns:
        Model instance
    """
    if architecture.lower() == 'transformer':
        return PromoterTransformerModel(config)
    elif architecture.lower() == 'convolutional':
        return PromoterConvolutionalModel(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Legacy compatibility
def create_promoter_model(config: DictConfig, architecture: str = None):
    """Legacy function for backward compatibility."""
    if architecture is None:
        architecture = config.model.architecture
    return create_model(config, architecture)