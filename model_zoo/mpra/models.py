"""
MPRA Dataset-Specific Models

This module provides MPRA-specific wrappers around the core architectures.
These handle the dataset-specific signal preprocessing and model instantiation.
"""

import torch
from omegaconf import DictConfig
from model.transformer import TransformerModel, create_transformer_model
from model.cnn import ConvolutionalModel, create_convolutional_model


class MPRATransformerModel(TransformerModel):
    """MPRA-specific transformer model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'mpra'
        config.dataset.signal_dim = 3
        config.dataset.num_classes = 3
        config.dataset.sequence_length = 200
        
        super().__init__(config)


class MPRAConvolutionalModel(ConvolutionalModel):
    """MPRA-specific convolutional model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'mpra'
        config.dataset.signal_dim = 3
        config.dataset.num_classes = 3
        config.dataset.sequence_length = 200
        
        super().__init__(config)
    
    def forward(self, indices: torch.Tensor, labels: torch.Tensor, 
                train: bool, sigma: torch.Tensor) -> torch.Tensor:
        """
        MPRA-specific forward pass with label preprocessing.
        
        MPRA uses a similar approach to DeepSTARR but with 3D signals.
        """
        # Convert indices to one-hot
        x = torch.nn.functional.one_hot(indices, num_classes=4).float()
        
        # MPRA-specific label processing for conv architecture
        # Similar to DeepSTARR but with 3D input
        label_emb = torch.nn.Sequential(
            torch.nn.Linear(3, self.config.dataset.sequence_length),
            torch.nn.SiLU(),
            torch.nn.Linear(self.config.dataset.sequence_length, self.config.dataset.sequence_length),
        ).to(labels.device)
        
        processed_labels = torch.unsqueeze(label_emb(labels), dim=2)
        
        # Concatenate sequence and processed labels
        x = torch.cat([x, processed_labels], dim=-1)
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
    Factory function to create MPRA models.
    
    Args:
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        
    Returns:
        Model instance
    """
    if architecture.lower() == 'transformer':
        return MPRATransformerModel(config)
    elif architecture.lower() == 'convolutional':
        return MPRAConvolutionalModel(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Legacy compatibility
def create_mpra_model(config: DictConfig, architecture: str = None):
    """Legacy function for backward compatibility."""
    if architecture is None:
        architecture = config.model.architecture
    return create_model(config, architecture)