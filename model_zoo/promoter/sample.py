#!/usr/bin/env python3
"""
Promoter Sampling Script

Inherits from base sampling framework while using Promoter-specific models directly.
Uses proper PC sampling methodology.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Optional
import numpy as np

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and Promoter-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.promoter.models import create_model
from model_zoo.promoter.data import get_promoter_datasets


class PromoterSampler(BaseSampler):
    """Promoter-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("Promoter")
    
    def create_model(self, config: OmegaConf, architecture: str):
        """Create Promoter-specific model."""
        return create_model(config, architecture)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get Promoter sequence length."""
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 1024  # Promoter default sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        """Generate conditioning labels for Promoter sampling."""
        # Promoter dataset has expression targets
        # For sampling, we generate random target expression values
        # This should match the expected conditioning format for Promoter models
        
        seq_length = self.get_sequence_length(config)
        
        # Check if the model expects per-position targets or global targets
        # This may need adjustment based on the specific Promoter model configuration
        if hasattr(config, 'model') and hasattr(config.model, 'target_dim'):
            target_dim = config.model.target_dim
        else:
            target_dim = 1  # Default assumption
        
        # Generate random expression targets
        if target_dim == 1:
            # Global target for the entire sequence
            labels = torch.randn(num_samples, target_dim, device=self.device) * 2.0
        else:
            # Per-position targets (less common but possible)
            labels = torch.randn(num_samples, seq_length, target_dim, device=self.device) * 2.0
            
        return labels


def load_config(architecture: str):
    """Load Promoter configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    # Add Promoter-specific conditioning arguments
    parser.add_argument('--expression_target', type=float, help='Expression target value (if not provided, uses random)')
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
    args = parser.parse_args()
    
    # Load config if not provided
    if not args.config:
        try:
            config_path = Path(__file__).parent / 'configs' / 'transformer.yaml'  # Default to transformer
            if config_path.exists():
                args.config = str(config_path)
                print(f"Using default config: {args.config}")
            else:
                print(f"Error: No config provided and default config not found: {config_path}")
                print("Please provide a config file with --config")
                return 1
        except Exception as e:
            print(f"Error loading default config: {e}")
            return 1
    
    config = OmegaConf.load(args.config)
    sampler = PromoterSampler()
    
    # Generate conditioning labels based on arguments
    conditioning_labels = None
    if not args.unconditional:
        if args.expression_target is not None:
            # User-specified expression target
            conditioning_labels = torch.tensor([[args.expression_target]], device=sampler.device).expand(args.num_samples, -1)
            print(f"Using specified expression target: {args.expression_target}")
        else:
            # Random expression targets (default behavior)
            conditioning_labels = sampler.generate_conditioning_labels(args.num_samples, config)
            print("Using random expression targets")
    else:
        print("Sampling unconditionally (no conditioning labels)")
    
    # Set default steps to sequence length if not provided
    steps = args.steps
    if steps is None:
        steps = sampler.get_sequence_length(config)
        print(f"Using default steps: {steps} (sequence length)")
    
    # Run sampling only (no evaluation)
    results = sampler.sample_and_save(
        model_path=args.model_path,
        config=config,
        num_samples=args.num_samples,
        steps=steps,
        conditioning_labels=conditioning_labels,
        output_path=args.output,
        format=args.format
    )
    
    # Print results
    print(f"\nPromoter Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print(f"\n✓ Promoter sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())