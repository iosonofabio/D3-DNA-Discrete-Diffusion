#!/usr/bin/env python3
"""
DeepSTARR Sampling Script

Inherits from base sampling framework while using DeepSTARR-specific models directly.
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

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and DeepSTARR-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.deepstarr.models import DeepSTARRTransformerModel, DeepSTARRConvolutionalModel
from model_zoo.deepstarr.data import get_deepstarr_datasets
from model_zoo.deepstarr.deepstarr import PL_DeepSTARR


class DeepSTARRSampler(BaseSampler):
    """DeepSTARR-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("DeepSTARR")
    
    def create_model(self, config: OmegaConf, architecture: str):
        """Create DeepSTARR-specific model."""
        if architecture.lower() == 'transformer':
            return DeepSTARRTransformerModel(config)
        elif architecture.lower() == 'convolutional':
            return DeepSTARRConvolutionalModel(config)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get DeepSTARR sequence length."""
        return 249  # DeepSTARR fixed sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        """Generate conditioning labels for DeepSTARR sampling."""
        # DeepSTARR has 2 activities: Dev and HK enhancer activities
        # Generate random activities in a reasonable range
        labels = torch.randn(num_samples, 2, device=self.device)
        return labels


def load_config(architecture: str):
    """Load DeepSTARR configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    # Add DeepSTARR-specific conditioning arguments
    parser.add_argument('--dev_activity', type=float, help='Dev enhancer activity value (if not provided, uses random)')
    parser.add_argument('--hk_activity', type=float, help='HK enhancer activity value (if not provided, uses random)')
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
    sampler = DeepSTARRSampler()
    
    # Generate conditioning labels based on arguments
    conditioning_labels = None
    if not args.unconditional:
        if args.dev_activity is not None and args.hk_activity is not None:
            # User-specified activities
            conditioning_labels = torch.tensor([[args.dev_activity, args.hk_activity]], device=sampler.device).expand(args.num_samples, -1)
            print(f"Using specified activities: Dev={args.dev_activity}, HK={args.hk_activity}")
        else:
            # Random activities (default behavior)
            conditioning_labels = sampler.generate_conditioning_labels(args.num_samples, config)
            print("Using random activities")
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
    print(f"\nDeepSTARR Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print(f"\n✓ DeepSTARR sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())