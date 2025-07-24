#!/usr/bin/env python3
"""
MPRA Sampling Script

This script provides sampling functionality specifically for the MPRA dataset,
inheriting from the base sampling classes and implementing MPRA-specific
model creation and label generation.
"""

import os
import sys
from pathlib import Path

# Package imports

from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.mpra.models import create_model
from omegaconf import OmegaConf
import torch


class MPRASampler(BaseSampler):
    """Sampler specifically for MPRA dataset."""
    
    def __init__(self):
        super().__init__('mpra')
        
    def create_model(self, config, architecture):
        """Create MPRA-specific model."""
        return create_model(config, architecture)
        
    def get_sequence_length(self, config):
        """Get MPRA sequence length."""
        # MPRA uses 200 bp sequences
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 200
    
    def generate_labels(self, num_samples, config):
        """
        Generate conditioning labels for MPRA sampling.
        
        MPRA typically has 3 labels for different regulatory activities.
        """
        # Generate random labels in the range typical for MPRA targets
        # You might want to adjust these ranges based on your specific needs
        labels = torch.randn(num_samples, 3, device=self.device) * 1.5  # Adjust scale as needed
        
        return labels
    
    def sample_with_specific_labels(self, checkpoint_path, config, architecture,
                                   regulatory_activities=None, **kwargs):
        """
        Sample sequences with specific regulatory activity targets.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object
            architecture: Architecture name
            regulatory_activities: Target regulatory activities (list of 3 values or list of lists)
            **kwargs: Additional sampling arguments
        """
        # Load model
        model = self.load_model_from_checkpoint(checkpoint_path, config, architecture)
        
        num_samples = kwargs.get('num_samples', 1000)
        
        # Create specific labels if provided
        if regulatory_activities is not None:
            if isinstance(regulatory_activities[0], (int, float)):
                # Single set of activities, repeat for all samples
                regulatory_activities = [regulatory_activities] * num_samples
                
            # Create label tensor
            labels = torch.tensor(
                regulatory_activities,
                device=self.device, 
                dtype=torch.float32
            )
            
            # Override the generate_labels method for this call
            original_generate_labels = self.generate_labels
            self.generate_labels = lambda n, cfg: labels
            
            try:
                sequences = self.sample_sequences(model, config, num_samples, **kwargs)
            finally:
                # Restore original method
                self.generate_labels = original_generate_labels
        else:
            sequences = self.sample_sequences(model, config, num_samples, **kwargs)
        
        return sequences


def load_config(architecture):
    """Load MPRA configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function."""
    parser = parse_base_args()
    parser.description = 'MPRA Sampling Script'
    
    # Add MPRA-specific arguments
    parser.add_argument('--regulatory_activities', nargs=3, type=float, 
                       help='Target regulatory activities (3 values)')
    
    args = parser.parse_args()
    
    # Create sampler
    sampler = MPRASampler()
    
    # Load config if not provided
    if not args.config:
        try:
            config = load_config(args.architecture)
        except FileNotFoundError:
            print(f"Error: No config provided and default config not found for architecture: {args.architecture}")
            return 1
    else:
        config = OmegaConf.load(args.config)
    
    # Use specific label sampling if requested
    if args.regulatory_activities is not None:
        sequences = sampler.sample_with_specific_labels(
            checkpoint_path=args.checkpoint,
            config=config,
            architecture=args.architecture,
            regulatory_activities=args.regulatory_activities,
            num_samples=args.num_samples,
            method=args.method,
            num_steps=args.num_steps,
            eta=args.eta,
            temperature=args.temperature
        )
        
        # Save sequences
        output_path = args.output or f"samples/mpra_{args.architecture}_{args.method}_specific_samples.{args.format}"
        sampler.save_sequences(sequences, output_path, args.format)
        
        print(f"Sampling completed with specific labels. {args.num_samples} sequences generated.")
    else:
        # Use standard sampling
        sequences = sampler.sample(
            checkpoint_path=args.checkpoint,
            config=config,
            architecture=args.architecture,
            num_samples=args.num_samples,
            method=args.method,
            num_steps=args.num_steps,
            output_path=args.output,
            format=args.format,
            eta=args.eta,
            temperature=args.temperature
        )
        
        print(f"Sampling completed. {args.num_samples} sequences generated.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())