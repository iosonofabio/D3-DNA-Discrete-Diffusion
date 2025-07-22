#!/usr/bin/env python3
"""
Promoter Sampling Script

This script provides sampling functionality specifically for the Promoter dataset,
inheriting from the base sampling classes and implementing Promoter-specific
model creation and label generation.
"""

import os
import sys
from pathlib import Path

# Package imports

from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.promoter.models import create_model
from omegaconf import OmegaConf
import torch
import numpy as np


class PromoterSampler(BaseSampler):
    """Sampler specifically for Promoter dataset."""
    
    def __init__(self):
        super().__init__('promoter')
        
    def create_model(self, config, architecture):
        """Create Promoter-specific model."""
        return create_model(config, architecture)
        
    def get_sequence_length(self, config):
        """Get Promoter sequence length."""
        # Promoter uses 1024 bp sequences
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 1024
    
    def generate_labels(self, num_samples, config):
        """
        Generate conditioning labels for Promoter sampling.
        
        Promoter dataset has expression targets concatenated with sequences.
        For sampling, we generate random target expression values.
        """
        # Generate random expression targets
        # Shape: (num_samples, seq_length, 1) to match the concatenated format
        seq_length = self.get_sequence_length(config)
        labels = torch.randn(num_samples, seq_length, 1, device=self.device) * 2.0  # Adjust scale as needed
        
        return labels
    
    def process_sampled_sequences(self, sequences, labels=None):
        """
        Process sampled sequences for promoter dataset.
        
        For promoter, we might want to combine sequences with their target expressions.
        """
        if labels is not None and labels.dim() == 3:
            # Convert sequences to one-hot
            one_hot_sequences = torch.nn.functional.one_hot(sequences, num_classes=4).float()
            
            # Concatenate with labels (expressions)
            combined = torch.cat([one_hot_sequences, labels], dim=-1)
            
            return combined
        else:
            return sequences
    
    def sample_with_expression_targets(self, checkpoint_path, config, architecture,
                                     expression_targets=None, **kwargs):
        """
        Sample sequences with specific expression targets.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object
            architecture: Architecture name
            expression_targets: Target expression values (float, list, or tensor)
            **kwargs: Additional sampling arguments
        """
        # Load model
        model = self.load_model_from_checkpoint(checkpoint_path, config, architecture)
        
        num_samples = kwargs.get('num_samples', 1000)
        seq_length = self.get_sequence_length(config)
        
        # Create specific expression targets if provided
        if expression_targets is not None:
            if isinstance(expression_targets, (int, float)):
                # Single expression value for all samples
                expression_targets = torch.full(
                    (num_samples, seq_length, 1), 
                    expression_targets, 
                    device=self.device, 
                    dtype=torch.float32
                )
            elif isinstance(expression_targets, (list, np.ndarray)):
                if len(expression_targets) == num_samples:
                    # Different expression for each sample
                    expression_targets = torch.tensor([
                        [[exp] for _ in range(seq_length)] for exp in expression_targets
                    ], device=self.device, dtype=torch.float32)
                else:
                    raise ValueError(f"Expression targets length {len(expression_targets)} != num_samples {num_samples}")
            elif isinstance(expression_targets, torch.Tensor):
                expression_targets = expression_targets.to(self.device)
            else:
                raise ValueError(f"Invalid expression_targets type: {type(expression_targets)}")
                
            # Override the generate_labels method for this call
            original_generate_labels = self.generate_labels
            self.generate_labels = lambda n, cfg: expression_targets
            
            try:
                sequences = self.sample_sequences(model, config, num_samples, **kwargs)
                # Process with expression targets
                processed_sequences = self.process_sampled_sequences(sequences, expression_targets)
            finally:
                # Restore original method
                self.generate_labels = original_generate_labels
        else:
            sequences = self.sample_sequences(model, config, num_samples, **kwargs)
            processed_sequences = sequences
        
        return processed_sequences
    
    def save_promoter_sequences(self, sequences, output_path, format='fasta', include_expression=False):
        """
        Save promoter sequences, optionally including expression information.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Handle different sequence formats
        if sequences.dim() == 3 and sequences.shape[-1] == 5:
            # Combined sequence + expression format
            seq_part = sequences[:, :, :4]  # One-hot sequence
            exp_part = sequences[:, :, 4:5]  # Expression values
            
            # Convert one-hot to indices
            seq_indices = torch.argmax(seq_part, dim=-1)
            sequences_str = self.sequences_to_strings(seq_indices)
            
            # Extract expression values (average over sequence length)
            expression_values = torch.mean(exp_part, dim=1).squeeze().cpu().numpy()
        else:
            # Standard sequence format
            sequences_str = self.sequences_to_strings(sequences)
            expression_values = None
        
        with open(output_path, 'w') as f:
            if format.lower() == 'fasta':
                for i, seq_str in enumerate(sequences_str):
                    if include_expression and expression_values is not None:
                        f.write(f">sequence_{i}_expression_{expression_values[i]:.3f}\n")
                    else:
                        f.write(f">sequence_{i}\n")
                    f.write(f"{seq_str}\n")
            elif format.lower() == 'csv':
                if include_expression and expression_values is not None:
                    f.write("sequence_id,sequence,expression\n")
                    for i, seq_str in enumerate(sequences_str):
                        f.write(f"sequence_{i},{seq_str},{expression_values[i]:.6f}\n")
                else:
                    f.write("sequence_id,sequence\n")
                    for i, seq_str in enumerate(sequences_str):
                        f.write(f"sequence_{i},{seq_str}\n")
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        print(f"Promoter sequences saved to: {output_path}")


def load_config(architecture):
    """Load Promoter configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function."""
    parser = parse_base_args()
    parser.description = 'Promoter Sampling Script'
    
    # Add Promoter-specific arguments
    parser.add_argument('--expression_target', type=float, 
                       help='Target expression value for all sequences')
    parser.add_argument('--include_expression', action='store_true',
                       help='Include expression values in output')
    
    args = parser.parse_args()
    
    # Create sampler
    sampler = PromoterSampler()
    
    # Load config if not provided
    if not args.config:
        try:
            config = load_config(args.architecture)
        except FileNotFoundError:
            print(f"Error: No config provided and default config not found for architecture: {args.architecture}")
            return 1
    else:
        config = OmegaConf.load(args.config)
    
    # Use specific expression target sampling if requested
    if args.expression_target is not None:
        sequences = sampler.sample_with_expression_targets(
            checkpoint_path=args.checkpoint,
            config=config,
            architecture=args.architecture,
            expression_targets=args.expression_target,
            num_samples=args.num_samples,
            method=args.method,
            num_steps=args.num_steps,
            eta=args.eta,
            temperature=args.temperature
        )
        
        # Save sequences with custom method
        output_path = args.output or f"samples/promoter_{args.architecture}_{args.method}_targeted_samples.{args.format}"
        sampler.save_promoter_sequences(sequences, output_path, args.format, args.include_expression)
        
        print(f"Sampling completed with expression target {args.expression_target}. {args.num_samples} sequences generated.")
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