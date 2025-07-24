#!/usr/bin/env python3
"""
Base Sampling Framework for D3-DNA Discrete Diffusion

This module provides the base sampling framework that dataset-specific
sampling scripts should inherit from. It uses the proper PC sampler
and provides common functionality while allowing datasets to implement
their own model and data loading.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import the proper sampling functionality
from scripts import sampling
from utils.load_model import load_model_local


class BaseSampler:
    """
    Base sampler class that provides common sampling functionality.
    
    Dataset-specific sampling scripts should inherit from this class and
    implement the abstract methods for their specific needs.
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Default token to nucleotide mapping (can be overridden by subclasses)
        self.token_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        
    def create_model(self, config: OmegaConf, architecture: str):
        """
        Create the model for sampling. Must be implemented by subclasses.
        
        Args:
            config: Configuration object
            architecture: Architecture name (e.g., 'transformer', 'convolutional')
            
        Returns:
            Model instance
        """
        raise NotImplementedError("Subclasses must implement create_model()")
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """
        Get sequence length for the dataset. Can be overridden by subclasses.
        
        Args:
            config: Configuration object
            
        Returns:
            Sequence length
        """
        # Try common config locations
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'sequence_length'):
            return config.dataset.sequence_length
        elif hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        elif hasattr(config, 'data') and hasattr(config.data, 'sequence_length'):
            return config.data.sequence_length
        else:
            # Dataset-specific defaults - should be overridden by subclasses
            defaults = {
                'deepstarr': 249,
                'mpra': 200,
                'promoter': 1024
            }
            return defaults.get(self.dataset_name.lower(), 249)
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        """
        Generate conditioning labels for sampling. Can be overridden by subclasses.
        
        Args:
            num_samples: Number of samples to generate labels for
            config: Configuration object
            
        Returns:
            Conditioning labels tensor
        """
        # Default: random conditioning (can be overridden by subclasses)
        if hasattr(config, 'model') and hasattr(config.model, 'num_classes'):
            num_classes = config.model.num_classes
            return torch.randn(num_samples, num_classes, device=self.device)
        else:
            # Fallback for datasets like DeepSTARR with 2 activities
            return torch.randn(num_samples, 2, device=self.device)
    
    def sample_sequences_with_pc_sampler(self, model_path: str, config: OmegaConf, 
                                       num_samples: int, steps: int, 
                                       conditioning_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample sequences using the proper PC sampler.
        
        Args:
            model_path: Path to model directory (for load_model_local)
            config: Configuration object
            num_samples: Number of sequences to sample
            steps: Number of sampling steps
            conditioning_labels: Optional conditioning labels (if None, generates random)
            
        Returns:
            Sampled sequences tensor
        """
        # Load model using the proper load_model_local function
        model, graph, noise = load_model_local(model_path, self.device)
        model.eval()
        
        sequence_length = self.get_sequence_length(config)
        
        # Generate conditioning labels if not provided
        if conditioning_labels is None:
            conditioning_labels = self.generate_conditioning_labels(num_samples, config)
        
        # Create PC sampler
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (num_samples, sequence_length), 'analytic', steps, device=self.device
        )
        
        # Sample sequences
        sampled_sequences = sampling_fn(model, conditioning_labels.to(self.device))
        
        return sampled_sequences
    
    
    def sequences_to_strings(self, sequences: torch.Tensor) -> List[str]:
        """
        Convert token sequences to nucleotide strings.
        
        Args:
            sequences: Token sequences (num_samples, seq_length)
            
        Returns:
            List of nucleotide strings
        """
        sequences_str = []
        for seq in sequences:
            seq_str = ''.join([self.token_to_nucleotide.get(token.item(), 'N') for token in seq])
            sequences_str.append(seq_str)
        return sequences_str
    
    def save_sequences(self, sequences: torch.Tensor, output_path: str, format: str = 'npz'):
        """
        Save generated sequences to file.
        
        Args:
            sequences: Generated sequences
            output_path: Output file path
            format: Output format ('npz', 'fasta', or 'csv')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'npz':
            # Save as numpy array (matches original implementation)
            np.savez(output_path, sequences.cpu().numpy())
            print(f"Sequences saved to: {output_path}")
            
        elif format.lower() == 'fasta':
            # Convert to strings and save as FASTA
            sequences_str = self.sequences_to_strings(sequences)
            with open(output_path, 'w') as f:
                for i, seq_str in enumerate(sequences_str):
                    f.write(f">{self.dataset_name}_sequence_{i}\n")
                    f.write(f"{seq_str}\n")
            print(f"Sequences saved to: {output_path}")
            
        elif format.lower() == 'csv':
            # Convert to strings and save as CSV
            sequences_str = self.sequences_to_strings(sequences)
            with open(output_path, 'w') as f:
                f.write("sequence_id,sequence\n")
                for i, seq_str in enumerate(sequences_str):
                    f.write(f"{self.dataset_name}_sequence_{i},{seq_str}\n")
            print(f"Sequences saved to: {output_path}")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def sample_and_save(self, model_path: str, config: OmegaConf, num_samples: int, steps: int,
                       conditioning_labels: Optional[torch.Tensor] = None,
                       output_path: Optional[str] = None, format: str = 'npz') -> Dict[str, Any]:
        """
        Main sampling method - just samples and saves (no evaluation).
        
        Args:
            model_path: Path to model directory
            config: Configuration object
            num_samples: Number of sequences to sample
            steps: Number of sampling steps
            conditioning_labels: Optional conditioning labels
            output_path: Output file path (optional, auto-generated if None)
            format: Output format ('npz', 'fasta', 'csv')
            
        Returns:
            Dictionary of sampling results
        """
        print(f"Sampling {num_samples} {self.dataset_name} sequences using PC sampler with {steps} steps...")
        
        # Sample sequences
        sampled_sequences = self.sample_sequences_with_pc_sampler(
            model_path, config, num_samples, steps, conditioning_labels
        )
        
        results = {
            'num_samples': sampled_sequences.shape[0],
            'sequence_length': sampled_sequences.shape[1],
            'sampling_steps': steps,
            'dataset': self.dataset_name
        }
        
        # Save sequences
        if output_path is None:
            output_path = os.path.join(model_path, f"sample.{format}")
        
        self.save_sequences(sampled_sequences, output_path, format)
        results['output_path'] = output_path
        
        return results


def parse_base_args():
    """Parse common command line arguments for sampling scripts."""
    parser = argparse.ArgumentParser(description='D3 Sampling Script')
    parser.add_argument('--model_path', required=True, help='Path to model directory')
    parser.add_argument('--config', help='Path to config file (optional, dataset may provide default)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--steps', type=int, help='Number of sampling steps (defaults to sequence length)')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['npz', 'fasta', 'csv'], default='npz', help='Output format')
    
    return parser


def main_sample(sampler: BaseSampler, args):
    """
    Common main sampling function that can be used by dataset-specific scripts.
    
    Args:
        sampler: Dataset-specific sampler instance
        args: Parsed command line arguments
    """
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return 1
    
    # Load configuration (required)
    if not args.config:
        print("Error: Config file is required. Please provide --config path/to/config.yaml")
        return 1
    
    config = OmegaConf.load(args.config)
    
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
        output_path=args.output,
        format=args.format
    )
    
    # Print results
    print(f"\n{sampler.dataset_name} Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print(f"\n✓ {sampler.dataset_name} sampling completed successfully!")
    return 0