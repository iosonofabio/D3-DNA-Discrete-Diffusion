#!/usr/bin/env python3
"""
Base Sampling Module for D3-DNA Discrete Diffusion

This module provides the base sampling functionality that can be inherited
by dataset-specific sampling scripts. It includes common model loading,
sampling algorithms (DDPM/DDIM), and sequence export capabilities.
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

# Package imports

from utils.model_interface import ModelLoader
from utils.checkpoint_utils import is_original_checkpoint


class BaseSampler:
    """
    Base sampler class that provides common sampling functionality.
    
    Dataset-specific sampling scripts should inherit from this class and
    implement the abstract methods for their specific needs.
    """
    
    def __init__(self, dataset_name: str, config: Optional[OmegaConf] = None):
        self.dataset_name = dataset_name
        self.config = config
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
    
    def load_model_from_checkpoint(self, checkpoint_path: str, config: OmegaConf, architecture: str):
        """
        Load model from checkpoint using the generic model loader.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            architecture: Architecture name (kept for compatibility)
            
        Returns:
            Loaded model
        """
        # Use the generic model loader
        loader = ModelLoader(self.device)
        model, _, _ = loader.load_model_from_config(config, checkpoint_path)
        return model
    
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
        
        # Get dataset name from config if available
        dataset_name = config.dataset.name.lower() if hasattr(config, 'dataset') and hasattr(config.dataset, 'name') else self.dataset_name.lower()
        
        # Dataset-specific defaults
        defaults = {
            'deepstarr': 249,
            'mpra': 200,
            'promoter': 1024
        }
        return defaults.get(dataset_name, 249)
    
    def get_vocab_size(self, config: OmegaConf) -> int:
        """
        Get vocabulary size for the dataset. Can be overridden by subclasses.
        
        Args:
            config: Configuration object
            
        Returns:
            Vocabulary size
        """
        if hasattr(config, 'tokens'):
            return config.tokens
        elif hasattr(config, 'model') and hasattr(config.model, 'vocab_size'):
            return config.model.vocab_size
        else:
            return 4  # Default for DNA sequences (A, C, G, T)
    
    def generate_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        """
        Generate conditioning labels for sampling. Can be overridden by subclasses.
        
        Args:
            num_samples: Number of samples to generate labels for
            config: Configuration object
            
        Returns:
            Label tensor
        """
        # Try to get label dimensions from config first
        if hasattr(config, 'model') and hasattr(config.model, 'label_dim'):
            label_dim = config.model.label_dim
            return torch.randn(num_samples, label_dim, device=self.device)
        elif hasattr(config, 'dataset') and hasattr(config.dataset, 'label_dim'):
            label_dim = config.dataset.label_dim
            return torch.randn(num_samples, label_dim, device=self.device)
        
        # Fallback to dataset-specific defaults
        dataset_name = config.dataset.name.lower() if hasattr(config, 'dataset') and hasattr(config.dataset, 'name') else self.dataset_name.lower()
        
        if dataset_name == 'deepstarr':
            # DeepSTARR has 2 labels (Dev and HK enhancer activity)
            return torch.randn(num_samples, 2, device=self.device)
        elif dataset_name == 'mpra':
            # MPRA has 3 labels
            return torch.randn(num_samples, 3, device=self.device)
        elif dataset_name == 'promoter':
            # Promoter has 1 label
            return torch.randn(num_samples, 1, device=self.device)
        else:
            # Default to single value
            return torch.randn(num_samples, 1, device=self.device)
    
    def sample_ddpm(self, model, config: OmegaConf, num_samples: int, num_steps: int = 128) -> torch.Tensor:
        """
        Sample using DDPM (Denoising Diffusion Probabilistic Models) algorithm.
        
        Args:
            model: Trained model
            config: Configuration object
            num_samples: Number of sequences to sample
            num_steps: Number of denoising steps
            
        Returns:
            Generated sequences
        """
        model.eval()
        
        sequence_length = self.get_sequence_length(config)
        vocab_size = self.get_vocab_size(config)
        
        with torch.no_grad():
            # Initialize with random tokens
            sequences = torch.randint(0, vocab_size, (num_samples, sequence_length), device=self.device)
            
            # Generate labels
            labels = self.generate_labels(num_samples, config)
            
            # DDPM sampling process
            for step in tqdm(range(num_steps), desc="DDPM Sampling"):
                # Noise level decreases over time
                t = (num_steps - step - 1) / num_steps
                sigma = torch.full((num_samples,), t, device=self.device) * 20
                
                # Model prediction
                with torch.no_grad():
                    if labels is not None:
                        output = model(sequences, labels, train=False, sigma=sigma)
                    else:
                        output = model(sequences, train=False, sigma=sigma)
                
                # Update sequences based on model output
                if step < num_steps - 1:
                    # Apply sampling strategy
                    if output.dim() == 3 and output.shape[-1] == vocab_size:
                        # Logits output - sample from distribution
                        probs = torch.softmax(output / max(0.1, t), dim=-1)
                        sequences = torch.multinomial(probs.view(-1, vocab_size), 1).view(num_samples, sequence_length)
                    else:
                        # Handle other output formats
                        sequences = torch.clamp(sequences + 0.1 * torch.randn_like(sequences.float()), 0, vocab_size - 1).long()
        
        return sequences
    
    def sample_ddim(self, model, config: OmegaConf, num_samples: int, num_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """
        Sample using DDIM (Denoising Diffusion Implicit Models) algorithm.
        
        Args:
            model: Trained model
            config: Configuration object  
            num_samples: Number of sequences to sample
            num_steps: Number of denoising steps
            eta: DDIM parameter (0 = deterministic, 1 = DDPM)
            
        Returns:
            Generated sequences
        """
        model.eval()
        
        sequence_length = self.get_sequence_length(config)
        vocab_size = self.get_vocab_size(config)
        
        with torch.no_grad():
            # Initialize with random tokens
            sequences = torch.randint(0, vocab_size, (num_samples, sequence_length), device=self.device)
            
            # Generate labels
            labels = self.generate_labels(num_samples, config)
            
            # Create DDIM schedule
            ddim_timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
            
            # DDIM sampling process
            for i in tqdm(range(num_steps), desc="DDIM Sampling"):
                t_curr = ddim_timesteps[i]
                t_next = ddim_timesteps[i + 1]
                
                sigma_curr = t_curr * 20
                sigma_next = t_next * 20
                
                # Model prediction
                with torch.no_grad():
                    if labels is not None:
                        output = model(sequences, labels, train=False, sigma=sigma_curr.expand(num_samples))
                    else:
                        output = model(sequences, train=False, sigma=sigma_curr.expand(num_samples))
                
                # DDIM update (simplified for discrete case)
                if i < num_steps - 1:
                    if output.dim() == 3 and output.shape[-1] == vocab_size:
                        # Use temperature sampling with decreasing temperature
                        temperature = max(0.1, float(t_curr))
                        probs = torch.softmax(output / temperature, dim=-1)
                        sequences = torch.multinomial(probs.view(-1, vocab_size), 1).view(num_samples, sequence_length)
        
        return sequences
    
    def sample_sequences(self, model, config: OmegaConf, num_samples: int = 1000, 
                        method: str = 'ddpm', num_steps: int = 128, **kwargs) -> torch.Tensor:
        """
        Main sampling method that dispatches to specific sampling algorithms.
        
        Args:
            model: Trained model
            config: Configuration object
            num_samples: Number of sequences to sample
            method: Sampling method ('ddpm' or 'ddim')
            num_steps: Number of sampling steps
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Generated sequences
        """
        if method.lower() == 'ddpm':
            return self.sample_ddpm(model, config, num_samples, num_steps)
        elif method.lower() == 'ddim':
            eta = kwargs.get('eta', 0.0)
            return self.sample_ddim(model, config, num_samples, num_steps, eta)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
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
    
    def save_sequences(self, sequences: torch.Tensor, output_path: str, format: str = 'fasta'):
        """
        Save generated sequences to file.
        
        Args:
            sequences: Generated sequences
            output_path: Output file path
            format: Output format ('fasta' or 'csv')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to strings
        sequences_str = self.sequences_to_strings(sequences)
        
        with open(output_path, 'w') as f:
            if format.lower() == 'fasta':
                for i, seq_str in enumerate(sequences_str):
                    f.write(f">sequence_{i}\n")
                    f.write(f"{seq_str}\n")
            elif format.lower() == 'csv':
                f.write("sequence_id,sequence\n")
                for i, seq_str in enumerate(sequences_str):
                    f.write(f"sequence_{i},{seq_str}\n")
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        print(f"Sequences saved to: {output_path}")
    
    def sample(self, checkpoint_path: str, config: OmegaConf, architecture: str,
               num_samples: int = 1000, method: str = 'ddpm', num_steps: int = 128,
               output_path: Optional[str] = None, format: str = 'fasta', **kwargs) -> torch.Tensor:
        """
        Main sampling method.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object
            architecture: Architecture name
            num_samples: Number of sequences to sample
            method: Sampling method ('ddpm' or 'ddim')
            num_steps: Number of sampling steps
            output_path: Output file path (if None, auto-generated)
            format: Output format ('fasta' or 'csv')
            **kwargs: Additional sampling arguments
            
        Returns:
            Generated sequences
        """
        # Load model
        model = self.load_model_from_checkpoint(checkpoint_path, config, architecture)
        
        print(f"Sampling {num_samples} sequences using {method.upper()} with {num_steps} steps...")
        
        # Sample sequences
        sequences = self.sample_sequences(model, config, num_samples, method, num_steps, **kwargs)
        
        # Save sequences
        if output_path is None:
            output_path = f"samples/{self.dataset_name}_{architecture}_{method}_samples.{format}"
        
        self.save_sequences(sequences, output_path, format)
        
        return sequences


def parse_base_args():
    """Parse common command line arguments for sampling scripts."""
    parser = argparse.ArgumentParser(description='D3 Sampling Script')
    parser.add_argument('--architecture', required=True, help='Architecture (transformer or convolutional)')
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config', help='Override config file (optional)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--method', choices=['ddpm', 'ddim'], default='ddpm', help='Sampling method')
    parser.add_argument('--num_steps', type=int, default=128, help='Number of sampling steps')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta parameter (0=deterministic, 1=DDPM)')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['fasta', 'csv'], default='fasta', help='Output format')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    
    return parser


def main_sample(sampler: BaseSampler, args):
    """
    Common main sampling function that can be used by dataset-specific scripts.
    
    Args:
        sampler: Dataset-specific sampler instance
        args: Parsed command line arguments
    """
    # Load configuration - now required
    if not args.config:
        raise ValueError("Config file is required. Please provide --config path/to/config.yaml")
    
    config = OmegaConf.load(args.config)
    
    # Validate that checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Run sampling
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