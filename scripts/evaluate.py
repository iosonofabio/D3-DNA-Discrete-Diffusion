#!/usr/bin/env python3
"""
Base Evaluation Module for D3-DNA Discrete Diffusion

This module provides the base evaluation functionality that can be inherited
by dataset-specific evaluation scripts. It includes common model loading,
evaluation metrics, and oracle-based evaluation capabilities.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Package imports

from utils.checkpoint_utils import is_original_checkpoint
from utils.load_model import load_model_from_checkpoint


class BaseEvaluator:
    """
    Base evaluator class that provides common evaluation functionality.
    
    Dataset-specific evaluation scripts should inherit from this class and
    implement the abstract methods for their specific needs.
    """
    
    def __init__(self, dataset_name: str, config: Optional[OmegaConf] = None):
        self.dataset_name = dataset_name
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def create_model(self, config: OmegaConf, architecture: str):
        """
        Create the model for evaluation. Must be implemented by subclasses.
        
        Args:
            config: Configuration object
            architecture: Architecture name (e.g., 'transformer', 'convolutional')
            
        Returns:
            Model instance
        """
        raise NotImplementedError("Subclasses must implement create_model()")
    
    def load_model_from_checkpoint(self, checkpoint_path: str, config: OmegaConf, architecture: str):
        """
        Load model from checkpoint with proper handling of different checkpoint formats.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            architecture: Architecture name
            
        Returns:
            Loaded model
        """
        print(f"Loading model from {checkpoint_path}")
        
        # Create model
        model = self.create_model(config, architecture)
        model.to(self.device)
        
        # Load checkpoint weights
        if checkpoint_path.endswith('.ckpt'):
            # PyTorch Lightning checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                # Remove 'model.' or 'score_model.' prefix if present
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith('score_model.'):
                        k = k[12:]  # Remove 'score_model.' prefix
                    elif k.startswith('model.'):
                        k = k[6:]  # Remove 'model.' prefix
                    state_dict[k] = v
            else:
                state_dict = checkpoint
        else:
            # Regular PyTorch checkpoint or original D3 format
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model' in checkpoint:
                # Original D3 checkpoint format
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        
        # Load state dict with non-strict loading to handle minor incompatibilities
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
        print("✓ Model loaded successfully")
        return model
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """
        Create dataloader for evaluation. Must be implemented by subclasses.
        
        Args:
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            batch_size: Batch size (if None, uses config default)
            
        Returns:
            DataLoader instance
        """
        raise NotImplementedError("Subclasses must implement create_dataloader()")
    
    def compute_basic_metrics(self, model, dataloader, config: OmegaConf) -> Dict[str, float]:
        """
        Compute basic evaluation metrics (loss, perplexity, etc.).
        
        Args:
            model: Trained model
            dataloader: Data loader
            config: Configuration object
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        
        total_loss = 0.0
        num_batches = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing basic metrics"):
                inputs, targets = self.process_batch(batch)
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                batch_size = inputs.shape[0]
                
                # Generate random noise levels for evaluation
                sigma = torch.rand(batch_size, device=self.device) * 20
                
                # Model prediction
                if targets is not None:
                    output = model(inputs, targets, train=False, sigma=sigma)
                else:
                    output = model(inputs, train=False, sigma=sigma)
                
                # Compute loss based on dataset type
                loss = self.compute_loss(output, inputs, targets)
                
                total_loss += loss.item() * batch_size
                num_batches += 1
                num_samples += batch_size
        
        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        
        metrics = {
            'average_loss': avg_loss,
            'num_batches': num_batches,
            'num_samples': num_samples,
            'perplexity': np.exp(avg_loss) if avg_loss < 10 else float('inf')  # Avoid overflow
        }
        
        return metrics
    
    def process_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process batch data into inputs and targets.
        Can be overridden by subclasses for dataset-specific processing.
        
        Args:
            batch: Raw batch data from dataloader
            
        Returns:
            Tuple of (inputs, targets)
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        else:
            return batch, None
    
    def compute_loss(self, output: torch.Tensor, inputs: torch.Tensor, targets: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for evaluation. Can be overridden by subclasses.
        
        Args:
            output: Model output
            inputs: Input sequences
            targets: Target values (can be None)
            
        Returns:
            Loss tensor
        """
        # Default cross-entropy loss for sequence modeling
        if output.dim() == 3:  # (batch_size, seq_len, vocab_size)
            loss = F.cross_entropy(
                output.view(-1, output.shape[-1]), 
                inputs.view(-1),
                reduction='mean'
            )
        else:
            # Handle other output shapes
            loss = F.mse_loss(output, torch.zeros_like(output))
            
        return loss
    
    def load_oracle_model(self, oracle_checkpoint: str, data_path: str):
        """
        Load oracle model for specialized evaluation metrics.
        Must be implemented by subclasses if oracle evaluation is supported.
        
        Args:
            oracle_checkpoint: Path to oracle model checkpoint
            data_path: Path to data file needed by oracle
            
        Returns:
            Oracle model instance or None if not supported
        """
        print(f"Warning: Oracle evaluation not implemented for {self.dataset_name}")
        return None
    
    def evaluate_with_oracle(self, model, oracle_model, dataloader, config: OmegaConf) -> Dict[str, Any]:
        """
        Evaluate model using oracle model (for SP-MSE and similar metrics).
        Can be overridden by subclasses for dataset-specific oracle evaluation.
        
        Args:
            model: Trained diffusion model
            oracle_model: Oracle model for evaluation
            dataloader: Data loader
            config: Configuration object
            
        Returns:
            Dictionary of oracle-based evaluation metrics
        """
        if oracle_model is None:
            return {'oracle_evaluation': 'not_supported'}
        
        # Default implementation returns placeholder
        return {
            'oracle_evaluation': 'not_implemented',
            'sp_mse': 0.0
        }
    
    def evaluate(self, checkpoint_path: str, config: OmegaConf, architecture: str,
                 split: str = 'test', oracle_checkpoint: Optional[str] = None,
                 data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main evaluation method.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object
            architecture: Architecture name
            split: Dataset split to evaluate on
            oracle_checkpoint: Path to oracle model checkpoint (optional)
            data_path: Path to data file (optional, needed for oracle)
            
        Returns:
            Dictionary of evaluation results
        """
        # Load model
        model = self.load_model_from_checkpoint(checkpoint_path, config, architecture)
        
        # Create dataloader
        dataloader = self.create_dataloader(config, split)
        
        print(f"Evaluating on {split} split...")
        
        # Compute basic metrics
        metrics = self.compute_basic_metrics(model, dataloader, config)
        
        # Oracle evaluation if requested
        if oracle_checkpoint:
            print("Loading oracle model for specialized evaluation...")
            oracle_model = self.load_oracle_model(oracle_checkpoint, data_path or "")
            oracle_metrics = self.evaluate_with_oracle(model, oracle_model, dataloader, config)
            metrics.update(oracle_metrics)
        
        return metrics
    
    def save_results(self, metrics: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)  # default=str handles numpy types
        
        print(f"Results saved to: {output_path}")
    
    def print_results(self, metrics: Dict[str, Any]):
        """Print evaluation results in a formatted way."""
        print("\nEvaluation Results:")
        print("=" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")


def parse_base_args():
    """Parse common command line arguments for evaluation scripts."""
    parser = argparse.ArgumentParser(description='D3 Evaluation Script')
    parser.add_argument('--architecture', required=True, help='Architecture (transformer or convolutional)')
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config', help='Override config file (optional)')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='Dataset split to evaluate on')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--use_oracle', action='store_true', help='Use oracle model for evaluation')
    parser.add_argument('--oracle_checkpoint', help='Path to oracle model checkpoint')
    parser.add_argument('--data_path', help='Path to data file (needed for oracle models)')
    parser.add_argument('--batch_size', type=int, help='Batch size for evaluation')
    
    return parser


def main_evaluate(evaluator: BaseEvaluator, args):
    """
    Common main evaluation function that can be used by dataset-specific scripts.
    
    Args:
        evaluator: Dataset-specific evaluator instance
        args: Parsed command line arguments
    """
    # Load configuration
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        # Use default config loading method (to be implemented by subclasses)
        raise ValueError("Config file must be provided")
    
    # Run evaluation
    metrics = evaluator.evaluate(
        checkpoint_path=args.checkpoint,
        config=config,
        architecture=args.architecture,
        split=args.split,
        oracle_checkpoint=args.oracle_checkpoint if args.use_oracle else None,
        data_path=args.data_path
    )
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results
    output_path = args.output or f"evaluation_results/{evaluator.dataset_name}_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    return 0