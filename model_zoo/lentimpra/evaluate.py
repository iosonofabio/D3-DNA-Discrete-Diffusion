#!/usr/bin/env python3
"""
LentIMPRA Evaluation Script

Inherits from base evaluation framework while using LentIMPRA-specific models directly.
Uses the existing LegNet oracle from mpralegnet.py.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Optional
from tqdm import tqdm
import h5py
import numpy as np

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and LentIMPRA-specific components
from scripts.evaluate import BaseEvaluator, parse_base_args, main_evaluate
from model_zoo.lentimpra.data import get_lentimpra_datasets


class LentIMPRAEvaluator(BaseEvaluator):
    """LentIMPRA-specific evaluator that inherits from base framework."""
    
    def __init__(self):
        super().__init__("LentIMPRA")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load LentIMPRA model using dataset-specific model loading."""
        from model_zoo.lentimpra.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get LentIMPRA sequence length."""
        return 230  # LentIMPRA fixed sequence length
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """Create LentIMPRA dataloader."""
        # Load datasets
        train_ds, val_ds, test_ds = get_lentimpra_datasets(config.paths.data_file)
        
        # Select appropriate dataset
        if split == 'train':
            dataset = train_ds
        elif split == 'val':  # Use val as test for now
            dataset = val_ds
        elif split == 'test':
            dataset = test_ds
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # Use config batch size if not specified
        if batch_size is None:
            batch_size = getattr(config.eval, 'batch_size', 32)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def load_oracle_model(self, oracle_checkpoint: str, data_path: str):
        """Load LentIMPRA oracle model using existing LegNet infrastructure."""
        try:
            # Import existing LegNet components from mpralegnet
            from model_zoo.lentimpra.mpralegnet import load_model, TrainingConfig
            
            # Check if config file exists alongside checkpoint
            oracle_dir = os.path.dirname(oracle_checkpoint)
            config_path = os.path.join(oracle_dir, 'config.json')
            
            if os.path.exists(config_path):
                # Load using existing load_model function
                oracle, config = load_model(oracle_checkpoint, config_path)
                oracle.eval()
                oracle.to(self.device)
                print(f"✓ Loaded LegNet oracle model from {oracle_checkpoint}")
                return oracle
            else:
                # Fallback: create default config and load checkpoint
                print(f"Config file not found at {config_path}, using default config")
                from model_zoo.lentimpra.mpralegnet import get_default_config, LitModel
                
                config = get_default_config()
                oracle = LitModel(config)
                
                # Load checkpoint weights
                checkpoint = torch.load(oracle_checkpoint, map_location=self.device)
                if 'state_dict' in checkpoint:
                    oracle.load_state_dict(checkpoint['state_dict'])
                else:
                    oracle.load_state_dict(checkpoint)
                
                oracle.eval()
                oracle.to(self.device)
                print(f"✓ Loaded LegNet oracle model from {oracle_checkpoint} (default config)")
                return oracle
                
        except Exception as e:
            print(f"Failed to load LentIMPRA oracle model: {e}")
            return None
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """Get original test data for SP-MSE comparison."""
        try:
            # Load LentIMPRA test data from H5
            with h5py.File(data_path, 'r') as data_file:
                # Load one-hot data: (N, 230, 4)
                X = torch.tensor(np.array(data_file['onehot_test']))
            return X
        except Exception as e:
            print(f"Error loading original test data: {e}")
            return torch.zeros(100, 230, 4)
    
    def evaluate_with_sampling(self, checkpoint_path: str, config: OmegaConf, 
                              oracle_checkpoint: str, data_path: str,
                              split: str = 'test', steps: Optional[int] = None, 
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              show_progress: bool = False, save_sequences: bool = False):
        """
        Evaluate LentIMPRA model with sampling and SP-MSE computation.
        """
        print(f"Evaluating {self.dataset_name} on {split} split with sampling...")
        
        # Set default steps to sequence length if not provided
        if steps is None:
            steps = self.get_sequence_length(config)
            print(f"Using default steps: {steps} (sequence length)")
        
        # Create dataloader
        dataloader = self.create_dataloader(config, split, batch_size)
        
        # Sample sequences using PC sampler
        print(f"Sampling sequences with PC sampler ({steps} steps)...")
        sampled_sequences, target_labels = self.sample_sequences_for_evaluation(
            checkpoint_path, config, dataloader, steps, architecture, show_progress
        )
        
        # Save sequences as NPZ if requested
        if save_sequences:
            # Create output path based on checkpoint directory
            checkpoint_dir = os.path.dirname(checkpoint_path)
            npz_path = os.path.join(checkpoint_dir, "sample.npz")
            self.save_sequences_as_npz(sampled_sequences, npz_path)
        
        # Load oracle model
        print("Loading LegNet oracle model for SP-MSE evaluation...")
        oracle_model = self.load_oracle_model(oracle_checkpoint, data_path)
        
        if oracle_model is None:
            return {
                'error': 'oracle_model_not_loaded',
                'num_samples': sampled_sequences.shape[0],
                'sequence_length': sampled_sequences.shape[1],
                'sampling_steps': steps
            }
        
        # Get original test data for comparison
        original_data = self.get_original_test_data(data_path)
        
        # Compute SP-MSE
        print("Computing SP-MSE...")
        sp_mse = self.compute_sp_mse(sampled_sequences, oracle_model, original_data)
        
        results = {
            'dataset': self.dataset_name,
            'split': split,
            'num_samples': sampled_sequences.shape[0],
            'sequence_length': sampled_sequences.shape[1],
            'sampling_steps': steps,
            'sp_mse': sp_mse,
            'oracle_evaluation': 'completed'
        }
        
        print(f"SP-MSE: {sp_mse:.6f}")
        
        return results


def load_default_config():
    """Load LentIMPRA default configuration (transformer)."""
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main evaluation function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    args = parser.parse_args()
    
    # Set save_sequences default to True for LentIMPRA
    if not hasattr(args, 'save_sequences') or args.save_sequences is False:
        args.save_sequences = True
        print("✓ LentIMPRA: Enabled sequence saving by default")
    
    # Load config if not provided
    if not args.config:
        try:
            config_path = Path(__file__).parent / 'configs' / 'transformer.yaml'
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
    evaluator = LentIMPRAEvaluator()
    
    # Run evaluation (always includes sampling + SP-MSE computation)
    metrics = evaluator.evaluate_with_sampling(
        checkpoint_path=args.checkpoint,
        config=config,
        oracle_checkpoint=args.oracle_checkpoint,
        data_path=args.data_path,
        split=args.split,
        steps=args.steps,
        batch_size=args.batch_size,
        architecture=args.architecture,
        show_progress=args.show_progress,
        save_sequences=args.save_sequences
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/lentimpra_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ LentIMPRA evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())