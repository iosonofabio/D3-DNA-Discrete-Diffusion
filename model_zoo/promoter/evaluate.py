#!/usr/bin/env python3
"""
Promoter Evaluation Script

Inherits from base evaluation framework while using Promoter-specific models directly.
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

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and Promoter-specific components
from scripts.evaluate import BaseEvaluator, parse_base_args, main_evaluate
from model_zoo.promoter.models import create_model
from model_zoo.promoter.data import get_promoter_datasets
from model_zoo.promoter.sei import Sei, NonStrandSpecific


class PromoterEvaluator(BaseEvaluator):
    """Promoter-specific evaluator that inherits from base framework."""
    
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
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """Create Promoter dataloader."""
        # Load datasets 
        train_ds, val_ds = get_promoter_datasets()
        
        # Select appropriate dataset
        if split == 'train':
            dataset = train_ds
        elif split in ['val', 'test']:  # Use val as test for now
            dataset = val_ds
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # Use config batch size if not specified
        if batch_size is None:
            batch_size = getattr(config, 'batch_size', 32)
            if hasattr(config, 'eval') and hasattr(config.eval, 'batch_size'):
                batch_size = config.eval.batch_size
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def load_oracle_model(self, oracle_checkpoint: str, data_path: str):
        """Load Promoter oracle model (Sei)."""
        try:
            # Load Sei oracle model
            oracle = Sei(sequence_length=1024, n_targets=21907)
            oracle = NonStrandSpecific(oracle)
            
            # Load checkpoint if provided
            if oracle_checkpoint and os.path.exists(oracle_checkpoint):
                checkpoint = torch.load(oracle_checkpoint, map_location=self.device)
                oracle.load_state_dict(checkpoint)
            
            oracle.to(self.device)
            oracle.eval()
            
            print("✓ Loaded Promoter oracle model (Sei)")
            return oracle
            
        except Exception as e:
            print(f"Failed to load Promoter oracle model: {e}")
            return None
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """Get original test data for SP-MSE comparison."""
        try:
            # Load Promoter test data
            train_ds, val_ds = get_promoter_datasets()
            
            # Create a small batch for comparison
            dataloader = DataLoader(val_ds, batch_size=100, shuffle=False)
            batch = next(iter(dataloader))
            
            if len(batch) == 2:
                sequences, _ = batch
                return sequences
            else:
                return batch
                
        except Exception as e:
            print(f"Error loading original test data: {e}")
            # Return dummy data as fallback
            return torch.zeros(100, 1024, 4)  # One-hot encoded sequences


def load_config(architecture: str):
    """Load Promoter configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main evaluation function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    parser.add_argument('--model_path', required=True, help='Path to model directory (required for evaluation)')
    parser.add_argument('--steps', type=int, help='Number of sampling steps (defaults to sequence length)')
    args = parser.parse_args()
    
    # Validate required arguments for evaluation
    if not args.oracle_checkpoint:
        print("Error: --oracle_checkpoint is required for evaluation")
        return 1
    if not args.data_path:
        print("Error: --data_path is required for evaluation")
        return 1
    
    # Load config if not provided
    if not args.config:
        try:
            config_path = Path(__file__).parent / 'configs' / f'{args.architecture}.yaml'
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
    evaluator = PromoterEvaluator()
    
    # Run evaluation (always includes sampling + SP-MSE computation)
    metrics = evaluator.evaluate_with_sampling(
        model_path=args.model_path,
        config=config,
        oracle_checkpoint=args.oracle_checkpoint,
        data_path=args.data_path,
        split=args.split,
        steps=args.steps,
        batch_size=args.batch_size
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/promoter_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ Promoter evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())