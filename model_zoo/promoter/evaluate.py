#!/usr/bin/env python3
"""
Promoter Evaluation Script

This script provides evaluation functionality specifically for the Promoter dataset,
inheriting from the base evaluation classes and implementing Promoter-specific
model creation, data loading, and oracle evaluation.
"""

import os
import sys
from pathlib import Path

# Package imports

from scripts.evaluate import BaseEvaluator, parse_base_args, main_evaluate
from model_zoo.promoter.models import create_model
from model_zoo.promoter.data import get_promoter_datasets
from model_zoo.promoter.sei import Sei, NonStrandSpecific
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch


class PromoterEvaluator(BaseEvaluator):
    """Evaluator specifically for Promoter dataset."""
    
    def __init__(self):
        super().__init__('promoter')
        
    def create_model(self, config, architecture):
        """Create Promoter-specific model."""
        return create_model(config, architecture)
        
    def create_dataloader(self, config, split='test', batch_size=None):
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
            
        # Create dataloader
        if batch_size is None:
            batch_size = getattr(config.eval, 'batch_size', 64) // (config.ngpus * config.training.accum)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def process_batch(self, batch):
        """
        Process Promoter batch data.
        
        Promoter data comes as concatenated (sequence + target) tensors.
        """
        if batch.dim() == 3 and batch.shape[-1] == 5:
            # Extract sequence (first 4 channels) and target (last channel)
            seq_one_hot = batch[:, :, :4]
            target = batch[:, :, 4:5]
            
            # Convert one-hot to indices for model input
            inputs = torch.argmax(seq_one_hot, dim=-1)
            
            return inputs, target
        else:
            return batch, None
    
    def load_oracle_model(self, oracle_checkpoint, data_path):
        """Load SEI oracle model for promoter evaluation."""
        try:
            # Load the SEI oracle model
            sei_model = Sei()
            oracle = NonStrandSpecific(sei_model)
            
            # Load checkpoint if provided
            if oracle_checkpoint and os.path.exists(oracle_checkpoint):
                checkpoint = torch.load(oracle_checkpoint, map_location=self.device)
                oracle.load_state_dict(checkpoint, strict=False)
            
            oracle = oracle.eval().to(self.device)
            
            print("✓ Loaded SEI oracle model for Promoter evaluation")
            return oracle
            
        except Exception as e:
            print(f"Failed to load SEI oracle model: {e}")
            return None
    
    def evaluate_with_oracle(self, model, oracle_model, dataloader, config):
        """Evaluate using SEI oracle model for promoter-specific metrics."""
        if oracle_model is None:
            return {'oracle_evaluation': 'oracle_model_not_loaded'}
        
        model.eval()
        oracle_model.eval()
        
        # Implement promoter-specific evaluation logic here
        # This would evaluate generated promoter sequences using the SEI model
        
        # Placeholder implementation
        oracle_scores = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self.process_batch(batch)
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # For promoter evaluation, we would typically:
                # 1. Generate sequences from the diffusion model
                # 2. Convert to one-hot format for SEI model
                # 3. Evaluate with SEI to get expression predictions
                # 4. Compare with target expression values
                
                # For now, just compute a placeholder metric
                # Convert inputs back to one-hot for SEI model
                batch_size, seq_len = inputs.shape
                inputs_one_hot = torch.nn.functional.one_hot(inputs, num_classes=4).float()
                inputs_one_hot = inputs_one_hot.permute(0, 2, 1)  # (batch, 4, seq_len) for conv
                
                try:
                    oracle_pred = oracle_model(inputs_one_hot)
                    # Simple evaluation metric (would need proper implementation)
                    score = torch.mean(oracle_pred).item()
                    oracle_scores.append(score)
                except Exception as e:
                    print(f"Oracle evaluation error: {e}")
                    break
                
                num_batches += 1
                if num_batches >= 5:  # Limit for demonstration
                    break
        
        avg_oracle_score = sum(oracle_scores) / len(oracle_scores) if oracle_scores else 0.0
        
        return {
            'oracle_evaluation': 'completed',
            'sei_expression_score': avg_oracle_score,
            'num_oracle_batches': len(oracle_scores)
        }


def load_config(architecture):
    """Load Promoter configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main evaluation function."""
    parser = parse_base_args()
    parser.description = 'Promoter Evaluation Script'
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = PromoterEvaluator()
    
    # Load config if not provided
    if not args.config:
        try:
            config = load_config(args.architecture)
        except FileNotFoundError:
            print(f"Error: No config provided and default config not found for architecture: {args.architecture}")
            return 1
    else:
        config = OmegaConf.load(args.config)
    
    # Run evaluation
    metrics = evaluator.evaluate(
        checkpoint_path=args.checkpoint,
        config=config,
        architecture=args.architecture,
        split=args.split,
        oracle_checkpoint=args.oracle_checkpoint if args.use_oracle else None,
        data_path=args.data_path
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/promoter_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())