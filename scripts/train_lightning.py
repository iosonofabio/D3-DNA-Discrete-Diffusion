#!/usr/bin/env python3
"""
PyTorch Lightning training script for D3-DNA-Discrete-Diffusion.
This script provides a Lightning-based training interface while maintaining 
compatibility with the original Hydra configuration system.
"""

import os
import sys
import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

torch.set_float32_matmul_precision('medium')

from scripts.lightning_trainer import (
    D3LightningModule, D3DataModule, create_trainer_from_config,
    create_lightning_module, get_model_class_for_dataset
)
from utils.utils import load_hydra_config_from_run, makedirs, get_logger
from utils.checkpoint_utils import is_original_checkpoint
from utils.sp_mse_callback import SPMSEValidationCallback


def setup_logging(cfg, work_dir):
    """Setup logging configuration."""
    log_dir = os.path.join(work_dir, "logs")
    makedirs(log_dir)
    
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=work_dir,
        name="lightning_logs",
        version=None
    )
    loggers.append(tb_logger)
    
    # WandB logger if configured
    if hasattr(cfg, 'wandb') and cfg.wandb.get('enabled', False):
        # Convert OmegaConf to plain dict for WandB compatibility
        from omegaconf import OmegaConf
        config_dict = OmegaConf.to_yaml(cfg)
        
        wandb_logger = WandbLogger(
            entity=cfg.wandb.get('entity', 'alejandraduran-u-cshl'),
            project=cfg.wandb.get('project', 'D3'),
            name=cfg.wandb.get('name', os.path.basename(work_dir)),
            save_dir=work_dir,
            config=OmegaConf.to_container(cfg, resolve=True)  # Convert to plain dict
        )
        loggers.append(wandb_logger)
    
    return loggers


def setup_callbacks(cfg, work_dir, dataset_name=None):
    """Setup Lightning callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(work_dir, "lightning_checkpoints"),
        filename="d3-{epoch:02d}-{step:06d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        every_n_train_steps=cfg.training.snapshot_freq,
        save_on_train_epoch_end=False
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Early stopping (optional)
    if hasattr(cfg.training, 'early_stopping') and cfg.training.early_stopping.get('enabled', False):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=cfg.training.early_stopping.get('patience', 10),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # SP-MSE validation callback (optional)
    sp_mse_callback = setup_sp_mse_callback(cfg, dataset_name)
    if sp_mse_callback:
        callbacks.append(sp_mse_callback)
    
    return callbacks


def setup_sp_mse_callback(cfg, dataset_name):
    """Setup SP-MSE validation callback if enabled."""
    if not hasattr(cfg, 'sp_mse_validation') or not cfg.sp_mse_validation.enabled:
        return None
    
    # Auto-resolve paths if not provided
    oracle_path = cfg.sp_mse_validation.oracle_path
    data_path = cfg.sp_mse_validation.data_path
    
    if oracle_path is None and dataset_name:
        oracle_files = {
            'deepstarr': 'oracle_DeepSTARR_DeepSTARR_data.ckpt',
            'mpra': 'oracle_mpra_mpra_data.ckpt',
            'promoter': 'best.sei.model.pth.tar'
        }
        oracle_path = f"model_zoo/{dataset_name.lower()}/oracle_models/{oracle_files[dataset_name.lower()]}"
    
    if data_path is None and dataset_name:
        data_files = {
            'deepstarr': 'DeepSTARR_data.h5',
            'mpra': 'mpra_data.h5',
            'promoter': 'promoter_data.h5'
        }
        data_path = data_files[dataset_name.lower()]
    
    sp_mse_callback = SPMSEValidationCallback(
        dataset=dataset_name or 'deepstarr',
        oracle_path=oracle_path,
        data_path=data_path,
        validation_freq=cfg.sp_mse_validation.validation_freq,
        validation_samples=cfg.sp_mse_validation.validation_samples,
        enabled=cfg.sp_mse_validation.enabled,
        sampling_steps=cfg.sp_mse_validation.sampling_steps,
        early_stopping_patience=cfg.sp_mse_validation.early_stopping_patience
    )
    
    print(f"✓ Added SP-MSE validation callback for {dataset_name} dataset")
    return sp_mse_callback


def load_from_checkpoint(checkpoint_path, model, cfg, dataset_name):
    """Load model from checkpoint, handling both formats."""
    if is_original_checkpoint(checkpoint_path):
        print(f"Loading from original D3 checkpoint: {checkpoint_path}")
        step = model.load_from_original_checkpoint(checkpoint_path)
        return model, step
    else:
        print(f"Loading from Lightning checkpoint: {checkpoint_path}")
        # Use factory to get the right Lightning module class
        model = create_lightning_module(cfg, dataset_name=dataset_name)
        model = model.load_from_checkpoint(
            checkpoint_path, 
            cfg=cfg
        )
        return model, None


def main():
    parser = argparse.ArgumentParser(description='PyTorch Lightning training for D3-DNA')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['deepstarr', 'mpra', 'promoter'],
                       help='Dataset to train on')
    parser.add_argument('--arch', type=str, required=True,
                       choices=['Conv', 'Tran'], 
                       help='Model architecture: Conv or Tran')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config file (auto-resolved if not provided)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--work_dir', type=str, default=None,
                       help='Working directory for outputs')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (overrides config)')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum training steps (overrides config)')
    parser.add_argument('--val_check_interval', type=int, default=None,
                       help='Validation check interval (overrides config)')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a fast development run for debugging')
    parser.add_argument('--dry_run', action='store_true',
                       help='Setup everything but don\'t run training')
    
    args = parser.parse_args()
    
    # Auto-resolve config path if not provided
    if args.config_path is None:
        args.config_path = f"model_zoo/{args.dataset}/config/{args.arch}/hydra/config.yaml"
        print(f"Using auto-resolved config path: {args.config_path}")
    
    # Load config
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    
    config_dir = os.path.dirname(os.path.abspath(args.config_path))
    config_name = os.path.basename(args.config_path).replace('.yaml', '')
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    
    # Override config with command line arguments
    with open_dict(cfg):
        if args.gpus is not None:
            cfg.ngpus = args.gpus
        if args.max_steps is not None:
            cfg.training.n_iters = args.max_steps
        if args.val_check_interval is not None:
            cfg.training.eval_freq = args.val_check_interval
    
    # Setup working directory
    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = f"model_zoo/{args.dataset}/lightning_runs/{args.arch}"
    
    makedirs(work_dir)
    
    with open_dict(cfg):
        cfg.work_dir = work_dir
        cfg.dataset_name = args.dataset
        # Architecture is already correctly set by loading the arch-specific config
        # Verify it matches the command line argument
        expected_arch = "convolutional" if args.arch == "Conv" else "transformer"
        if not hasattr(cfg.model, 'architecture') or cfg.model.architecture != expected_arch:
            print(f"WARNING: Config architecture mismatch. Setting to {expected_arch}")
            cfg.model.architecture = expected_arch
    
    print(f"Working directory: {work_dir}")
    print(f"Training {args.dataset} with {args.arch} architecture")
    print(f"Config: {cfg}")
    
    # Set random seed for reproducibility
    if hasattr(cfg, 'seed'):
        pl.seed_everything(cfg.seed)
    else:
        pl.seed_everything(42)
    
    # Initialize model and data module using factory pattern
    model = create_lightning_module(cfg, dataset_name=args.dataset, config_path=args.config_path)
    data_module = D3DataModule(cfg, dataset_name=args.dataset)
    
    # Load from checkpoint if specified
    resume_from_checkpoint = None
    if args.resume_from:
        if os.path.exists(args.resume_from):
            model, original_step = load_from_checkpoint(args.resume_from, model, cfg, args.dataset)
            if original_step:
                print(f"Resuming from step: {original_step}")
            # For Lightning checkpoints, we'll use the built-in resume mechanism
            if not is_original_checkpoint(args.resume_from):
                resume_from_checkpoint = args.resume_from
        else:
            print(f"Warning: Checkpoint {args.resume_from} not found")
    
    # Setup loggers and callbacks
    loggers = setup_logging(cfg, work_dir)
    callbacks = setup_callbacks(cfg, work_dir, args.dataset)
    
    # Create trainer
    trainer_kwargs = {
        'logger': loggers,
        'callbacks': callbacks,
        'default_root_dir': work_dir,
    }
    
    if args.fast_dev_run:
        trainer_kwargs['fast_dev_run'] = True
    
    if resume_from_checkpoint:
        trainer_kwargs['ckpt_path'] = resume_from_checkpoint
    
    trainer = create_trainer_from_config(cfg, args.dataset, **trainer_kwargs)
    
    # Print training info
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Architecture: {args.arch}")
    print(f"Max steps: {cfg.training.n_iters}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Accumulate grad batches: {cfg.training.accum}")
    print(f"Learning rate: {cfg.optim.lr}")
    print(f"GPUs: {cfg.ngpus}")
    print(f"Working directory: {work_dir}")
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
    print("="*50 + "\n")
    
    if args.dry_run:
        print("Dry run mode - setup complete, not running training")
        return
    
    # Run training
    try:
        trainer.fit(model, data_module)
        print("Training completed successfully!")
        
        # Save final model info
        final_checkpoint = trainer.checkpoint_callback.best_model_path
        print(f"Best model saved at: {final_checkpoint}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()