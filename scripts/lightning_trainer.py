import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from itertools import chain
import os
import sys
import datetime
from typing import Any, Dict, Optional

from model import SEDD
from model.ema import ExponentialMovingAverage
from utils import losses
from utils import graph_lib
from utils import noise_lib
from utils.utils import get_score_fn


class D3LightningModule(pl.LightningModule):
    """PyTorch Lightning module for D3 DNA Discrete Diffusion model."""
    
    def __init__(self, cfg, dataset_name: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.dataset_name = dataset_name
        
        # Build core components
        self.score_model = SEDD(cfg)
        self.graph = None  # Will be initialized in setup()
        self.noise = None  # Will be initialized in setup()
        
        # EMA setup
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(), 
            decay=cfg.training.ema
        )
        
        # Loss function will be set up in setup()
        self.loss_fn = None
        self.sampling_eps = 1e-5
        
        # Accumulation setup
        self.accum_iter = 0
        self.total_loss = 0
        
    def setup(self, stage: str = None):
        """Setup method called after the model is moved to device."""
        # Initialize graph and noise on the correct device
        self.graph = graph_lib.get_graph(self.cfg, self.device)
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
        
        # Move EMA shadow parameters to the correct device (important for distributed training)
        if hasattr(self, 'ema') and hasattr(self.ema, 'shadow_params'):
            for i, shadow_param in enumerate(self.ema.shadow_params):
                self.ema.shadow_params[i] = shadow_param.to(self.device)
        
        # Setup loss function
        self.loss_fn = losses.get_loss_fn(
            self.noise, self.graph, train=True, sampling_eps=self.sampling_eps
        )
        
    def training_step(self, batch, batch_idx):
        """Training step with gradient accumulation support."""
        # Handle dataset-specific input processing
        if self.dataset_name and self.dataset_name.lower() == 'promoter':
            seq_one_hot = batch[:, :, :4]
            inputs = torch.argmax(seq_one_hot, dim=-1)
            target = batch[:, :, 4:5]
        else:
            # For DeepSTARR and MPRA
            inputs, target = batch
            
        # Compute loss
        loss = self.loss_fn(self.score_model, inputs, target).mean()
        loss = loss / self.cfg.training.accum
        
        # Accumulation logic
        self.accum_iter += 1
        self.total_loss += loss.detach()
        
        if self.accum_iter == self.cfg.training.accum:
            self.accum_iter = 0
            # Update EMA
            self.ema.update(self.score_model.parameters())
            
            # Log the accumulated loss (detached for logging)
            accumulated_loss_log = self.total_loss
            self.log('train_loss', accumulated_loss_log, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.total_loss = 0
            
            # Return the current loss (with gradients) for Lightning to handle backward
            return loss
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step using EMA weights."""
        # Handle dataset-specific input processing
        if self.dataset_name and self.dataset_name.lower() == 'promoter':
            seq_one_hot = batch[:, :, :4]
            inputs = torch.argmax(seq_one_hot, dim=-1)
            target = batch[:, :, 4:5]
        else:
            # For DeepSTARR and MPRA
            inputs, target = batch
            
        # Use EMA weights for validation
        self.ema.store(self.score_model.parameters())
        self.ema.copy_to(self.score_model.parameters())
        
        # Setup eval loss function
        eval_loss_fn = losses.get_loss_fn(
            self.noise, self.graph, train=False, sampling_eps=self.sampling_eps
        )
        
        with torch.no_grad():
            loss = eval_loss_fn(self.score_model, inputs, target).mean()
            
        # Restore original weights
        self.ema.restore(self.score_model.parameters())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        params = chain(self.score_model.parameters(), self.noise.parameters())
        optimizer = losses.get_optimizer(self.cfg, params)
        
        # Setup warmup scheduler if specified
        if self.cfg.optim.warmup > 0:
            def lr_lambda(step):
                if step < self.cfg.optim.warmup:
                    return step / self.cfg.optim.warmup
                return 1.0
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping before optimizer step."""
        if self.cfg.optim.grad_clip >= 0:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=self.cfg.optim.grad_clip, 
                gradient_clip_algorithm="norm"
            )
    
    def load_from_original_checkpoint(self, checkpoint_path: str):
        """Load weights from original D3 .pth checkpoint format."""
        print(f"Loading original checkpoint from: {checkpoint_path}")
        
        # Load the original checkpoint
        loaded_state = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights (handle DDP wrapper)
        if 'model' in loaded_state:
            model_state = loaded_state['model']
            self.score_model.load_state_dict(model_state, strict=False)
            print("✓ Loaded model weights from original checkpoint")
        
        # Load EMA weights
        if 'ema' in loaded_state:
            self.ema.load_state_dict(loaded_state['ema'])
            print("✓ Loaded EMA weights from original checkpoint")
        
        # Load step counter for logging
        if 'step' in loaded_state:
            print(f"✓ Original checkpoint was at step: {loaded_state['step']}")
            
        return loaded_state.get('step', 0)
    
    def state_dict(self):
        """Override to include EMA state in Lightning checkpoints."""
        # Get the default Lightning state dict
        state = super().state_dict()
        
        # Add EMA state with proper prefixes
        if hasattr(self, 'ema') and self.ema is not None:
            ema_state = self.ema.state_dict()
            for key, value in ema_state.items():
                state[f'ema.{key}'] = value
        
        return state
    
    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Override to handle both Lightning and original checkpoint formats."""
        # Check if this is an original D3 checkpoint format
        if 'model' in state_dict and 'ema' in state_dict and 'step' in state_dict:
            # This is an original checkpoint, use our custom loader
            step = self.load_from_original_checkpoint_dict(state_dict)
            return step
        else:
            # This is a Lightning checkpoint
            # Separate EMA state from model state
            model_state = {}
            ema_state = {}
            
            for key, value in state_dict.items():
                if key.startswith('ema.'):
                    ema_key = key.replace('ema.', '')
                    ema_state[ema_key] = value
                else:
                    model_state[key] = value
            
            # Load model state using parent method
            result = super().load_state_dict(model_state, strict=False)
            
            # Load EMA state separately if it exists
            if ema_state and hasattr(self, 'ema'):
                try:
                    self.ema.load_state_dict(ema_state)
                    print("✓ Loaded EMA state from Lightning checkpoint")
                except Exception as e:
                    print(f"⚠ Could not load EMA state: {e}")
            
            return result
    
    def load_from_original_checkpoint_dict(self, state_dict: dict):
        """Load from original checkpoint dictionary."""
        # Load model weights
        if 'model' in state_dict:
            self.score_model.load_state_dict(state_dict['model'], strict=False)
        
        # Load EMA weights  
        if 'ema' in state_dict:
            self.ema.load_state_dict(state_dict['ema'])
            
        return state_dict.get('step', 0)


class D3DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for D3 datasets."""
    
    def __init__(self, cfg, dataset_name: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        
    def setup(self, stage: str = None):
        """Setup train and validation datasets."""
        from utils import data
        
        # Use dataset loading function (not dataloaders)
        if self.dataset_name:
            self.train_ds, self.val_ds = data.get_datasets(dataset=self.dataset_name)
        else:
            self.train_ds, self.val_ds = data.get_datasets()
    
    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum),
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.eval.batch_size // (self.cfg.ngpus * self.cfg.training.accum),
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )


# Model Factory Pattern for Dataset-Specific Implementations

def get_model_class_for_dataset(dataset: str, config_path: Optional[str] = None):
    """Get the appropriate SEDD model class for the given dataset."""
    if dataset.lower() == 'promoter':
        # Import promoter-specific SEDD
        sys.path.insert(0, 'model_zoo/promoter')
        try:
            from transformer_promoter import SEDD
            return SEDD
        finally:
            sys.path.pop(0)
    elif dataset.lower() == 'mpra':
        # Import MPRA-specific SEDD
        sys.path.insert(0, 'model_zoo/mpra')
        try:
            from transformer_mpra import SEDD
            return SEDD
        finally:
            sys.path.pop(0)
    else:
        # Use generic SEDD for deepstarr and others
        from model import SEDD
        return SEDD


def create_lightning_module(cfg, dataset_name: Optional[str] = None, config_path: Optional[str] = None):
    """Factory function to create the appropriate Lightning module for the dataset."""
    if dataset_name is None:
        return D3LightningModule(cfg, dataset_name)
    
    dataset_lower = dataset_name.lower()
    
    if dataset_lower == 'promoter':
        return PromoterD3LightningModule(cfg)
    elif dataset_lower == 'mpra':
        return MPRAD3LightningModule(cfg)
    else:
        return D3LightningModule(cfg, dataset_name)


class PromoterD3LightningModule(D3LightningModule):
    """Lightning module specifically for Promoter dataset using promoter-specific SEDD."""
    
    def __init__(self, cfg):
        # Don't call super().__init__ as it will create the wrong model
        # Instead, initialize the LightningModule directly
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.cfg = cfg
        self.dataset_name = 'promoter'
        
        # Use promoter-specific SEDD model
        SEDD_class = get_model_class_for_dataset('promoter')
        self.score_model = SEDD_class(cfg)
        
        # Initialize other components
        self.graph = None  # Will be initialized in setup()
        self.noise = None  # Will be initialized in setup()
        
        # EMA setup
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(), 
            decay=cfg.training.ema
        )
        
        # Loss function will be set up in setup()
        self.loss_fn = None
        self.sampling_eps = 1e-5
        
        # Accumulation setup
        self.accum_iter = 0
        self.total_loss = 0
        
        print("✓ Initialized PromoterD3LightningModule with promoter-specific SEDD")


class MPRAD3LightningModule(D3LightningModule):
    """Lightning module specifically for MPRA dataset using MPRA-specific SEDD."""
    
    def __init__(self, cfg):
        # Don't call super().__init__ as it will create the wrong model
        # Instead, initialize the LightningModule directly
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.cfg = cfg
        self.dataset_name = 'mpra'
        
        # Use MPRA-specific SEDD model
        SEDD_class = get_model_class_for_dataset('mpra')
        self.score_model = SEDD_class(cfg)
        
        # Initialize other components
        self.graph = None  # Will be initialized in setup()
        self.noise = None  # Will be initialized in setup()
        
        # EMA setup
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(), 
            decay=cfg.training.ema
        )
        
        # Loss function will be set up in setup()
        self.loss_fn = None
        self.sampling_eps = 1e-5
        
        # Accumulation setup
        self.accum_iter = 0
        self.total_loss = 0
        
        print("✓ Initialized MPRAD3LightningModule with MPRA-specific SEDD")


def create_trainer_from_config(cfg, dataset_name: Optional[str] = None, **trainer_kwargs):
    """Create a PyTorch Lightning trainer from D3 config."""
    
    # Extract relevant training parameters
    default_trainer_args = {
        'max_steps': cfg.training.n_iters,
        'log_every_n_steps': cfg.training.log_freq,
        'val_check_interval': cfg.training.eval_freq,
        'check_val_every_n_epoch': None,  # Allow step-based validation across epochs
        'accumulate_grad_batches': cfg.training.accum,
        'precision': 'bf16-mixed',  # Use mixed precision like original
        'gradient_clip_val': cfg.optim.grad_clip if cfg.optim.grad_clip >= 0 else None,
        'enable_checkpointing': True,
        'enable_progress_bar': False, #True
        'enable_model_summary': True,
    }
    
    # Handle multi-GPU setup
    if cfg.ngpus > 1:
        default_trainer_args.update({
            'devices': cfg.ngpus,
            'num_nodes': getattr(cfg, 'nnodes', 1),  # Default to 1 node if not specified
            'strategy': 'ddp_find_unused_parameters_true',  # More robust for cluster environments
            'sync_batchnorm': True,
        })
    else:
        default_trainer_args['devices'] = 1
    
    # Setup callbacks
    callbacks = []
    
    # Add SP-MSE validation callback if enabled
    if hasattr(cfg, 'sp_mse_validation') and cfg.sp_mse_validation.enabled:
        from utils.sp_mse_callback import SPMSEValidationCallback
        
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
        callbacks.append(sp_mse_callback)
        print(f"✓ Added SP-MSE validation callback for {dataset_name} dataset")
    
    # Add callbacks to trainer args if any exist
    if callbacks:
        default_trainer_args['callbacks'] = callbacks
    
    # Override with any custom trainer arguments
    default_trainer_args.update(trainer_kwargs)
    
    # Create trainer
    trainer = pl.Trainer(**default_trainer_args)
    
    return trainer