import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Optional, Any
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule, Trainer

from scripts import sampling


class SPMSEValidationCallback(Callback):
    """
    Callback for evaluating generated sequences using oracle models and computing SP-MSE.
    
    This callback generates sequences from the diffusion model during training,
    evaluates them using oracle models, and computes MSE between oracle predictions
    on real vs generated sequences. Saves checkpoints based on best SP-MSE performance.
    """
    
    def __init__(
        self,
        dataset: str,
        oracle_path: str,
        data_path: str,
        validation_freq: int = 5000,
        validation_samples: int = 1000,
        enabled: bool = True,
        sampling_steps: Optional[int] = None,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Args:
            dataset: Dataset name ('deepstarr', 'mpra', 'promoter')
            oracle_path: Path to oracle model checkpoint
            data_path: Path to dataset file
            validation_freq: Frequency in training steps to run SP-MSE validation
            validation_samples: Number of samples to use from validation set (subsampling)
            enabled: Whether SP-MSE validation is enabled
            sampling_steps: Number of sampling steps (defaults to sequence length)
            early_stopping_patience: Stop training if no improvement for N validations
        """
        super().__init__()
        self.dataset = dataset.lower()
        self.oracle_path = oracle_path
        self.data_path = data_path
        self.validation_freq = validation_freq
        self.validation_samples = validation_samples
        self.enabled = enabled
        self.sampling_steps = sampling_steps
        self.early_stopping_patience = early_stopping_patience
        
        # State tracking
        self.oracle_model = None
        self.best_sp_mse = float('inf')
        self.steps_since_improvement = 0
        self.validation_data_cache = None
        
        # Sequence lengths for different datasets
        self.seq_lengths = {
            'deepstarr': 249,
            'mpra': 200,
            'promoter': 1024
        }
        
        if not self.enabled:
            return
            
        # Auto-resolve oracle path if relative
        if not os.path.isabs(self.oracle_path):
            oracle_files = {
                'deepstarr': 'oracle_DeepSTARR_DeepSTARR_data.ckpt',
                'mpra': 'oracle_mpra_mpra_data.ckpt',
                'promoter': 'best.sei.model.pth.tar'
            }
            self.oracle_path = f"model_zoo/{self.dataset}/oracle_models/{oracle_files[self.dataset]}"
        
        # Set default sampling steps
        if self.sampling_steps is None:
            self.sampling_steps = self.seq_lengths.get(self.dataset, 249)
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup callback - load oracle model"""
        if not self.enabled:
            return
            
        if stage == 'fit':
            self.oracle_model = self._load_oracle_model()
            if self.oracle_model is not None:
                self.oracle_model.eval()
                print(f"SP-MSE Callback: Loaded oracle model from {self.oracle_path}")
            else:
                print(f"SP-MSE Callback: Failed to load oracle model, disabling callback")
                self.enabled = False
    
    def on_train_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Check if it's time to run SP-MSE validation"""
        if not self.enabled or self.oracle_model is None:
            return
            
        global_step = trainer.global_step
        
        # Check if it's time for SP-MSE validation
        if global_step > 0 and global_step % self.validation_freq == 0:
            self._run_sp_mse_validation(trainer, pl_module)
    
    def _load_oracle_model(self):
        """Load the appropriate oracle model based on dataset"""
        try:
            if self.dataset == 'deepstarr':
                sys.path.insert(0, 'model_zoo/deepstarr')
                try:
                    from deepstarr import PL_DeepSTARR
                    oracle = PL_DeepSTARR.load_from_checkpoint(
                        self.oracle_path, input_h5_file=self.data_path
                    ).eval()
                    return oracle
                finally:
                    sys.path.pop(0)
                    
            elif self.dataset == 'mpra':
                sys.path.insert(0, 'model_zoo/mpra')
                try:
                    from mpra import PL_mpra
                    oracle = PL_mpra.load_from_checkpoint(
                        self.oracle_path, input_h5_file=self.data_path
                    ).eval()
                    return oracle
                finally:
                    sys.path.pop(0)
                    
            elif self.dataset == 'promoter':
                raise NotImplementedError("Promoter evaluation requires SEI model setup")
            else:
                raise ValueError(f"Unknown dataset: {self.dataset}")
                
        except Exception as e:
            print(f"Failed to load oracle model: {e}")
            return None
    
    def _get_validation_data(self, trainer: Trainer):
        """Get validation data with optional subsampling"""
        if self.validation_data_cache is not None:
            return self.validation_data_cache
            
        # Get validation dataloader
        val_dataloader = trainer.datamodule.val_dataloader()
        
        # Collect all validation data
        all_sequences = []
        all_targets = []
        
        for batch in val_dataloader:
            sequences, targets = batch
            all_sequences.append(sequences)
            all_targets.append(targets)
        
        all_sequences = torch.cat(all_sequences, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Subsample if requested
        if self.validation_samples > 0 and len(all_sequences) > self.validation_samples:
            indices = torch.randperm(len(all_sequences))[:self.validation_samples]
            all_sequences = all_sequences[indices]
            all_targets = all_targets[indices]
        
        self.validation_data_cache = (all_sequences, all_targets)
        return self.validation_data_cache
    
    def _run_sp_mse_validation(self, trainer: Trainer, pl_module: LightningModule):
        """Run SP-MSE validation"""
        device = pl_module.device
        
        # Get validation data
        val_sequences, val_targets = self._get_validation_data(trainer)
        val_targets = val_targets.to(device)
        
        # Get sequence length
        seq_length = self.seq_lengths.get(self.dataset, 249)
        
        # Initialize sampling function
        sampling_fn = sampling.get_pc_sampler(
            pl_module.graph, pl_module.noise, 
            (len(val_sequences), seq_length), 
            'analytic', self.sampling_steps, device=device
        )
        
        # Generate sequences using the diffusion model
        with torch.no_grad():
            # Set model to eval mode temporarily
            was_training = pl_module.training
            pl_module.eval()
            
            # Use EMA parameters if available
            if hasattr(pl_module, 'ema') and pl_module.ema is not None:
                pl_module.ema.store(pl_module.score_model.parameters())
                pl_module.ema.copy_to(pl_module.score_model.parameters())
            
            try:
                # Generate samples
                generated_sequences = sampling_fn(pl_module.score_model, val_targets)
                generated_one_hot = F.one_hot(generated_sequences, num_classes=4).float()
                
                # Get oracle predictions
                val_score = self._get_oracle_predictions(val_sequences, device)
                generated_score = self._get_oracle_predictions_from_one_hot(generated_one_hot, device)
                
                # Calculate SP-MSE
                sp_mse = (val_score - generated_score) ** 2
                mean_sp_mse = torch.mean(sp_mse).cpu().item()
                
                # Log metrics
                trainer.logger.log_metrics({
                    'sp_mse/validation': mean_sp_mse,
                    'sp_mse/best': self.best_sp_mse
                }, step=trainer.global_step)
                
                print(f"Step {trainer.global_step}: SP-MSE = {mean_sp_mse:.6f}, Best = {self.best_sp_mse:.6f}")
                
                # Check if this is the best SP-MSE
                if mean_sp_mse < self.best_sp_mse:
                    self.best_sp_mse = mean_sp_mse
                    self.steps_since_improvement = 0
                    
                    # Save best checkpoint
                    checkpoint_path = os.path.join(
                        trainer.default_root_dir, 
                        f"best_sp_mse_checkpoint_step_{trainer.global_step}.ckpt"
                    )
                    trainer.save_checkpoint(checkpoint_path)
                    print(f"Saved best SP-MSE checkpoint: {checkpoint_path}")
                    
                else:
                    self.steps_since_improvement += 1
                    
                    # Check early stopping
                    if (self.early_stopping_patience is not None and 
                        self.steps_since_improvement >= self.early_stopping_patience):
                        print(f"Early stopping: No SP-MSE improvement for {self.early_stopping_patience} validations")
                        trainer.should_stop = True
                
            finally:
                # Restore EMA parameters if used
                if hasattr(pl_module, 'ema') and pl_module.ema is not None:
                    pl_module.ema.restore(pl_module.score_model.parameters())
                
                # Restore training mode
                if was_training:
                    pl_module.train()
    
    def _get_oracle_predictions(self, sequences, device):
        """Get oracle predictions for sequences"""
        if self.dataset in ['deepstarr', 'mpra']:
            # Convert sequences to one-hot if needed
            if sequences.dtype == torch.long:
                sequences_one_hot = F.one_hot(sequences, num_classes=4).float()
            else:
                sequences_one_hot = sequences
            
            # Get oracle predictions
            return self.oracle_model.predict_custom(sequences_one_hot.permute(0, 2, 1).to(device))
        else:
            raise NotImplementedError(f"Oracle predictions not implemented for dataset: {self.dataset}")
    
    def _get_oracle_predictions_from_one_hot(self, sequences_one_hot, device):
        """Get oracle predictions from one-hot encoded sequences"""
        if self.dataset in ['deepstarr', 'mpra']:
            return self.oracle_model.predict_custom(sequences_one_hot.permute(0, 2, 1).to(device))
        else:
            raise NotImplementedError(f"Oracle predictions not implemented for dataset: {self.dataset}")