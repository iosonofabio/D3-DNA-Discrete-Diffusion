"""
Promoter Dataset Loader

This module provides dataset loading functionality specific to the Promoter dataset.
It handles loading promoter sequence data and provides appropriate preprocessing for D3 training.

For production use, this should integrate with the Dirichlet-flow-matching or DDSM repositories:
- https://github.com/HannesStark/dirichlet-flow-matching
- https://github.com/jzhoulab/ddsm
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Tuple, Optional
from utils.data_utils import cycle_loader


class PromoterDataset(Dataset):
    """
    Promoter dataset for D3-DNA training.
    
    This implementation provides synthetic promoter data with the correct format
    (sequence + target concatenated) for development purposes.
    
    For production use, replace this with actual promoter data loading
    from the Dirichlet-flow-matching or DDSM repositories.
    """
    
    def __init__(self, n_tsses: int = 100000, rand_offset: int = 10, 
                 split: str = 'train', seq_length: int = 1024):
        """
        Initialize the Promoter dataset.
        
        Args:
            n_tsses: Number of transcription start sites (samples)
            rand_offset: Random offset for data generation
            split: Dataset split ('train', 'test', 'valid')
            seq_length: Length of sequences
        """
        self.n_tsses = n_tsses
        self.rand_offset = rand_offset
        self.split = split
        self.seq_length = seq_length
        
        # Generate synthetic data for now
        # In practice, this should load real promoter sequences and targets
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic promoter data with correct format."""
        # Use different seeds for different splits for reproducibility
        seed_map = {'train': 42, 'test': 123, 'valid': 456}
        torch.manual_seed(seed_map.get(self.split, 42))
        
        # Generate random one-hot sequences (batch_size, seq_length, 4)
        sequences = torch.randint(0, 4, (self.n_tsses, self.seq_length))
        sequences_one_hot = torch.nn.functional.one_hot(sequences, num_classes=4).float()
        
        # Generate random targets (batch_size, seq_length, 1)
        # These represent regulatory activity or expression levels
        targets = torch.randn(self.n_tsses, self.seq_length, 1)
        
        # Normalize targets to a reasonable range
        targets = torch.sigmoid(targets)  # Values between 0 and 1
        
        # Concatenate sequence and target: (batch_size, seq_length, 5)
        # This is the expected format for promoter data in D3
        data = torch.cat([sequences_one_hot, targets], dim=-1)
        
        return data
    
    def __len__(self) -> int:
        return self.n_tsses
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample.
        
        Returns:
            Tensor of shape (seq_length, 5) where the last dimension contains
            [A, T, G, C, target_value] for each position
        """
        return self.data[idx]


def get_promoter_datasets(n_train: int = 100000, n_valid: int = 10000, 
                         seq_length: int = 1024) -> Tuple[Dataset, Dataset]:
    """
    Get Promoter train and validation datasets.
    
    Args:
        n_train: Number of training samples
        n_valid: Number of validation samples
        seq_length: Length of sequences
        
    Returns:
        Tuple of (train_dataset, valid_dataset)
    """
    train_set = PromoterDataset(
        n_tsses=n_train, 
        rand_offset=10, 
        split='train', 
        seq_length=seq_length
    )
    valid_set = PromoterDataset(
        n_tsses=n_valid, 
        rand_offset=0, 
        split='valid', 
        seq_length=seq_length
    )
    
    return train_set, valid_set


def get_promoter_dataloaders(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get Promoter dataloaders for training and validation.
    
    Args:
        config: Configuration object with training parameters
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, valid_loader)
    """
    # Validation checks
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Train Batch Size {config.training.batch_size} is not divisible by "
            f"{config.ngpus} gpus with accumulation {config.training.accum}."
        )
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Eval Batch Size {config.eval.batch_size} is not divisible by "
            f"{config.ngpus} gpus with accumulation {config.training.accum}."
        )
    
    # Get dataset configuration from config if available
    seq_length = getattr(config.dataset, 'sequence_length', 1024)
    n_train = getattr(config.data, 'n_train_samples', 100000)
    n_valid = getattr(config.data, 'n_valid_samples', 10000)
    
    # Get datasets
    train_set, valid_set = get_promoter_datasets(n_train, n_valid, seq_length)
    
    print(f"Promoter dataset sizes - Train: {len(train_set)}, Valid: {len(valid_set)}")
    
    # Setup samplers
    if distributed:
        train_sampler = DistributedSampler(train_set)
        valid_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        valid_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    )
    
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    
    return train_loader, valid_loader


def get_promoter_dataloaders_with_cycle(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get Promoter dataloaders with cycle_loader applied for training and validation.
    
    Args:
        config: Configuration object with training parameters
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, valid_loader) with cycle_loader applied
    """
    train_loader, valid_loader = get_promoter_dataloaders(config, distributed)
    
    # Apply cycle_loader
    train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None
    valid_sampler = valid_loader.sampler if hasattr(valid_loader, 'sampler') else None
    
    cycled_train_loader = cycle_loader(train_loader, train_sampler)
    cycled_valid_loader = cycle_loader(valid_loader, valid_sampler)
    
    return cycled_train_loader, cycled_valid_loader


# =============================================================================
# Integration Notes for Production Use
# =============================================================================
"""
To integrate with actual promoter data from research repositories:

1. Replace PromoterDataset._generate_synthetic_data() with real data loading:
   - Load actual promoter sequences from FASTA/BED files
   - Load corresponding expression/activity targets
   - Apply proper sequence preprocessing (padding, truncation, etc.)

2. Add support for different promoter dataset variants:
   - Human promoters
   - Mouse promoters  
   - Cell-type specific promoters
   - Tissue-specific expression data

3. Implement proper data splits:
   - Use predefined train/val/test splits from the research community
   - Ensure no data leakage between splits
   - Handle sequence similarity clustering if needed

4. Add data augmentation:
   - Reverse complement augmentation
   - Random cropping/sliding windows
   - Noise injection for robustness

5. Integrate with Dirichlet Flow Matching:
   - Import PromoterDataset from dirichlet-flow-matching
   - Use their preprocessing pipeline
   - Match their data format expectations

Example integration:
```python
try:
    from promoter_dataset import PromoterDataset as RealPromoterDataset
    PromoterDataset = RealPromoterDataset
except ImportError:
    # Fall back to synthetic data
    pass
```
"""