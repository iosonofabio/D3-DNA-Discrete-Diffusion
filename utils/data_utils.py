"""
Shared Data Utilities for D3-DNA Discrete Diffusion

This module contains shared data utilities that are used across all datasets,
including the cycle_loader function and other common data processing utilities.
"""

import numpy as np
import torch


def cycle_loader(dataloader, sampler=None):
    """
    Create an infinite iterator from a DataLoader.
    
    This function creates an infinite loop over the dataloader, which is useful
    for training where you want to iterate indefinitely without worrying about
    epoch boundaries.
    
    Args:
        dataloader: PyTorch DataLoader to cycle through
        sampler: Optional distributed sampler for multi-GPU training
        
    Yields:
        Batches from the dataloader infinitely
    """
    while True:
        if sampler is not None:
            # Set random epoch for distributed training
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def collate_fn_generic(batch):
    """
    Generic collate function for batching data.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    if isinstance(batch[0], (list, tuple)):
        # Handle (input, target) pairs
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        inputs = torch.stack(inputs) if isinstance(inputs[0], torch.Tensor) else torch.tensor(inputs)
        targets = torch.stack(targets) if isinstance(targets[0], torch.Tensor) else torch.tensor(targets)
        
        return inputs, targets
    else:
        # Handle single tensors
        return torch.stack(batch) if isinstance(batch[0], torch.Tensor) else torch.tensor(batch)


def compute_dataset_stats(dataset):
    """
    Compute basic statistics for a dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        Dictionary of dataset statistics
    """
    stats = {
        'num_samples': len(dataset),
        'sample_shape': None,
        'data_type': None
    }
    
    if len(dataset) > 0:
        sample = dataset[0]
        if isinstance(sample, (list, tuple)):
            stats['sample_shape'] = [item.shape if hasattr(item, 'shape') else len(item) for item in sample]
            stats['data_type'] = [type(item).__name__ for item in sample]
        else:
            stats['sample_shape'] = sample.shape if hasattr(sample, 'shape') else len(sample)
            stats['data_type'] = type(sample).__name__
    
    return stats


def create_sequence_mask(sequences, pad_token=None):
    """
    Create attention mask for sequences (useful for transformer models).
    
    Args:
        sequences: Tensor of shape (batch_size, seq_length)
        pad_token: Token used for padding (if None, no masking)
        
    Returns:
        Boolean mask of shape (batch_size, seq_length)
    """
    if pad_token is None:
        # No padding, all positions are valid
        return torch.ones_like(sequences, dtype=torch.bool)
    else:
        # Mask out padding tokens
        return sequences != pad_token


def one_hot_encode_sequences(sequences, num_classes=4):
    """
    Convert token sequences to one-hot encoding.
    
    Args:
        sequences: Token sequences of shape (batch_size, seq_length)
        num_classes: Number of possible tokens (default 4 for DNA: A, C, G, T)
        
    Returns:
        One-hot encoded sequences of shape (batch_size, seq_length, num_classes)
    """
    return torch.nn.functional.one_hot(sequences, num_classes=num_classes).float()


def sequences_to_strings(sequences, token_to_char=None):
    """
    Convert token sequences to strings.
    
    Args:
        sequences: Token sequences of shape (batch_size, seq_length)
        token_to_char: Dictionary mapping tokens to characters
        
    Returns:
        List of sequence strings
    """
    if token_to_char is None:
        # Default DNA mapping
        token_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    sequences_str = []
    for seq in sequences:
        seq_str = ''.join([token_to_char.get(token.item(), 'N') for token in seq])
        sequences_str.append(seq_str)
    
    return sequences_str


def calculate_gc_content(sequences):
    """
    Calculate GC content for DNA sequences.
    
    Args:
        sequences: Token sequences where 0=A, 1=C, 2=G, 3=T
        
    Returns:
        Tensor of GC content ratios for each sequence
    """
    # Count G (token 2) and C (token 1)
    gc_counts = ((sequences == 1) | (sequences == 2)).sum(dim=1).float()
    total_length = sequences.shape[1]
    
    return gc_counts / total_length


def reverse_complement(sequences):
    """
    Generate reverse complement of DNA sequences.
    
    Args:
        sequences: Token sequences where 0=A, 1=C, 2=G, 3=T
        
    Returns:
        Reverse complement sequences
    """
    # Complement mapping: A<->T (0<->3), C<->G (1<->2)
    complement_map = torch.tensor([3, 2, 1, 0], device=sequences.device)
    
    # Apply complement mapping
    complement_sequences = complement_map[sequences]
    
    # Reverse the sequences
    reverse_complement_sequences = torch.flip(complement_sequences, dims=[1])
    
    return reverse_complement_sequences