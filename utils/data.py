import re
# from datasets import load_dataset
from itertools import chain
import numpy as np
import torch

import urllib.request
import zipfile
import requests
import json
import h5py, os
# from datasets import Dataset

'''
Please follow codes from "Dirichlet-flow-matching" (https://github.com/HannesStark/dirichlet-flow-matching) and "Dirichlet diffusion score model" (https://github.com/jzhoulab/ddsm)
for setting up the code to train for Promoter dataset and run import by uncommenting below line.
'''
# from promoter_dataset import PromoterDataset


class PromoterDataset(torch.utils.data.Dataset):
    """
    Basic PromoterDataset implementation for D3-DNA training.
    
    This is a simplified implementation that generates synthetic promoter data
    with the correct format (sequence + target concatenated).
    
    For production use, replace this with actual promoter data loading
    from the Dirichlet-flow-matching or DDSM repositories.
    """
    
    def __init__(self, n_tsses=100000, rand_offset=10, split='train', seq_length=1024):
        self.n_tsses = n_tsses
        self.rand_offset = rand_offset
        self.split = split
        self.seq_length = seq_length
        
        # Generate synthetic data for now
        # In practice, this should load real promoter sequences and targets
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic promoter data with correct format."""
        torch.manual_seed(42 if self.split == 'train' else 123)
        
        # Generate random one-hot sequences (batch_size, seq_length, 4)
        sequences = torch.randint(0, 4, (self.n_tsses, self.seq_length))
        sequences_one_hot = torch.nn.functional.one_hot(sequences, num_classes=4).float()
        
        # Generate random targets (batch_size, seq_length, 1)
        targets = torch.randn(self.n_tsses, self.seq_length, 1)
        
        # Concatenate sequence and target: (batch_size, seq_length, 5)
        data = torch.cat([sequences_one_hot, targets], dim=-1)
        
        return data
    
    def __len__(self):
        return self.n_tsses
    
    def __getitem__(self, idx):
        return self.data[idx]

from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


# Placeholder implementations for SEI model dependencies
# In production, these should be replaced with actual SEI model implementations
# from the Dirichlet-flow-matching or DDSM repositories

class Sei(torch.nn.Module):
    """
    Placeholder SEI (Sequence-to-Expression and Interaction) model.
    
    This is a simplified implementation for testing purposes.
    For production use, replace with the actual SEI model from:
    https://github.com/FunctionLab/selene
    """
    
    def __init__(self, input_size=4096, output_size=21907):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Simple placeholder network
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(4, 64, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = torch.nn.Linear(128, output_size)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 4, seq_length)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # Remove last dimension
        x = self.classifier(x)
        return x


class NonStrandSpecific(torch.nn.Module):
    """
    Placeholder NonStrandSpecific wrapper for SEI model.
    
    This is a simplified implementation for testing purposes.
    For production use, replace with the actual implementation from selene_sdk.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def load_state_dict(self, state_dict, strict=True):
        # For the placeholder, just initialize with random weights
        return super().load_state_dict(state_dict, strict=False)


def upgrade_state_dict(state_dict, prefixes=None):
    """
    Utility function to upgrade state dict format.
    
    This is a placeholder implementation.
    """
    if prefixes is None:
        return state_dict
    
    upgraded_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                break
        upgraded_dict[new_key] = value
    
    return upgraded_dict


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def get_datasets(dataset='deepstarr'):
    """
    Get train and validation datasets without DataLoader wrapping.
    Used by Lightning DataModule.
    """
    if dataset.lower() == 'deepstarr':
        # Deepstarr
        filepath = os.path.join('model_zoo', 'deepstarr', 'DeepSTARR_data.h5')
        data = h5py.File(filepath, 'r')
        X_train = torch.tensor(np.array(data['X_train']))
        y_train = torch.tensor(np.array(data['Y_train']))
        X_train = torch.argmax(X_train, dim=1)
        X_valid = torch.tensor(np.array(data['X_valid']))
        y_valid = torch.tensor(np.array(data['Y_valid']))
        X_valid = torch.argmax(X_valid, dim=1)
        train_set = TensorDataset(X_train, y_train)
        valid_set = TensorDataset(X_valid, y_valid)
        
    elif dataset.lower() == 'mpra':
        # MPRA
        filepath = os.path.join('model_zoo', 'mpra', 'mpra_data.h5')
        dataset_file = h5py.File(filepath, 'r')
        x_train = torch.tensor(np.array(dataset_file['x_train']).astype(np.float32)).permute(0,2,1)
        x_train = torch.argmax(x_train, dim=1)
        y_train = torch.tensor(np.array(dataset_file['y_train']).astype(np.float32))
        x_valid = torch.tensor(np.array(dataset_file['x_valid']).astype(np.float32)).permute(0,2,1)
        x_valid = torch.argmax(x_valid, dim=1)
        y_valid = torch.tensor(np.array(dataset_file['y_valid']).astype(np.float32))
        train_set = TensorDataset(x_train, y_train)
        valid_set = TensorDataset(x_valid, y_valid)

    elif dataset.lower() == 'atacseq':
        # MPRA
        filepath = os.path.join('model_zoo', 'atacseq', 'atacseq_data.h5')
        dataset_file = h5py.File(filepath, 'r')
        x_train = torch.tensor(np.array(dataset_file['x_train']).astype(np.float32)).permute(0,2,1)
        x_train = torch.argmax(x_train, dim=1)
        y_train = torch.tensor(np.array(dataset_file['y_train']).astype(np.float32))
        x_valid = torch.tensor(np.array(dataset_file['x_valid']).astype(np.float32)).permute(0,2,1)
        x_valid = torch.argmax(x_valid, dim=1)
        y_valid = torch.tensor(np.array(dataset_file['y_valid']).astype(np.float32))
        train_set = TensorDataset(x_train, y_train)
        valid_set = TensorDataset(x_valid, y_valid)
        
        
    elif dataset.lower() == 'promoter':
        # Promoter - use dataset objects directly
        train_set = PromoterDataset(n_tsses=100000, rand_offset=10, split='train')
        valid_set = PromoterDataset(n_tsses=100000, rand_offset=0, split='test')
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported datasets: 'deepstarr', 'mpra', 'promoter', 'atacseq'")

    return train_set, valid_set


def get_dataloaders(config, distributed=True, dataset='deepstarr'):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    # Use the shared dataset loading function
    train_set, valid_set = get_datasets(dataset)

    print (len(train_set), len(valid_set))

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))


    return train_loader, valid_loader

