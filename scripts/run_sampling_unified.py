import torch
import argparse
import sys
import os

from utils import data
from utils.load_model import load_model_local
import torch.nn.functional as F
from scripts import sampling
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def get_sequence_length(dataset):
    """Get sequence length for dataset"""
    lengths = {
        'deepstarr': 249,
        'mpra': 200,
        'promoter': 1024
    }
    return lengths.get(dataset.lower(), 249)


def load_conditioning_data(dataset, data_path, split='test', batch_size=256):
    """Load conditioning data for guided sampling"""
    if dataset.lower() == 'deepstarr':
        data_file = h5py.File(data_path, 'r')
        if split == 'test':
            y_data = torch.tensor(np.array(data_file['Y_test']))
        elif split == 'valid':
            y_data = torch.tensor(np.array(data_file['Y_valid']))
        else:
            y_data = torch.tensor(np.array(data_file['Y_train']))
        return DataLoader(TensorDataset(y_data), batch_size=batch_size, shuffle=False)
        
    elif dataset.lower() == 'mpra':
        data_file = h5py.File(data_path, 'r')
        if split == 'test':
            y_data = torch.tensor(np.array(data_file['y_test']).astype(np.float32))
        elif split == 'valid':
            y_data = torch.tensor(np.array(data_file['y_valid']).astype(np.float32))
        else:
            y_data = torch.tensor(np.array(data_file['y_train']).astype(np.float32))
        return DataLoader(TensorDataset(y_data), batch_size=batch_size, shuffle=False)
        
    elif dataset.lower() == 'promoter':
        # For promoter, conditioning might be different
        # This would need to be implemented based on the specific requirements
        raise NotImplementedError("Promoter conditioning data loading not implemented")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser(description="Generate samples using trained diffusion model")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=['deepstarr', 'mpra', 'promoter'],
                       help="Dataset to generate samples for")
    parser.add_argument("--arch", type=str, required=True,
                       choices=['Conv', 'Tran'],
                       help="Model architecture: Conv (Convolutional) or Tran (Transformer)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to data file for conditioning (auto-resolved if not provided)")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for sampling")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Total number of samples to generate")
    parser.add_argument("--steps", type=int, default=None,
                       help="Number of sampling steps (defaults to sequence length)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save samples (defaults to model_path)")
    parser.add_argument("--conditioning", type=str, default=None,
                       choices=['test', 'valid', 'train'],
                       help="Use conditioning data from specified split")
    parser.add_argument("--unconditional", action='store_true',
                       help="Generate unconditional samples")
    args = parser.parse_args()

    # Auto-resolve data path if not provided and conditioning is requested
    if args.conditioning and args.data_path is None:
        data_files = {
            'deepstarr': 'DeepSTARR_data.h5',
            'mpra': 'mpra_data.h5',
            'promoter': 'promoter_data.h5'  # Update this if different
        }
        args.data_path = data_files[args.dataset]
        print(f"Using auto-resolved data path for conditioning: {args.data_path}")

    # Set defaults
    if args.steps is None:
        args.steps = get_sequence_length(args.dataset)
    if args.output_dir is None:
        args.output_dir = args.model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Generating samples for {args.dataset} with {args.arch} architecture")
    
    # Load the diffusion model
    model, graph, noise = load_model_local(args.model_path, device)
    
    # Get sequence length
    seq_length = get_sequence_length(args.dataset)
    
    # Setup conditioning if specified
    conditioning_loader = None
    if args.conditioning and args.data_path:
        conditioning_loader = load_conditioning_data(args.dataset, args.data_path, 
                                                   args.conditioning, args.batch_size)
        print(f"Using {args.conditioning} split for conditioning")
    elif args.unconditional:
        print("Generating unconditional samples")
    else:
        print("Generating samples with zero conditioning")
    
    # Initialize sampling function
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, seq_length), 'analytic', args.steps, device=device
    )
    
    generated_samples = []
    conditioning_values = []
    
    print(f"Generating {args.num_samples} samples for {args.dataset} dataset...")
    
    samples_generated = 0
    batch_idx = 0
    
    while samples_generated < args.num_samples:
        # Determine batch size for this iteration
        current_batch_size = min(args.batch_size, args.num_samples - samples_generated)
        
        if current_batch_size != args.batch_size:
            # Adjust sampling function for last batch
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (current_batch_size, seq_length), 'analytic', args.steps, device=device
            )
        
        # Get conditioning if available
        if conditioning_loader:
            try:
                if batch_idx == 0:
                    conditioning_iter = iter(conditioning_loader)
                conditioning_batch = next(conditioning_iter)[0]  # Get first element of batch
                if conditioning_batch.shape[0] < current_batch_size:
                    # If we've exhausted conditioning data, restart
                    conditioning_iter = iter(conditioning_loader)
                    conditioning_batch = next(conditioning_iter)[0]
                conditioning_target = conditioning_batch[:current_batch_size].to(device)
            except StopIteration:
                # Restart conditioning data if exhausted
                conditioning_iter = iter(conditioning_loader)
                conditioning_batch = next(conditioning_iter)[0]
                conditioning_target = conditioning_batch[:current_batch_size].to(device)
        elif args.unconditional:
            # Generate without conditioning (this might need model-specific implementation)
            conditioning_target = None
        else:
            # Use zero conditioning
            if args.dataset.lower() == 'deepstarr':
                conditioning_target = torch.zeros((current_batch_size, 2)).to(device)
            elif args.dataset.lower() == 'mpra':
                conditioning_target = torch.zeros((current_batch_size, 3)).to(device)
            elif args.dataset.lower() == 'promoter':
                conditioning_target = torch.zeros((current_batch_size, 1)).to(device)
        
        # Generate samples
        if conditioning_target is not None:
            sample = sampling_fn(model, conditioning_target)
            conditioning_values.append(conditioning_target.cpu())
        else:
            # For unconditional sampling, this might need modification based on model implementation
            sample = sampling_fn(model, torch.zeros((current_batch_size, 1)).to(device))
        
        # Convert to one-hot encoding
        seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()
        generated_samples.append(seq_pred_one_hot.cpu())
        
        samples_generated += current_batch_size
        batch_idx += 1
        
        if batch_idx % 10 == 0:
            print(f"Generated {samples_generated}/{args.num_samples} samples")
    
    # Concatenate all samples
    all_samples = torch.cat(generated_samples, dim=0)
    print(f"Generated {all_samples.shape[0]} samples of shape {all_samples.shape}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.conditioning:
        output_filename = f"samples_{args.dataset}_{args.conditioning}_conditioned.npz"
        all_conditioning = torch.cat(conditioning_values, dim=0) if conditioning_values else None
    elif args.unconditional:
        output_filename = f"samples_{args.dataset}_unconditional.npz"
        all_conditioning = None
    else:
        output_filename = f"samples_{args.dataset}_zero_conditioned.npz"
        all_conditioning = torch.cat(conditioning_values, dim=0) if conditioning_values else None
    
    output_path = os.path.join(args.output_dir, output_filename)
    
    save_dict = {
        'generated_sequences': all_samples.numpy(),
        'dataset': args.dataset,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'steps': args.steps,
        'sequence_length': seq_length
    }
    
    if all_conditioning is not None:
        save_dict['conditioning_values'] = all_conditioning.numpy()
    
    np.savez(output_path, **save_dict)
    
    print(f"Samples saved to: {output_path}")
    print(f"Sample shape: {all_samples.shape}")


if __name__ == "__main__":
    main()