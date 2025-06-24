import torch
import argparse
import sys
import os
import importlib.util

from utils import data
from utils.load_model import load_model_local
import torch.nn.functional as F
from scripts import sampling
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def load_oracle_model(dataset, oracle_path, data_path):
    """Load the appropriate oracle model based on dataset"""
    if dataset.lower() == 'deepstarr':
        # Import PL_DeepSTARR from model_zoo
        sys.path.insert(0, 'model_zoo/deepstarr')
        try:
            from deepstarr import PL_DeepSTARR
            oracle = PL_DeepSTARR.load_from_checkpoint(oracle_path, input_h5_file=data_path).eval()
            return oracle
        finally:
            sys.path.pop(0)
    elif dataset.lower() == 'mpra':
        # Import PL_mpra from model_zoo
        sys.path.insert(0, 'model_zoo/mpra')
        try:
            from mpra import PL_mpra
            oracle = PL_mpra.load_from_checkpoint(oracle_path, input_h5_file=data_path).eval()
            return oracle
        finally:
            sys.path.pop(0)
    elif dataset.lower() == 'promoter':
        # For promoter, we would need to load SEI model
        # This requires additional setup as mentioned in the original code
        raise NotImplementedError("Promoter evaluation requires SEI model setup")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_test_data(dataset, data_path, batch_size):
    """Load test data based on dataset"""
    if dataset.lower() == 'deepstarr':
        data_file = h5py.File(data_path, 'r')
        X_test = torch.tensor(np.array(data_file['X_test']))
        y_test = torch.tensor(np.array(data_file['Y_test']))
        X_test = torch.argmax(X_test, dim=1)
        testing_ds = TensorDataset(X_test, y_test)
        test_loader = DataLoader(testing_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_loader, X_test, y_test
        
    elif dataset.lower() == 'mpra':
        data_file = h5py.File(data_path, 'r')
        X_test = torch.tensor(np.array(data_file['x_test']).astype(np.float32)).permute(0,2,1)
        y_test = torch.tensor(np.array(data_file['y_test']).astype(np.float32))
        X_test = torch.argmax(X_test, dim=1)
        testing_ds = TensorDataset(X_test, y_test)
        test_loader = DataLoader(testing_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_loader, X_test, y_test
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_sequence_length(dataset):
    """Get sequence length for dataset"""
    lengths = {
        'deepstarr': 249,
        'mpra': 200,
        'promoter': 1024
    }
    return lengths.get(dataset.lower(), 249)


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated samples using oracle models")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=['deepstarr', 'mpra', 'promoter'],
                       help="Dataset to evaluate")
    parser.add_argument("--arch", type=str, required=True,
                       choices=['Conv', 'Tran'],
                       help="Model architecture: Conv (Convolutional) or Tran (Transformer)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--oracle_path", type=str, default=None,
                       help="Path to the oracle model checkpoint (auto-resolved if not provided)")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to the dataset file (auto-resolved if not provided)")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for evaluation")
    parser.add_argument("--steps", type=int, default=None,
                       help="Number of sampling steps (defaults to sequence length)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save results (defaults to model_path)")
    args = parser.parse_args()

    # Auto-resolve paths if not provided
    if args.oracle_path is None:
        oracle_files = {
            'deepstarr': 'oracle_DeepSTARR_DeepSTARR_data.ckpt',
            'mpra': 'oracle_mpra_mpra_data.ckpt',
            'promoter': 'best.sei.model.pth.tar'
        }
        args.oracle_path = f"model_zoo/{args.dataset}/oracle_models/{oracle_files[args.dataset]}"
        print(f"Using auto-resolved oracle path: {args.oracle_path}")
    
    if args.data_path is None:
        data_files = {
            'deepstarr': 'DeepSTARR_data.h5',
            'mpra': 'mpra_data.h5',
            'promoter': 'promoter_data.h5'  # Update this if different
        }
        args.data_path = data_files[args.dataset]
        print(f"Using auto-resolved data path: {args.data_path}")

    # Set defaults
    if args.steps is None:
        args.steps = get_sequence_length(args.dataset)
    if args.output_dir is None:
        args.output_dir = args.model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Evaluating {args.dataset} model with {args.arch} architecture")
    
    # Load the diffusion model
    model, graph, noise = load_model_local(args.model_path, device)
    
    # Load oracle model
    oracle = load_oracle_model(args.dataset, args.oracle_path, args.data_path)
    
    # Load test data
    test_loader, X_test, y_test = load_test_data(args.dataset, args.data_path, args.batch_size)
    
    # Get sequence length
    seq_length = get_sequence_length(args.dataset)
    
    # Initialize sampling function
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, seq_length), 'analytic', args.steps, device=device
    )
    
    val_pred_seq = []
    
    print(f"Starting evaluation for {args.dataset} dataset...")
    
    for batch_idx, (batch, val_target) in enumerate(test_loader):
        if batch.shape[0] != args.batch_size:
            # Adjust sampling function for last batch
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (batch.shape[0], seq_length), 'analytic', args.steps, device=device
            )
        
        # Generate samples
        sample = sampling_fn(model, val_target.to(device))
        seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()
        val_pred_seq.append(seq_pred_one_hot)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Concatenate all predictions
    val_pred_seqs = torch.cat(val_pred_seq, dim=0)
    
    # Evaluate using oracle model
    print("Evaluating with oracle model...")
    
    if args.dataset.lower() == 'deepstarr':
        val_score = oracle.predict_custom(oracle.X_test.to(device))
        val_pred_score = oracle.predict_custom(val_pred_seqs.permute(0, 2, 1).to(device))
    elif args.dataset.lower() == 'mpra':
        val_score = oracle.predict_custom(oracle.X_test.to(device))
        val_pred_score = oracle.predict_custom(val_pred_seqs.permute(0, 2, 1).to(device))
    
    # Calculate MSE
    sp_mse = (val_score - val_pred_score) ** 2
    mean_sp_mse = torch.mean(sp_mse).cpu()
    
    print(f"Test-sp-mse: {mean_sp_mse}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"evaluation_results_{args.dataset}.npz")
    
    np.savez(output_path,
             generated_sequences=val_pred_seqs.cpu(),
             oracle_scores=val_score.cpu(),
             predicted_scores=val_pred_score.cpu(),
             mse=mean_sp_mse.numpy(),
             test_sequences=X_test.cpu(),
             test_targets=y_test.cpu())
    
    print(f"Results saved to: {output_path}")
    print(f"Final MSE: {mean_sp_mse}")


if __name__ == "__main__":
    main()