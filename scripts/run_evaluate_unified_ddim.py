import torch
import argparse
import sys
import os
import importlib.util
import time

from utils import data
from utils.load_model import load_model_local
import torch.nn.functional as F
from scripts import sampling
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm


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
    parser = argparse.ArgumentParser(description="Evaluate generated samples using oracle models with DDIM sampling")
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
    parser.add_argument("--ddim_steps", type=int, default=20,
                       help="Number of DDIM sampling steps (default: 20, much fewer than DDPM)")
    parser.add_argument("--eta", type=float, default=0.0,
                       help="DDIM stochasticity parameter (0=deterministic, 1=stochastic like DDPM)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature for score-based sampling")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save results (defaults to model_path)")
    parser.add_argument("--show_progress", action="store_true",
                       help="Show progress bar during evaluation")
    parser.add_argument("--compare_ddpm", action="store_true",
                       help="Also run standard DDPM sampling for comparison")
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
    if args.output_dir is None:
        args.output_dir = args.model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Evaluating {args.dataset} model with {args.arch} architecture using DDIM sampling")
    print(f"DDIM steps: {args.ddim_steps}, eta: {args.eta}, temperature: {args.temperature}")
    
    # Load the diffusion model
    model, graph, noise = load_model_local(args.model_path, device)
    
    # Check if graph is uniform (required for DDIM)
    if graph.absorb:
        raise ValueError("DDIM sampler only works with uniform (non-absorbing) graphs. "
                        "Your model uses an absorbing graph.")
    
    # Load oracle model
    oracle = load_oracle_model(args.dataset, args.oracle_path, args.data_path)
    
    # Load test data
    test_loader, X_test, y_test = load_test_data(args.dataset, args.data_path, args.batch_size)
    
    # Get sequence length
    seq_length = get_sequence_length(args.dataset)
    
    # Initialize DDIM sampling function
    ddim_sampling_fn = sampling.get_ddim_sampler_wrapper(
        graph=graph, 
        noise=noise,
        batch_dims=(args.batch_size, seq_length),
        num_inference_steps=args.ddim_steps,
        eta=args.eta,
        temperature=args.temperature,
        device=device
    )
    
    # Optionally initialize standard DDPM sampling for comparison
    if args.compare_ddpm:
        ddpm_sampling_fn = sampling.get_pc_sampler(
            graph, noise, (args.batch_size, seq_length), 'analytic', seq_length, device=device
        )
    
    val_pred_seq_ddim = []
    val_pred_seq_ddmp = [] if args.compare_ddpm else None
    ddim_times = []
    ddpm_times = [] if args.compare_ddpm else None
    
    print(f"Starting DDIM evaluation for {args.dataset} dataset...")
    
    # Setup progress bar if requested
    if args.show_progress:
        pbar = tqdm(test_loader, desc="Processing batches", unit="batch")
    else:
        pbar = test_loader
    
    for batch_idx, (batch, val_target) in enumerate(pbar):
        current_batch_size = batch.shape[0]
        
        if current_batch_size != args.batch_size:
            # Adjust sampling function for last batch
            ddim_sampling_fn = sampling.get_ddim_sampler_wrapper(
                graph=graph, 
                noise=noise,
                batch_dims=(current_batch_size, seq_length),
                num_inference_steps=args.ddim_steps,
                eta=args.eta,
                temperature=args.temperature,
                device=device
            )
            
            if args.compare_ddpm:
                ddmp_sampling_fn = sampling.get_pc_sampler(
                    graph, noise, (current_batch_size, seq_length), 'analytic', seq_length, device=device
                )
        
        # DDIM sampling
        start_time = time.time()
        sample_ddim = ddim_sampling_fn(model, val_target.to(device))
        ddim_time = time.time() - start_time
        ddim_times.append(ddim_time)
        
        seq_pred_one_hot_ddim = F.one_hot(sample_ddim, num_classes=4).float()
        val_pred_seq_ddim.append(seq_pred_one_hot_ddim)
        
        # Optional DDPM sampling for comparison
        if args.compare_ddpm:
            start_time = time.time()
            sample_ddpm = ddmp_sampling_fn(model, val_target.to(device))
            ddpm_time = time.time() - start_time
            ddpm_times.append(ddpm_time)
            
            seq_pred_one_hot_ddpm = F.one_hot(sample_ddpm, num_classes=4).float()
            val_pred_seq_ddmp.append(seq_pred_one_hot_ddpm)
        
        # Update progress bar description if using tqdm
        if args.show_progress:
            postfix = {
                'batch': f"{batch_idx + 1}/{len(test_loader)}",
                'ddim_time': f"{ddim_time:.3f}s"
            }
            if args.compare_ddpm:
                postfix['ddpm_time'] = f"{ddpm_time:.3f}s"
                postfix['speedup'] = f"{ddpm_time/ddim_time:.1f}x"
            pbar.set_postfix(postfix)
        elif (batch_idx + 1) % 10 == 0:
            avg_ddim_time = np.mean(ddim_times)
            print(f"Processed {batch_idx + 1}/{len(test_loader)} batches, avg DDIM time: {avg_ddim_time:.3f}s")
            if args.compare_ddpm:
                avg_ddpm_time = np.mean(ddpm_times)
                speedup = avg_ddpm_time / avg_ddim_time
                print(f"  DDPM time: {avg_ddpm_time:.3f}s, speedup: {speedup:.1f}x")
    
    # Concatenate all predictions
    val_pred_seqs_ddim = torch.cat(val_pred_seq_ddim, dim=0)
    if args.compare_ddpm:
        val_pred_seqs_ddpm = torch.cat(val_pred_seq_ddmp, dim=0)
    
    # Calculate timing statistics
    total_ddim_time = sum(ddim_times)
    avg_ddim_time = np.mean(ddim_times)
    
    print(f"\nTiming Results:")
    print(f"DDIM - Total: {total_ddim_time:.2f}s, Average: {avg_ddim_time:.3f}s per batch")
    
    if args.compare_ddpm:
        total_ddpm_time = sum(ddpm_times)
        avg_ddpm_time = np.mean(ddpm_times)
        speedup = avg_ddpm_time / avg_ddim_time
        print(f"DDPM - Total: {total_ddpm_time:.2f}s, Average: {avg_ddpm_time:.3f}s per batch")
        print(f"Speedup: {speedup:.1f}x faster with DDIM")
    
    # Evaluate using oracle model
    print("\nEvaluating with oracle model...")
    
    if args.dataset.lower() == 'deepstarr':
        val_score = oracle.predict_custom(oracle.X_test.to(device))
        val_pred_score_ddim = oracle.predict_custom(val_pred_seqs_ddim.permute(0, 2, 1).to(device))
        if args.compare_ddpm:
            val_pred_score_ddpm = oracle.predict_custom(val_pred_seqs_ddpm.permute(0, 2, 1).to(device))
    elif args.dataset.lower() == 'mpra':
        val_score = oracle.predict_custom(oracle.X_test.to(device))
        val_pred_score_ddim = oracle.predict_custom(val_pred_seqs_ddim.permute(0, 2, 1).to(device))
        if args.compare_ddpm:
            val_pred_score_ddpm = oracle.predict_custom(val_pred_seqs_ddpm.permute(0, 2, 1).to(device))
    
    # Calculate MSE
    sp_mse_ddim = (val_score - val_pred_score_ddim) ** 2
    mean_sp_mse_ddim = torch.mean(sp_mse_ddim).cpu()
    
    print(f"\nQuality Results:")
    print(f"DDIM Test-sp-mse: {mean_sp_mse_ddim}")
    
    if args.compare_ddpm:
        sp_mse_ddpm = (val_score - val_pred_score_ddpm) ** 2
        mean_sp_mse_ddpm = torch.mean(sp_mse_ddpm).cpu()
        print(f"DDPM Test-sp-mse: {mean_sp_mse_ddpm}")
        quality_ratio = mean_sp_mse_ddim / mean_sp_mse_ddpm
        print(f"Quality ratio (DDIM/DDPM): {quality_ratio:.3f} {'(better)' if quality_ratio < 1 else '(worse)'}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"ddim_evaluation_results_{args.dataset}.npz")
    
    save_dict = {
        'generated_sequences_ddim': val_pred_seqs_ddim.cpu(),
        'oracle_scores': val_score.cpu(),
        'predicted_scores_ddim': val_pred_score_ddim.cpu(),
        'mse_ddim': mean_sp_mse_ddim.numpy(),
        'test_sequences': X_test.cpu(),
        'test_targets': y_test.cpu(),
        'ddim_times': np.array(ddim_times),
        'avg_ddim_time': avg_ddim_time,
        'total_ddim_time': total_ddim_time,
        'ddim_steps': args.ddim_steps,
        'eta': args.eta,
        'temperature': args.temperature
    }
    
    if args.compare_ddpm:
        save_dict.update({
            'generated_sequences_ddpm': val_pred_seqs_ddpm.cpu(),
            'predicted_scores_ddpm': val_pred_score_ddpm.cpu(),
            'mse_ddpm': mean_sp_mse_ddpm.numpy(),
            'ddpm_times': np.array(ddpm_times),
            'avg_ddpm_time': avg_ddpm_time,
            'total_ddpm_time': total_ddpm_time,
            'speedup': avg_ddpm_time / avg_ddim_time,
            'quality_ratio': quality_ratio.numpy()
        })
    
    np.savez(output_path, **save_dict)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Final DDIM MSE: {mean_sp_mse_ddim}")
    if args.compare_ddpm:
        print(f"Final DDPM MSE: {mean_sp_mse_ddpm}")
        print(f"Speedup: {avg_ddpm_time / avg_ddim_time:.1f}x")


if __name__ == "__main__":
    main()