import torch
import argparse
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from utils import data
from utils.load_model import load_model_local
import torch.nn.functional as F
from scripts import sampling
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_oracle_model(dataset, oracle_path, data_path):
    """Load the appropriate oracle model based on dataset"""
    if dataset.lower() == 'deepstarr':
        sys.path.insert(0, 'model_zoo/deepstarr')
        try:
            from deepstarr import PL_DeepSTARR
            oracle = PL_DeepSTARR.load_from_checkpoint(oracle_path, input_h5_file=data_path).eval()
            return oracle
        finally:
            sys.path.pop(0)
    elif dataset.lower() == 'mpra':
        sys.path.insert(0, 'model_zoo/mpra')
        try:
            from mpra import PL_mpra
            oracle = PL_mpra.load_from_checkpoint(oracle_path, input_h5_file=data_path).eval()
            return oracle
        finally:
            sys.path.pop(0)
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
    lengths = {'deepstarr': 249, 'mpra': 200, 'promoter': 1024}
    return lengths.get(dataset.lower(), 249)


def load_existing_results(csv_path):
    """Load existing results if available"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Remove any rows with NaN values (failed runs)
            df = df.dropna()
            print(f"📂 Found existing results: {len(df)} completed configurations")
            return df
        except Exception as e:
            print(f"⚠️  Error loading existing results: {e}")
            return pd.DataFrame()
    else:
        print("🆕 Starting fresh - no existing results found")
        return pd.DataFrame()


def get_remaining_combinations(eta_values, steps_values, existing_df):
    """Get list of remaining combinations to run"""
    all_combinations = list(product(eta_values, steps_values))
    
    if len(existing_df) == 0:
        return all_combinations
    
    # Create set of completed combinations
    completed = set()
    for _, row in existing_df.iterrows():
        completed.add((row['eta'], row['steps']))
    
    # Filter out completed combinations
    remaining = [combo for combo in all_combinations if combo not in completed]
    
    print(f"📊 Progress: {len(existing_df)}/{len(all_combinations)} completed, {len(remaining)} remaining")
    
    return remaining


def save_results(df, csv_path, backup=True):
    """Save results to CSV with optional backup"""
    try:
        # Create backup of existing file
        if backup and os.path.exists(csv_path):
            backup_path = csv_path.replace('.csv', '_backup.csv')
            os.rename(csv_path, backup_path)
        
        # Save current results
        df.to_csv(csv_path, index=False)
        
        # Remove backup if save was successful
        if backup:
            backup_path = csv_path.replace('.csv', '_backup.csv')
            if os.path.exists(backup_path):
                os.remove(backup_path)
                
    except Exception as e:
        print(f"⚠️  Error saving results: {e}")
        # Restore backup if save failed
        if backup:
            backup_path = csv_path.replace('.csv', '_backup.csv')
            if os.path.exists(backup_path):
                os.rename(backup_path, csv_path)


def evaluate_single_config(model, graph, noise, oracle, test_loader, seq_length, 
                          eta, ddim_steps, batch_size, device, dataset):
    """Evaluate a single eta/steps configuration"""
    
    # Initialize DDIM sampling function
    ddim_sampling_fn = sampling.get_ddim_sampler_wrapper(
        graph=graph, 
        noise=noise,
        batch_dims=(batch_size, seq_length),
        num_inference_steps=ddim_steps,
        eta=eta,
        temperature=1.0,
        device=device
    )
    
    val_pred_seq_ddim = []
    batch_times = []
    
    for batch_idx, (batch, val_target) in enumerate(test_loader):
        current_batch_size = batch.shape[0]
        
        # Adjust sampling function for last batch if needed
        if current_batch_size != batch_size:
            ddim_sampling_fn = sampling.get_ddim_sampler_wrapper(
                graph=graph, 
                noise=noise,
                batch_dims=(current_batch_size, seq_length),
                num_inference_steps=ddim_steps,
                eta=eta,
                temperature=1.0,
                device=device
            )
        
        # DDIM sampling
        start_time = time.time()
        sample_ddim = ddim_sampling_fn(model, val_target.to(device))
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        seq_pred_one_hot_ddim = F.one_hot(sample_ddim, num_classes=4).float()
        val_pred_seq_ddim.append(seq_pred_one_hot_ddim)
    
    # Concatenate all predictions
    val_pred_seqs_ddim = torch.cat(val_pred_seq_ddim, dim=0)
    
    # Evaluate using oracle model
    if dataset.lower() == 'deepstarr':
        val_score = oracle.predict_custom(oracle.X_test.to(device))
        val_pred_score_ddim = oracle.predict_custom(val_pred_seqs_ddim.permute(0, 2, 1).to(device))
    elif dataset.lower() == 'mpra':
        val_score = oracle.predict_custom(oracle.X_test.to(device))
        val_pred_score_ddim = oracle.predict_custom(val_pred_seqs_ddim.permute(0, 2, 1).to(device))
    
    # Calculate MSE
    sp_mse_ddim = (val_score - val_pred_score_ddim) ** 2
    mean_sp_mse_ddim = torch.mean(sp_mse_ddim).cpu().item()
    avg_time = np.mean(batch_times)
    
    return mean_sp_mse_ddim, avg_time


def create_plots(df, output_dir):
    """Create visualization plots"""
    if len(df) == 0:
        print("⚠️  No data to plot")
        return
        
    plt.style.use('default')
    sns.set_palette("viridis")
    
    try:
        # 1. MSE Heatmap
        plt.figure(figsize=(10, 6))
        pivot_mse = df.pivot(index='eta', columns='steps', values='mse')
        sns.heatmap(pivot_mse, annot=True, fmt='.4f', cmap='RdYlBu_r', cbar_kws={'label': 'MSE'})
        plt.title('DDIM MSE vs Eta and Steps')
        plt.xlabel('Steps')
        plt.ylabel('Eta')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mse_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Time Heatmap  
        plt.figure(figsize=(10, 6))
        pivot_time = df.pivot(index='eta', columns='steps', values='avg_time')
        sns.heatmap(pivot_time, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Time (s)'})
        plt.title('DDIM Average Time vs Eta and Steps')
        plt.xlabel('Steps')
        plt.ylabel('Eta')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. MSE vs Steps
        plt.figure(figsize=(10, 6))
        for eta in sorted(df['eta'].unique()):
            subset = df[df['eta'] == eta]
            plt.plot(subset['steps'], subset['mse'], marker='o', label=f'η = {eta}', linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('MSE')
        plt.title('MSE vs Steps for Different Eta Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mse_vs_steps.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. MSE vs Time scatter
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df['avg_time'], df['mse'], c=df['steps'], s=60, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, label='Steps')
        plt.xlabel('Average Time (seconds)')
        plt.ylabel('MSE')
        plt.title('MSE vs Computation Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mse_vs_time.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"⚠️  Error creating plots: {e}")


def print_summary(df):
    """Print summary of results"""
    if len(df) == 0:
        print("⚠️  No results to summarize")
        return
        
    print(f"\n📈 SUMMARY ({len(df)} configurations):")
    print(f"Best MSE: {df['mse'].min():.6f} (η={df.loc[df['mse'].idxmin(), 'eta']}, steps={df.loc[df['mse'].idxmin(), 'steps']})")
    print(f"Fastest: {df['avg_time'].min():.3f}s (η={df.loc[df['avg_time'].idxmin(), 'eta']}, steps={df.loc[df['avg_time'].idxmin(), 'steps']})")
    
    # Best trade-off (normalize both metrics and find minimum sum)
    df_norm = df.copy()
    df_norm['mse_norm'] = (df['mse'] - df['mse'].min()) / (df['mse'].max() - df['mse'].min())
    df_norm['time_norm'] = (df['avg_time'] - df['avg_time'].min()) / (df['avg_time'].max() - df['avg_time'].min())
    df_norm['combined'] = df_norm['mse_norm'] + df_norm['time_norm']
    best_tradeoff_idx = df_norm['combined'].idxmin()
    
    print(f"Best trade-off: MSE={df.loc[best_tradeoff_idx, 'mse']:.6f}, Time={df.loc[best_tradeoff_idx, 'avg_time']:.3f}s (η={df.loc[best_tradeoff_idx, 'eta']}, steps={df.loc[best_tradeoff_idx, 'steps']})")


def main():
    parser = argparse.ArgumentParser(description="DDIM Parameter Sweep (Resumable)")
    parser.add_argument("--dataset", type=str, required=True, choices=['deepstarr', 'mpra'])
    parser.add_argument("--arch", type=str, required=True, choices=['Conv', 'Tran'])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--oracle_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="ddim_sweep_output")
    parser.add_argument("--force_restart", action="store_true", 
                       help="Force restart from beginning, ignoring existing results")
    args = parser.parse_args()

    # Auto-resolve paths if not provided
    if args.oracle_path is None:
        oracle_files = {
            'deepstarr': 'oracle_DeepSTARR_DeepSTARR_data.ckpt',
            'mpra': 'oracle_mpra_mpra_data.ckpt'
        }
        args.oracle_path = f"model_zoo/{args.dataset}/oracle_models/{oracle_files[args.dataset]}"
    
    if args.data_path is None:
        data_files = {'deepstarr': 'DeepSTARR_data.h5', 'mpra': 'mpra_data.h5'}
        args.data_path = data_files[args.dataset]

    # Parameter sweep ranges
    eta_values = [0.0, 0.2, 0.5, 0.7, 0.9, 1.0]
    steps_values = [20, 30, 50, 100, 150, 200, 249]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "ddim_sweep_results.csv")
    
    # Load existing results or start fresh
    if args.force_restart:
        print("🔄 Force restart - ignoring existing results")
        results_df = pd.DataFrame()
    else:
        results_df = load_existing_results(csv_path)
    
    # Get remaining combinations to run
    remaining_combinations = get_remaining_combinations(eta_values, steps_values, results_df)
    
    if len(remaining_combinations) == 0:
        print("✅ All combinations already completed!")
        print_summary(results_df)
        create_plots(results_df, args.output_dir)
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🚀 DDIM Parameter Sweep for {args.dataset}")
    print(f"Eta values: {eta_values}")
    print(f"Steps values: {steps_values}")
    print(f"Remaining combinations: {len(remaining_combinations)}")
    print(f"Results will be saved to: {csv_path}")
    print("-" * 50)
    
    # Load models and data once
    print("Loading model and data...")
    model, graph, noise = load_model_local(args.model_path, device)
    
    if graph.absorb:
        raise ValueError("DDIM sampler only works with uniform (non-absorbing) graphs.")
    
    oracle = load_oracle_model(args.dataset, args.oracle_path, args.data_path)
    test_loader, X_test, y_test = load_test_data(args.dataset, args.data_path, args.batch_size)
    seq_length = get_sequence_length(args.dataset)
    
    # Run parameter sweep
    with tqdm(total=len(remaining_combinations), desc="Parameter sweep") as pbar:
        for eta, steps in remaining_combinations:
            pbar.set_description(f"η={eta}, steps={steps}")
            
            try:
                mse, avg_time = evaluate_single_config(
                    model, graph, noise, oracle, test_loader, seq_length,
                    eta, steps, args.batch_size, device, args.dataset
                )
                
                # Add new result to DataFrame
                new_row = pd.DataFrame({
                    'eta': [eta],
                    'steps': [steps],
                    'mse': [mse],
                    'avg_time': [avg_time]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                # Save results immediately
                save_results(results_df, csv_path)
                
                pbar.set_postfix({'MSE': f'{mse:.4f}', 'Time': f'{avg_time:.3f}s', 'Total': len(results_df)})
                
            except Exception as e:
                print(f"\n❌ Error with η={eta}, steps={steps}: {e}")
                # Continue with next combination
            
            pbar.update(1)
    
    # Final summary and plots
    print(f"\n✅ Sweep completed! Final results: {len(results_df)} configurations")
    print_summary(results_df)
    create_plots(results_df, args.output_dir)
    
    print(f"\n🎯 All results saved to: {args.output_dir}")
    print(f"💡 To resume if interrupted, just re-run the same command")


if __name__ == "__main__":
    main()