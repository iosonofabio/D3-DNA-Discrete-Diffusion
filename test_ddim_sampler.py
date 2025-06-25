#!/usr/bin/env python3
"""
Test script for the DDIM sampler implementation.
This script demonstrates how to use the new DDIM sampler with a trained model.
"""

import torch
import argparse
import os
from utils.load_model import load_model_local
from scripts.sampling import get_ddim_sampler_wrapper, get_pc_sampler
import time


def test_ddim_sampler(model_path, seq_length=249, batch_size=4, ddim_steps=20):
    """
    Test the DDIM sampler with a trained model.
    
    Args:
        model_path: Path to trained model
        seq_length: Sequence length
        batch_size: Batch size for testing
        ddim_steps: Number of DDIM steps
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model, graph, noise = load_model_local(model_path, device)
    
    # Check graph type
    print(f"Graph type: {'Absorbing' if graph.absorb else 'Uniform'}")
    print(f"Graph dimension: {graph.dim}")
    
    if graph.absorb:
        print("WARNING: Model uses absorbing graph. DDIM sampler only works with uniform graphs.")
        return
    
    # Create dummy labels (e.g., for conditional generation)
    dummy_labels = torch.zeros(batch_size, device=device)
    
    # Test DDIM sampler
    print(f"\nTesting DDIM sampler with {ddim_steps} steps...")
    ddim_sampler = get_ddim_sampler_wrapper(
        graph=graph,
        noise=noise,
        batch_dims=(batch_size, seq_length),
        num_inference_steps=ddim_steps,
        eta=0.0,  # Deterministic
        temperature=1.0,
        device=device
    )
    
    start_time = time.time()
    samples_ddim = ddim_sampler(model, dummy_labels)
    ddim_time = time.time() - start_time
    
    print(f"DDIM sampling completed in {ddim_time:.3f}s")
    print(f"Generated shape: {samples_ddim.shape}")
    print(f"Sample values range: {samples_ddim.min()} to {samples_ddim.max()}")
    
    # Test standard DDPM for comparison
    print(f"\nTesting standard DDPM sampler with {seq_length} steps...")
    ddpm_sampler = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(batch_size, seq_length),
        predictor='analytic',
        steps=seq_length,
        device=device
    )
    
    start_time = time.time()
    samples_ddpm = ddpm_sampler(model, dummy_labels)
    ddpm_time = time.time() - start_time
    
    print(f"DDPM sampling completed in {ddpm_time:.3f}s")
    print(f"Generated shape: {samples_ddpm.shape}")
    print(f"Sample values range: {samples_ddpm.min()} to {samples_ddpm.max()}")
    
    # Compare results
    speedup = ddpm_time / ddim_time
    print(f"\nComparison:")
    print(f"DDIM: {ddim_steps} steps, {ddim_time:.3f}s")
    print(f"DDPM: {seq_length} steps, {ddpm_time:.3f}s")
    print(f"Speedup: {speedup:.1f}x")
    
    # Check if sequences are different (they should be due to different sampling)
    if torch.equal(samples_ddim, samples_ddpm):
        print("WARNING: DDIM and DDPM generated identical sequences")
    else:
        print("✓ DDIM and DDPM generated different sequences (expected)")
    
    return samples_ddim, samples_ddpm


def main():
    parser = argparse.ArgumentParser(description="Test DDIM sampler implementation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--seq_length", type=int, default=249,
                       help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for testing")
    parser.add_argument("--ddim_steps", type=int, default=20,
                       help="Number of DDIM steps")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return
    
    try:
        test_ddim_sampler(
            model_path=args.model_path,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            ddim_steps=args.ddim_steps
        )
        print("\n✓ DDIM sampler test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()