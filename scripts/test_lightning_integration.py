#!/usr/bin/env python3
"""
Test script to validate Lightning integration works with all dataset/architecture combinations.
This ensures that the factory pattern correctly routes to dataset-specific implementations.
"""

import os
import sys
import tempfile
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.lightning_trainer import (
    create_lightning_module, 
    get_model_class_for_dataset,
    D3LightningModule,
    PromoterD3LightningModule, 
    MPRAD3LightningModule
)
from utils.checkpoint_utils import get_model_class_for_checkpoint


def test_model_class_routing():
    """Test that the factory correctly routes to dataset-specific model classes."""
    print("=" * 60)
    print("Testing Model Class Routing")
    print("=" * 60)
    
    test_cases = [
        ('deepstarr', 'model.transformer.SEDD'),
        ('promoter', 'transformer_promoter.SEDD'),
        ('mpra', 'transformer_mpra.SEDD'),
    ]
    
    for dataset, expected_module in test_cases:
        try:
            model_class = get_model_class_for_dataset(dataset)
            actual_module = f"{model_class.__module__}.{model_class.__name__}"
            
            print(f"Dataset: {dataset}")
            print(f"  Expected: {expected_module}")
            print(f"  Actual: {actual_module}")
            
            if expected_module in actual_module:
                print(f"  ✓ PASS: Correct model class for {dataset}")
            else:
                print(f"  ✗ FAIL: Wrong model class for {dataset}")
                return False
                
        except Exception as e:
            print(f"  ✗ FAIL: Error loading model class for {dataset}: {e}")
            return False
    
    print("✓ All model class routing tests passed!")
    return True


def test_lightning_module_factory():
    """Test that Lightning module factory creates correct module types."""
    print("\n" + "=" * 60)
    print("Testing Lightning Module Factory")
    print("=" * 60)
    
    # Create dummy config
    from omegaconf import OmegaConf
    dummy_cfg = OmegaConf.create({
        'model': {
            'hidden_size': 128,
            'cond_dim': 64,
            'length': 200,
            'n_blocks': 2,
            'n_heads': 4,
            'scale_by_sigma': False,
            'dropout': 0.1
        },
        'training': {'ema': 0.999},
        'tokens': 4,
        'graph': {'type': 'uniform'},
        'noise': {'type': 'geometric', 'sigma_min': 1e-4, 'sigma_max': 20}
    })
    
    test_cases = [
        ('deepstarr', D3LightningModule),
        ('promoter', PromoterD3LightningModule),
        ('mpra', MPRAD3LightningModule),
        (None, D3LightningModule),  # Default case
    ]
    
    for dataset, expected_class in test_cases:
        try:
            module = create_lightning_module(dummy_cfg, dataset_name=dataset)
            actual_class = type(module)
            
            print(f"Dataset: {dataset}")
            print(f"  Expected: {expected_class.__name__}")
            print(f"  Actual: {actual_class.__name__}")
            
            if actual_class == expected_class:
                print(f"  ✓ PASS: Correct Lightning module for {dataset}")
            else:
                print(f"  ✗ FAIL: Wrong Lightning module for {dataset}")
                return False
                
        except Exception as e:
            print(f"  ✗ FAIL: Error creating Lightning module for {dataset}: {e}")
            return False
    
    print("✓ All Lightning module factory tests passed!")
    return True


def test_config_loading():
    """Test loading actual model_zoo configs."""
    print("\n" + "=" * 60)
    print("Testing Config Loading")
    print("=" * 60)
    
    config_combinations = [
        ('deepstarr', 'Conv'),
        ('deepstarr', 'Tran'),
        ('mpra', 'Conv'),
        ('mpra', 'Tran'),
        ('promoter', 'Conv'),
        ('promoter', 'Tran'),
    ]
    
    for dataset, arch in config_combinations:
        config_path = f"model_zoo/{dataset}/config/{arch}/hydra/config.yaml"
        
        if not os.path.exists(config_path):
            print(f"  ⚠ SKIP: Config not found: {config_path}")
            continue
            
        try:
            from hydra import compose, initialize_config_dir
            
            config_dir = os.path.dirname(os.path.abspath(config_path))
            config_name = os.path.basename(config_path).replace('.yaml', '')
            
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name=config_name)
            
            # Test creating Lightning module with real config
            module = create_lightning_module(cfg, dataset_name=dataset)
            
            print(f"  ✓ PASS: {dataset}/{arch} - Config loaded and module created")
            
            # Check if architecture matches expectation
            expected_arch = "convolutional" if arch == "Conv" else "transformer"
            if hasattr(cfg.model, 'architecture'):
                actual_arch = cfg.model.architecture
                if actual_arch == expected_arch:
                    print(f"    ✓ Architecture matches: {actual_arch}")
                else:
                    print(f"    ⚠ Architecture mismatch: expected {expected_arch}, got {actual_arch}")
            
        except Exception as e:
            print(f"  ✗ FAIL: {dataset}/{arch} - Error: {e}")
            return False
    
    print("✓ All config loading tests passed!")
    return True


def test_model_forward_pass():
    """Test that models can perform forward passes without errors."""
    print("\n" + "=" * 60)
    print("Testing Model Forward Pass")
    print("=" * 60)
    
    from omegaconf import OmegaConf
    
    # Test different sequence lengths for different datasets
    test_configs = {
        'deepstarr': {'length': 249, 'dataset': 'deepstarr'},
        'mpra': {'length': 200, 'dataset': 'mpra'},
        'promoter': {'length': 1024, 'dataset': 'promoter'},
    }
    
    for dataset_name, config_info in test_configs.items():
        try:
            # Create config
            cfg = OmegaConf.create({
                'model': {
                    'architecture': 'transformer',
                    'hidden_size': 128,
                    'cond_dim': 64,
                    'length': config_info['length'],
                    'n_blocks': 2,
                    'n_heads': 4,
                    'scale_by_sigma': False,
                    'dropout': 0.1
                },
                'training': {'ema': 0.999},
                'tokens': 4,
                'graph': {'type': 'uniform'},
                'noise': {'type': 'geometric', 'sigma_min': 1e-4, 'sigma_max': 20}
            })
            
            # Create Lightning module
            module = create_lightning_module(cfg, dataset_name=dataset_name)
            module.eval()
            
            # Create dummy input
            batch_size = 2
            seq_length = config_info['length']
            
            if dataset_name == 'promoter':
                # Promoter expects different input format
                inputs = torch.randint(0, 4, (batch_size, seq_length))
                labels = torch.randn(batch_size, 1)  # Single target for promoter
            elif dataset_name == 'mpra':
                inputs = torch.randint(0, 4, (batch_size, seq_length))
                labels = torch.randn(batch_size, 3)  # 3 targets for MPRA
            else:  # deepstarr
                inputs = torch.randint(0, 4, (batch_size, seq_length))
                labels = torch.randn(batch_size, 2)  # 2 targets for DeepSTARR
            
            # Test forward pass
            with torch.no_grad():
                # Create dummy batch for Lightning step
                if dataset_name == 'promoter':
                    # Promoter expects one-hot + labels concatenated
                    seq_one_hot = torch.nn.functional.one_hot(inputs, num_classes=4).float()
                    batch = torch.cat([seq_one_hot, labels.unsqueeze(-1).expand(-1, seq_length, -1)], dim=-1)
                else:
                    batch = (inputs, labels)
                
                # Test training step
                loss = module.training_step(batch, 0)
                
                print(f"  ✓ PASS: {dataset_name} - Forward pass successful, loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"  ✗ FAIL: {dataset_name} - Forward pass error: {e}")
            return False
    
    print("✓ All forward pass tests passed!")
    return True


def main():
    """Run all integration tests."""
    print("D3 LIGHTNING INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_model_class_routing,
        test_lightning_module_factory,
        test_config_loading,
        test_model_forward_pass,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"\n❌ Test {test.__name__} FAILED")
        except Exception as e:
            print(f"\n❌ Test {test.__name__} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    if passed == len(tests):
        print(f"🎉 ALL {len(tests)} TESTS PASSED!")
        print("Lightning integration is working correctly across all datasets!")
        return 0
    else:
        print(f"❌ {len(tests) - passed}/{len(tests)} TESTS FAILED")
        print("Lightning integration has issues that need to be fixed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)