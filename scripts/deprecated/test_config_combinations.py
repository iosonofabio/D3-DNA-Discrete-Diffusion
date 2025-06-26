#!/usr/bin/env python3
"""
Test script to verify that all dataset/architecture configuration combinations
are properly set up and can be loaded.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def test_config_loading(dataset, arch):
    """Test loading a specific dataset/architecture config."""
    print(f"\n{'='*50}")
    print(f"Testing config loading: {dataset} + {arch}")
    print(f"{'='*50}")
    
    try:
        # Check if config file exists
        config_path = f"model_zoo/{dataset}/config/{arch}/hydra/config.yaml"
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        print(f"✓ Config file exists: {config_path}")
        
        # Try to load with hydra
        try:
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf, open_dict
            
            config_dir = os.path.dirname(os.path.abspath(config_path))
            config_name = os.path.basename(config_path).replace('.yaml', '')
            
            print(f"Config dir: {config_dir}")
            print(f"Config name: {config_name}")
            
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name=config_name)
            
            print(f"✓ Config loaded successfully")
            
            # Check architecture field
            expected_arch = "convolutional" if arch == "Conv" else "transformer"
            if hasattr(cfg.model, 'architecture'):
                actual_arch = cfg.model.architecture
                if actual_arch == expected_arch:
                    print(f"✓ Architecture matches: {actual_arch}")
                else:
                    print(f"⚠ Architecture mismatch: expected {expected_arch}, got {actual_arch}")
                    return False
            else:
                print(f"❌ No architecture field in config")
                return False
            
            # Check important fields
            important_fields = ['cond_dim', 'length', 'hidden_size', 'n_blocks']
            for field in important_fields:
                if hasattr(cfg.model, field):
                    value = getattr(cfg.model, field)
                    print(f"✓ {field}: {value}")
                else:
                    print(f"⚠ Missing field: {field}")
            
            # Check architecture-specific cond_dim
            expected_cond_dim = 256 if arch == "Conv" else 128
            actual_cond_dim = getattr(cfg.model, 'cond_dim', None)
            if actual_cond_dim == expected_cond_dim:
                print(f"✓ cond_dim matches architecture: {actual_cond_dim}")
            else:
                print(f"⚠ cond_dim mismatch: expected {expected_cond_dim}, got {actual_cond_dim}")
            
            # Check dataset-specific sequence length
            expected_lengths = {
                'deepstarr': 249,
                'mpra': 200,
                'promoter': 1024
            }
            expected_length = expected_lengths.get(dataset.lower())
            actual_length = getattr(cfg.model, 'length', None)
            if actual_length == expected_length:
                print(f"✓ Sequence length matches dataset: {actual_length}")
            else:
                print(f"⚠ Sequence length mismatch: expected {expected_length}, got {actual_length}")
            
            print(f"✅ SUCCESS: {dataset} + {arch} config is valid")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return False
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return False


def test_import_structure():
    """Test that the Lightning trainer components can be imported."""
    print(f"\n{'='*50}")
    print("Testing import structure")
    print(f"{'='*50}")
    
    try:
        # Test basic imports
        from scripts.lightning_trainer import (
            D3LightningModule, D3DataModule, create_trainer_from_config,
            create_lightning_module, get_model_class_for_dataset
        )
        print("✓ Lightning trainer imports successful")
        
        from utils.data import get_datasets, PromoterDataset
        print("✓ Data utilities imports successful")
        
        # Test dataset-specific model imports
        datasets = ['deepstarr', 'mpra', 'promoter']
        for dataset in datasets:
            try:
                model_class = get_model_class_for_dataset(dataset)
                print(f"✓ {dataset} model class: {model_class.__name__}")
            except Exception as e:
                print(f"⚠ {dataset} model class error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def main():
    """Test all configurations and imports."""
    print("Testing PyTorch Lightning Pipeline - Configuration Validation")
    print("="*80)
    
    # Test import structure first
    imports_ok = test_import_structure()
    
    # Test config combinations
    datasets = ['deepstarr', 'mpra', 'promoter']
    architectures = ['Conv', 'Tran']
    
    results = {}
    
    for dataset in datasets:
        for arch in architectures:
            combination = f"{dataset}_{arch}"
            try:
                success = test_config_loading(dataset, arch)
                results[combination] = success
            except Exception as e:
                print(f"❌ CRITICAL FAILURE for {combination}: {e}")
                results[combination] = False
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    print(f"Import structure: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Total config combinations tested: {total_tests}")
    print(f"Config tests passed: {passed_tests}")
    print(f"Config tests failed: {failed_tests}")
    
    print(f"\nDetailed Results:")
    for combination, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {combination}: {status}")
    
    overall_success = imports_ok and failed_tests == 0
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED! Lightning pipeline configurations are valid.")
    else:
        print(f"\n⚠️  Some tests failed. Lightning pipeline needs fixes.")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)