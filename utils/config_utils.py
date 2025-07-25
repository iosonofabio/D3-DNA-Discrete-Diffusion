"""
Configuration utilities for dynamic path resolution and config management.
"""
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading with dynamic path resolution."""
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigManager.
        
        Args:
            base_dir: Base directory for path resolution. If None, uses current working directory.
        """
        if base_dir is None:
            base_dir = Path.cwd()
        self.base_dir = Path(base_dir).resolve()
        
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        """
        Load configuration file with dynamic path resolution.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration object with resolved paths
        """
        config_path = Path(config_path)
        
        # Try absolute path first, then relative to base_dir, then relative to configs/
        if config_path.is_absolute() and config_path.exists():
            full_path = config_path
        elif (self.base_dir / config_path).exists():
            full_path = self.base_dir / config_path
        elif (self.base_dir / 'configs' / config_path).exists():
            full_path = self.base_dir / 'configs' / config_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        logger.info(f"Loading config from: {full_path}")
        config = OmegaConf.load(full_path)
        
        # Resolve paths dynamically
        config = self._resolve_paths(config)
        
        return config
    
    def _resolve_paths(self, config: DictConfig) -> DictConfig:
        """
        Resolve relative paths in configuration to absolute paths.
        
        Args:
            config: Configuration object
            
        Returns:
            Configuration object with resolved paths
        """
        config = OmegaConf.create(config)  # Create a copy
        
        # Resolve dataset paths
        if 'dataset' in config:
            dataset_name = config.dataset.get('name', '')
            
            # Resolve data file path
            if 'data_file' not in config.paths and not config.paths.data_file:
                config.paths.data_file = self.base_dir / 'model_zoo' / dataset_name / 'DeepSTARR_data.h5'
                
            # Resolve oracle model path
            if 'oracle_model' not in config.paths and not config.paths.oracle_model:
                config.paths.oracle_model = self.base_dir / 'model_zoo' / dataset_name / 'oracle_DeepSTARR_DeepSTARR_data.ckpt'
                
            # Resolve checkpoints directory
            if 'checkpoints_dir' not in config.paths and not config.paths.checkpoints_dir:
                config.paths.checkpoints_dir = self.base_dir / 'model_zoo' / dataset_name / 'checkpoints'
        
        # Resolve SP-MSE validation paths if they're null (auto-resolve)
        if 'sp_mse_validation' in config:
            sp_mse_config = config.sp_mse_validation
            
            if sp_mse_config.get('data_path') is None and 'paths' in config:
                sp_mse_config.data_path = config.paths.get('data_file')
                
            if sp_mse_config.get('oracle_path') is None and 'paths' in config:
                sp_mse_config.oracle_path = config.paths.get('oracle_model')
                
            if sp_mse_config.get('sampling_steps') is None and 'model' in config:
                sp_mse_config.sampling_steps = config.model.get('length')
        
        return config
    
    def save_config(self, config: DictConfig, output_path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration object to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        OmegaConf.save(config, output_path)
        logger.info(f"Configuration saved to: {output_path}")


def get_dataset_config_path(dataset_name: str, architecture: str, 
                          configs_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Get the standard config path for a dataset and architecture.
    
    Args:
        dataset_name: Name of the dataset (deepstarr, mpra, promoter)
        architecture: Architecture name (transformer, convolutional)
        configs_dir: Directory containing config files. If None, uses 'configs' relative to cwd.
        
    Returns:
        Path to configuration file
    """
    if configs_dir is None:
        configs_dir = Path.cwd() / 'configs'
    else:
        configs_dir = Path(configs_dir)
    
    config_filename = f"{dataset_name.lower()}_{architecture.lower()}.yaml"
    return configs_dir / config_filename


def create_run_name(config: DictConfig) -> str:
    """
    Create a run name based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Run name string
    """
    dataset_name = config.get('dataset', {}).get('name', 'unknown')
    architecture = config.get('model', {}).get('architecture', 'unknown')
    
    # Add timestamp for uniqueness
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{dataset_name}_{architecture}_{timestamp}"


def validate_config(config: DictConfig) -> None:
    """
    Validate configuration for required fields.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = [
        'dataset.name',
        'model.architecture',
        'model.hidden_size',
        'model.n_heads',
        'model.n_blocks'
    ]
    
    for field in required_fields:
        if not OmegaConf.select(config, field):
            raise ValueError(f"Required configuration field missing: {field}")
    
    # Validate dataset-specific fields
    dataset_name = config.dataset.name.lower()
    if dataset_name not in ['deepstarr', 'mpra', 'promoter']:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Validate architecture
    architecture = config.model.architecture.lower()
    if architecture not in ['transformer', 'convolutional']:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    logger.info("Configuration validation passed")


# Convenience function for backward compatibility
def load_config_with_paths(config_path: Union[str, Path], 
                         base_dir: Optional[Union[str, Path]] = None) -> DictConfig:
    """
    Load configuration with automatic path resolution.
    
    Args:
        config_path: Path to configuration file
        base_dir: Base directory for path resolution
        
    Returns:
        Configuration object with resolved paths
    """
    manager = ConfigManager(base_dir)
    config = manager.load_config(config_path)
    validate_config(config)
    return config