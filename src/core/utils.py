# utils.py

"""
Utility functions for Maze RL
"""

import logging
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pickle


def setup_logging(name: str = "maze_rl", 
                  level: int = logging.INFO,
                  log_to_file: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'experiment': {
            'name': 'maze_rl_default',
            'seed': 42,
            'use_wandb': False
        },
        'environment': {
            'grid_size': 11,
            'max_steps': 100,
            'obstacle_fraction': 0.25,
            'n_food_sources': 4,
            'food_energy': 10.0,
            'initial_energy': 30.0,
            'energy_decay': 0.98,
            'energy_per_step': 0.1
        },
        'model': {
            'type': 'lstm',
            'hidden_size': 512,
            'use_auxiliary': False
        },
        'training': {
            'epochs': 10000,
            'batch_size': 64,
            'learning_rate': 0.0005,
            'gamma': 0.97,
            'entropy_coef': 0.01,
            'max_grad_norm': 1.0,
            'save_interval': 1000,
            'test_interval': 500
        }
    }


def save_checkpoint(state: Dict[str, Any], 
                    filename: str, 
                    is_best: bool = False):
    """Save training checkpoint"""
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = save_dir / filename
    torch.save(state, checkpoint_path)
    
    # Save as best if needed
    if is_best:
        best_path = save_dir / "best.pt"
        torch.save(state, best_path)


def load_checkpoint(filename: str) -> Dict[str, Any]:
    """Load training checkpoint"""
    checkpoint_path = Path("checkpoints") / filename
    if checkpoint_path.exists():
        return safe_load(checkpoint_path)
    return {}

def compute_grad_norm(model) -> float:
    """Compute gradient norm for monitoring"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def create_experiment_dir(experiment_name: str) -> Path:
    """Create directory for experiment results"""
    exp_dir = Path("experiments") / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    
    return exp_dir


def dict_to_str(d: Dict) -> str:
    """Convert dictionary to string representation"""
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.append(f"{k}: {dict_to_str(v)}")
        else:
            items.append(f"{k}: {v}")
    return "{" + ", ".join(items) + "}"


def save_results(results: Dict[str, Any], filename: str):
    """Save results to file"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    filepath = results_dir / filename
    
    if filename.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    elif filename.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif filename.endswith('.txt'):
        with open(filepath, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def load_results(filename: str) -> Dict[str, Any]:
    """Load results from file"""
    filepath = Path("results") / filename
    
    if not filepath.exists():
        return {}
    
    if filename.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filename.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.txt'):
        results = {}
        with open(filepath, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    results[key.strip()] = value.strip()
        return results
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    

def safe_load(path: str, map_location=None, **kwargs):
    """Safely load a checkpoint with backward compatibility"""
    try:
        # Try with weights_only=True first (PyTorch 2.6+ default)
        return torch.load(path, map_location=map_location, weights_only=True, **kwargs)
    except Exception as e:
        # If that fails, try without weights_only with a warning
        import warnings
        warnings.warn(
            f"Loading {path} with weights_only=False due to error: {str(e)[:100]}... "
            "Only use trusted checkpoints.",
            UserWarning
        )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
    


def get_model_name_from_path(model_path: str) -> str:
    """Extract a clean model name from a file path"""
    from pathlib import Path
    path = Path(model_path)
    
    # Get the filename without extension
    filename = path.stem
    
    # Remove common suffixes
    for suffix in ['_best', '_final', '_checkpoint', '_epoch']:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
    
    return filename