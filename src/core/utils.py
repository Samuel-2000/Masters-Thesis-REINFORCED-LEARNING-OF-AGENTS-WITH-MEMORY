# utils.py

"""
Utility functions for Maze RL
"""

import logging
import random
import numpy as np
import torch
from pathlib import Path



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