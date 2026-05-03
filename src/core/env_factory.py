# env_factory.py

"""
Environment factory for creating environments from configurations and command-line arguments
"""

from typing import Dict, Any
from .environment import GridMazeWorld



class EnvironmentFactory:
    """Factory for creating maze environments with various configurations"""
    
    @staticmethod
    def create_from_config(config: Dict[str, Any], test_mode: bool = False) -> GridMazeWorld:
        """Create environment instance from configuration dictionary"""
        if 'environment' in config:
            env_config = config['environment'].copy()
        else:
            env_config = config.copy()  # Assume it's already the env config
        
        # Set render size based on mode
        if test_mode:
            env_config['render_size'] = 512
        else:
            env_config['render_size'] = 0
        
        return GridMazeWorld(**env_config)