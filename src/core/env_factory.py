# env_factory.py

"""
Environment factory for creating environments from configurations and command-line arguments
"""

import numpy as np
from typing import Dict, Any, Optional
from .environment import GridMazeWorld

from src.core.constants import TaskClass


class EnvironmentFactory:
    """Factory for creating maze environments with various configurations"""
    
    @staticmethod
    def create_from_args(args, test_mode: bool = False) -> GridMazeWorld:
        """Create environment instance from command line arguments"""
        env_config = {
            'grid_size': 11,
            'max_steps': 100,
            'obstacle_fraction': 0.25,
            'n_food_sources': 4,
            'food_energy': 10.0,
            'initial_energy': 30.0,
            'energy_decay': 0.98,
            'energy_per_step': 0.1,
            'render_size': 512 if test_mode else 0,
            'task_class': args.task_class if hasattr(args, 'task_class') else TaskClass.BASIC,
            'complexity_level': max(0.0, min(1.0, args.complexity_level)) if hasattr(args, 'complexity_level') else 0.0,
        }
        
        # For door parameters: pass -1/None for defaults
        env_config['n_doors'] = -1  # Always use task class default unless explicitly overridden
        env_config['n_buttons_per_door'] = -1
        env_config['door_periodic'] = None
        env_config['button_break_probability'] = -1.0
        
        return GridMazeWorld(**env_config)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any], test_mode: bool = False) -> GridMazeWorld:
        """Create environment instance from configuration dictionary"""
        env_config = config.get('environment', {}).copy()
        
        # Set render size based on mode
        if test_mode:
            env_config['render_size'] = 512
        else:
            env_config['render_size'] = 0
        
        return GridMazeWorld(**env_config)
    
    @staticmethod
    def create_config_for_benchmark(args) -> Dict[str, Any]:
        """Create environment config for benchmarking"""
        env_config = {
            'grid_size': 11,
            'max_steps': 100,
            'obstacle_fraction': 0.25,
            'n_food_sources': 4,
            'render_size': 0,
        }
        
        # Add complexity parameters if provided
        if hasattr(args, 'task_class'):
            env_config['task_class'] = args.task_class
        
        if hasattr(args, 'complexity_level'):
            complexity = max(0.0, min(1.0, args.complexity_level))
            env_config['complexity_level'] = complexity
        
        if hasattr(args, 'n_doors'):
            env_config['n_doors'] = args.n_doors
        
        if hasattr(args, 'n_buttons_per_door'):
            env_config['n_buttons_per_door'] = args.n_buttons_per_door
        
        if hasattr(args, 'door_periodic'):
            env_config['door_periodic'] = args.door_periodic
        
        if hasattr(args, 'button_break_probability'):
            prob = max(0.0, min(1.0, args.button_break_probability))
            env_config['button_break_probability'] = prob
        
        return env_config
    
    @staticmethod
    def create_curriculum_environments(
        base_config: Dict[str, Any],
        complexity_levels: Optional[np.ndarray] = None,
        task_classes: Optional[list] = None
    ) -> list:
        """Create a curriculum of environments with increasing complexity"""
        if complexity_levels is None:
            complexity_levels = np.linspace(0.0, 1.0, 5)
        
        if task_classes is None:
            task_classes = ["basic", "doors", "buttons", "complex"]
        
        environments = []
        
        for task_class in task_classes:
            for complexity in complexity_levels:
                config = base_config.copy()
                config['task_class'] = task_class
                config['complexity_level'] = complexity
                
                # Adjust other parameters based on complexity
                if task_class != "basic":
                    config['n_doors'] = max(1, int(complexity * 4))
                    config['button_break_probability'] = complexity * 0.3
                
                env = GridMazeWorld(**config)
                environments.append(env)
        
        return environments
    
    @staticmethod
    def create_evaluation_suite(
        grid_sizes: list = [7, 11, 15],
        complexity_levels: list = [0.0, 0.5, 1.0],
        task_classes: list = ["basic", "doors", "buttons", "complex"]
    ) -> Dict[str, GridMazeWorld]:
        """Create a comprehensive evaluation suite of environments"""
        suite = {}
        
        for grid_size in grid_sizes:
            for task_class in task_classes:
                for complexity in complexity_levels:
                    key = f"{task_class}_grid{grid_size}_comp{complexity:.1f}"
                    
                    config = {
                        'grid_size': grid_size,
                        'max_steps': grid_size * 10,
                        'obstacle_fraction': 0.25,
                        'n_food_sources': max(2, grid_size // 3),
                        'task_class': task_class,
                        'complexity_level': complexity,
                        'render_size': 0
                    }
                    
                    if task_class != "basic":
                        config['n_doors'] = max(1, int(complexity * (grid_size // 3)))
                        config['n_buttons_per_door'] = 1 if complexity < 0.5 else 2
                        config['button_break_probability'] = complexity * 0.2
                    
                    suite[key] = GridMazeWorld(**config)
        
        return suite