# env_factory_vector.py

"""
Vectorized environment for parallel execution using environment
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from .environment import VectorGridMazeWorld  # Use the vector-optimized version


class VectorizedMazeEnv:
    """Vectorized wrapper for multiple maze environments - OPTIMIZED"""
    
    def __init__(self,  num_envs: int, env_config: Dict[str, Any], base_seed: int):
        self.num_envs = num_envs
        #self.env_config = env_config

        self.base_seed = base_seed          # store the base seed
        self.reset_counter = 0              # count how many times reset() has been called
        
        # Create optimized environments WITHOUT render overhead
        env_config_no_render = env_config.copy()
        env_config_no_render['render_size'] = 0  # Disable rendering
        
        self.envs = [VectorGridMazeWorld(**env_config_no_render) for _ in range(num_envs)]
        
        # Get observation and action spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Current states - use numpy arrays for faster operations
        self.observations = None
        self.dones = np.zeros(num_envs, dtype=bool)
        self.steps = np.zeros(num_envs, dtype=np.int32)
        
        # Pre-allocated arrays for step results
        self._obs_buffer = np.zeros((num_envs, 10), dtype=np.int32)
        self._reward_buffer = np.zeros(num_envs, dtype=np.float32)
        self._done_buffer = np.zeros(num_envs, dtype=bool)
        self._trunc_buffer = np.zeros(num_envs, dtype=bool)
    
    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments"""
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = self.base_seed + self.reset_counter * self.num_envs + i
            obs, info = env.reset(seed=env_seed)

            if i == 0: # Initialize observations array with first observation shape
                self.observations = np.zeros((self.num_envs, obs.shape[0]), dtype=obs.dtype)
            
            self.observations[i] = obs
            infos.append(info)
            self.dones[i] = False
            self.steps[i] = 0
        
        self.reset_counter += 1
        
        return self.observations, infos
    
    def soft_reset_all(self) -> Tuple[np.ndarray, List[Dict]]:
        """Soft reset all environments (keep grid layout, reset agent/food)."""
        infos = []
        for i, env in enumerate(self.envs):
            obs, info = env.soft_reset()
            if i == 0:
                self.observations = np.zeros((self.num_envs, obs.shape[0]), dtype=obs.dtype)
            self.observations[i] = obs
            infos.append(info)
            self.dones[i] = False
            self.steps[i] = 0
            
        return self.observations, infos

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments in parallel - OPTIMIZED"""
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if not self.dones[i]:
                obs, reward, terminated, truncated, info = env.step(action)
                
                self.observations[i] = obs
                self._reward_buffer[i] = reward
                self._done_buffer[i] = terminated
                self._trunc_buffer[i] = truncated
                infos.append(info)
                self.steps[i] += 1
                
                if terminated or truncated:
                    self.dones[i] = True
            else:
                # Already done, keep last observation
                self._reward_buffer[i] = 0.0
                self._done_buffer[i] = True
                infos.append(infos[-1] if infos else {})
        
        return (
            self.observations.copy(),
            self._reward_buffer.copy(),
            self._done_buffer.copy(),
            self._trunc_buffer.copy(),
            infos
        )
    
    def render(self, env_idx: int = 0):
        """Render a specific environment (only if needed)"""
        return self.envs[env_idx].render()
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()