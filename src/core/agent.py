"""
Agent class using consistent constants
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import cv2
import os

from src.networks.lstm import LSTMPolicyNet
from src.networks.transformer import TransformerPolicyNet
from src.networks.multimemory import MultiMemoryPolicyNet
from src.core.utils import safe_load
from src.core.constants import (
    OBSERVATION_SIZE, ACTION_SIZE, VOCAB_SIZE,
    ObservationTokens, Actions
)


class Agent:
    def __init__(self,
                 network_type: str = 'lstm',
                 observation_size: int = OBSERVATION_SIZE,
                 action_size: int = ACTION_SIZE,
                 hidden_size: int = 512,
                 use_auxiliary: bool = False,
                 device: str = 'auto'):
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.network_type = network_type
        self.use_auxiliary = use_auxiliary
        
        self._validate_observation_range()
        
        if network_type == 'lstm':
            self.network = LSTMPolicyNet(
                vocab_size=VOCAB_SIZE,
                embed_dim=hidden_size,
                observation_size=observation_size,
                hidden_size=hidden_size,
                action_size=action_size,
                use_auxiliary=use_auxiliary
            )
        elif network_type == 'transformer':
            self.network = TransformerPolicyNet(
                vocab_size=VOCAB_SIZE,
                embed_dim=hidden_size,
                observation_size=observation_size,
                hidden_size=hidden_size,
                action_size=action_size,
                num_heads=8,
                num_layers=3,
                memory_size=10,
                use_auxiliary=use_auxiliary
            )
        elif network_type == 'multimemory':
            self.network = MultiMemoryPolicyNet(
                vocab_size=VOCAB_SIZE,
                embed_dim=hidden_size,
                observation_size=observation_size,
                hidden_size=hidden_size,
                action_size=action_size,
                transformer_heads=8,
                transformer_layers=3,
                cache_size=50,
                use_auxiliary=use_auxiliary
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        self.network.to(self.device)
        print(f"Created {network_type} agent:")
        print(f"  Observation size: {observation_size}")
        print(f"  Action size: {action_size}")
        print(f"  Vocab size: {VOCAB_SIZE} (tokens 0-{VOCAB_SIZE-1})")
        print(f"  Device: {device}")
    
    def _validate_observation_range(self):
        max_token = ObservationTokens.ENERGY_LEVEL_4
        if max_token != VOCAB_SIZE - 1:
            raise ValueError(f"Observation token range mismatch: "
                           f"max_token={max_token}, VOCAB_SIZE={VOCAB_SIZE}")
        print(f"✓ Observation tokens valid: 0-{max_token}")
    
    def act(self, observation: np.ndarray, training: bool = False) -> int:
        with torch.set_grad_enabled(training):
            obs_min, obs_max = observation.min(), observation.max()
            if obs_min < 0 or obs_max >= VOCAB_SIZE:
                raise ValueError(f"Observation out of range [0, {VOCAB_SIZE-1}]: "
                               f"min={obs_min}, max={obs_max}")
            
            obs_tensor = torch.from_numpy(observation).long()
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.network(obs_tensor).squeeze(1)
            
            if training:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                action = logits.argmax(dim=-1).item()
            
            if not (0 <= action < ACTION_SIZE):
                raise ValueError(f"Invalid action: {action}")
            return action
    
    def reset(self):
        if hasattr(self.network, 'reset_state'):
            self.network.reset_state()
    
    def save(self, path: str, extra_data: Dict[str, Any] = None):
        """Save agent to file, optionally with extra metadata."""
        config = {
            'network_type': self.network_type,
            'use_auxiliary': self.use_auxiliary,
            'hidden_size': self.network.hidden_size,
        }
        if hasattr(self.network, 'get_config'):
            config.update(self.network.get_config())
        else:
            config.update({
                'vocab_size': VOCAB_SIZE,
                'observation_size': OBSERVATION_SIZE,
                'action_size': ACTION_SIZE,
            })

        save_dict = {
            'state_dict': self.network.state_dict(),
            'config': config,
        }
        if extra_data is not None:
            save_dict.update(extra_data)

        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = safe_load(path, map_location=device)
        config = checkpoint['config']   # will crash if missing
        
        # Support both flat and nested (older) config structures
        if 'model' in config:
            cfg = config['model']
        else:
            cfg = config
        
        # Direct key access – crash if any missing
        network_type = cfg['network_type']
        use_auxiliary = cfg['use_auxiliary']
        hidden_size = cfg['hidden_size']
        
        observation_size = OBSERVATION_SIZE
        action_size = ACTION_SIZE
        
        agent = cls(
            network_type=network_type,
            observation_size=observation_size,
            action_size=action_size,
            hidden_size=hidden_size,
            use_auxiliary=use_auxiliary,
            device=device
        )
        
        agent.network.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded agent from {path} (strict=False)")
        return agent
    
    def test(self, env, episodes: int = 10, visualize: bool = False,
             save_video: bool = False, model_name: str = None) -> Dict[str, Any]:
        self.network.eval()
        rewards = []
        success_flags = []
        steps_list = []
        
        if model_name is None:
            model_name = f"{self.network_type}_model"
        clean_model_name = model_name.replace('/', '_').replace('\\', '_')
        
        for episode in range(episodes):
            obs, info = env.reset()
            self.reset()
            episode_reward = 0
            steps = 0
            terminated = truncated = False
            frames = []
            
            while not (terminated or truncated) and steps < env.max_steps:
                if obs.min() < 0 or obs.max() >= VOCAB_SIZE:
                    print(f"Warning: Invalid observation in episode {episode}, step {steps}")
                    obs = np.clip(obs, 0, VOCAB_SIZE - 1)
                
                action = self.act(obs, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                if visualize or save_video:
                    frame = env.render()
                    if frame is not None:
                        if save_video:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            frames.append(frame_bgr)
                        if visualize:
                            cv2.imshow('Test', frame)
                            cv2.waitKey(50)
                
                if save_video and frames and (terminated or truncated or steps == env.max_steps):
                    if frames:
                        h, w, _ = frames[0].shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_path = f'results/videos/{clean_model_name}_ep_{episode}.mp4'
                        os.makedirs(os.path.dirname(video_path), exist_ok=True)
                        if os.path.exists(video_path):
                            os.remove(video_path)
                        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                        for frame in frames:
                            video_writer.write(frame)
                        video_writer.release()
                        print(f"✓ Saved video to {video_path}")
                        frames = []
            
            rewards.append(episode_reward)
            success_flags.append(steps == env.max_steps)
            steps_list.append(steps)
            if visualize:
                cv2.waitKey(500)
        
        if visualize:
            cv2.destroyAllWindows()
        
        return {
            'rewards': rewards,
            'success_flags': success_flags,
            'steps': steps_list,
            'avg_reward': np.mean(rewards),
            'success_rate': np.mean(success_flags) * 100,
            'avg_steps': np.mean(steps_list),
            'std_reward': np.std(rewards)
        }