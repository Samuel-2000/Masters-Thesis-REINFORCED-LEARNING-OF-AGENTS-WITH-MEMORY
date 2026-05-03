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

from src.visualization.visualizer import Visualizer
from pathlib import Path

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
        # MODIFIED: Support loading from checkpoint files (which contain model_state_dict and model_config)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = safe_load(path, map_location=device)
        
        # If the file is a checkpoint (saved by trainer with model_state_dict and model_config)
        if 'model_state_dict' in checkpoint and 'model_config' in checkpoint:
            config = checkpoint['model_config']
            agent = cls(
                network_type=config['network_type'],
                observation_size=config.get('observation_size', OBSERVATION_SIZE),
                action_size=config.get('action_size', ACTION_SIZE),
                hidden_size=config['hidden_size'],
                use_auxiliary=config.get('use_auxiliary', False),
                device=device
            )
            agent.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded agent from checkpoint {path} (strict=False)")
            return agent
        
        # Otherwise assume it's a standard agent file
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
    

    def test(self, env, args, model_name: str, seed: int) -> Dict[str, Any]:
        """
        Test the agent over multiple test epochs with options from args.
        """
        self.network.eval()
        all_rewards = []
        all_success_flags = []
        all_steps = []

        if model_name is None:
            model_name = f"{self.network_type}_model"
        clean_model_name = model_name.replace('/', '_').replace('\\', '_')

        original_render_size = env.render_size
        if args.visualize or args.save_video:
            env.render_size = 512

        total_episodes = 0

        for epoch in range(args.epochs):
            obs, info = env.reset(seed=seed)
            self.reset()

            print(f"\n--- Epoch {epoch+1}/{args.epochs}: New grid (Type: {env.task_class}, Complexity: {env.complexity_level:.2f}) ---")

            for ep_in_epoch in range(args.consecutive_episodes):
                if ep_in_epoch > 0:
                    obs, info = env.soft_reset()

                # Build video filename
                vid_name = f"{clean_model_name}_{env.task_class}_comp_{env.complexity_level:.2f}_ep_{epoch}_{ep_in_epoch}"
                vid_path = Path("results/videos") / f"{vid_name}.{'gif' if args.as_gif else 'mp4'}" if args.save_video else None
                viz = Visualizer(env, args.save_video, vid_path, args.agent_view, args.fog_of_war, args.show_trail, args.as_gif)

                episode_reward = 0
                steps = 0
                terminated = truncated = False

                while not (terminated or truncated) and steps < env.max_steps:
                    if obs.min() < 0 or obs.max() >= VOCAB_SIZE:
                        print(f"Warning: Invalid observation in epoch {epoch}, episode {ep_in_epoch}, step {steps}")
                        obs = np.clip(obs, 0, VOCAB_SIZE - 1)

                    action = self.act(obs, training=False)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1

                    if args.visualize or args.save_video:
                        frame = viz.render(steps)
                        if args.visualize and frame is not None:
                            cv2.imshow('Test', frame)
                            cv2.waitKey(50)

                viz.finalize()

                all_rewards.append(episode_reward)
                all_success_flags.append(steps == env.max_steps)
                all_steps.append(steps)
                total_episodes += 1

        env.render_size = original_render_size
        if args.visualize:
            cv2.destroyAllWindows()

        return {
            'rewards': all_rewards,
            'success_flags': all_success_flags,
            'steps': all_steps,
            'avg_reward': np.mean(all_rewards) if all_rewards else 0,
            'success_rate': np.mean(all_success_flags) * 100 if all_success_flags else 0,
            'avg_steps': np.mean(all_steps) if all_steps else 0,
            'std_reward': np.std(all_rewards) if all_rewards else 0,
            'total_episodes': total_episodes
        }