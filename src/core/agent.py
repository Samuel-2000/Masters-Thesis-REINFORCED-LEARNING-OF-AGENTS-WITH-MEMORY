"""
Agent class using consistent constants
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import cv2
import os

# Import network types
from src.networks.lstm import LSTMPolicyNet
from src.networks.transformer import TransformerPolicyNet
from src.networks.multimemory import MultiMemoryPolicyNet
from src.core.utils import safe_load

# Import constants
from src.core.constants import (
    OBSERVATION_SIZE, ACTION_SIZE, VOCAB_SIZE,
    ObservationTokens, Actions
)


class Agent:
    """Agent that interacts with environment using a policy network"""
    
    def __init__(self,
                 network_type: str = 'lstm',
                 observation_size: int = OBSERVATION_SIZE,  # Use constant (10)
                 action_size: int = ACTION_SIZE,  # Use constant (6)
                 hidden_size: int = 512,
                 use_auxiliary: bool = False,
                 device: str = 'auto'):
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.network_type = network_type
        self.use_auxiliary = use_auxiliary
        
        # Validate observation tokens are in range
        self._validate_observation_range()
        
        # Create network based on type with consistent sizes
        if network_type == 'lstm':
            self.network = LSTMPolicyNet(
                vocab_size=VOCAB_SIZE,  # Now 19 tokens (0-18)
                embed_dim=hidden_size,
                observation_size=observation_size,  # Always 10
                hidden_size=hidden_size,
                action_size=action_size,  # Always 6
                use_auxiliary=use_auxiliary
            )
        elif network_type == 'transformer':
            self.network = TransformerPolicyNet(
                vocab_size=VOCAB_SIZE,  # Now 19 tokens
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
                vocab_size=VOCAB_SIZE,  # Now 19 tokens
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
        
        # Debug
        print(f"Created {network_type} agent:")
        print(f"  Observation size: {observation_size}")
        print(f"  Action size: {action_size}")
        print(f"  Vocab size: {VOCAB_SIZE} (tokens 0-{VOCAB_SIZE-1})")
        print(f"  Device: {device}")
    
    def _validate_observation_range(self):
        """Validate that observation tokens are within expected range"""
        max_token = ObservationTokens.ENERGY_LEVEL_4  # Should be 18
        if max_token != VOCAB_SIZE - 1:
            raise ValueError(f"Observation token range mismatch: "
                           f"max_token={max_token}, VOCAB_SIZE={VOCAB_SIZE}")
        print(f"✓ Observation tokens valid: 0-{max_token}")
    
    def act(self, 
            observation: np.ndarray,
            training: bool = False) -> int:
        """Select action based on observation"""
        with torch.set_grad_enabled(training):
            # Validate observation range
            obs_min, obs_max = observation.min(), observation.max()
            if obs_min < 0 or obs_max >= VOCAB_SIZE:
                raise ValueError(f"Observation out of range [0, {VOCAB_SIZE-1}]: "
                               f"min={obs_min}, max={obs_max}")
            
            # Convert observation to LongTensor for embedding layer
            obs_tensor = torch.from_numpy(observation).long()
            
            # Add batch and sequence dimensions [1, 1, obs_dim]
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
            obs_tensor = obs_tensor.to(self.device)
            
            # Get action logits
            logits = self.network(obs_tensor)
            
            # Remove sequence dimension [1, action_size]
            logits = logits.squeeze(1)
            
            if training:
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                # Take greedy action
                action = logits.argmax(dim=-1).item()
            
            # Validate action is within range
            if not (0 <= action < ACTION_SIZE):
                raise ValueError(f"Invalid action: {action}")
            
            return action
    
    def reset(self):
        """Reset agent state"""
        if hasattr(self.network, 'reset_state'):
            self.network.reset_state()
    
    def save(self, path: str):
        """Save agent to file"""
        # Get configuration from network
        config = {
            'network_type': self.network_type,
            'use_auxiliary': self.use_auxiliary,
        }
        
        # Get network-specific config
        if hasattr(self.network, 'get_config'):
            config.update(self.network.get_config())
        else:
            # Fallback to constant values
            config.update({
                'vocab_size': VOCAB_SIZE,
                'observation_size': OBSERVATION_SIZE,
                'action_size': ACTION_SIZE,
                'hidden_size': self.network.hidden_size if hasattr(self.network, 'hidden_size') else 512
            })
        
        torch.save({
            'state_dict': self.network.state_dict(),
            'config': config
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'auto'):
        """Load agent from file"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = safe_load(path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Get network type, default to 'lstm' for backward compatibility
        network_type = config['network_type']
        
        # Use constants for sizes (override any saved values for consistency)
        observation_size = OBSERVATION_SIZE
        action_size = ACTION_SIZE
        hidden_size = config['hidden_size']
        use_auxiliary = config['use_auxiliary']
        
        # Create agent with consistent sizes
        agent = cls(
            network_type=network_type,
            observation_size=observation_size,
            action_size=action_size,
            hidden_size=hidden_size,
            use_auxiliary=use_auxiliary,
            device=device
        )
        
        # Load weights
        agent.network.load_state_dict(checkpoint['state_dict'])
        
        return agent
    
    def test(self, 
            env,
            episodes: int = 10,
            visualize: bool = False,
            save_video: bool = False,
            model_name: str = None) -> Dict[str, Any]:
        """Test agent performance"""
        self.network.eval()
        
        rewards = []
        success_flags = []
        steps_list = []
        
        # Extract model name if not provided
        if model_name is None:
            if hasattr(self, 'network_type'):
                model_name = f"{self.network_type}_model"
            else:
                model_name = "model"
        
        # Clean model name for filename
        clean_model_name = model_name.replace('/', '_').replace('\\', '_')
        
        for episode in range(episodes):
            obs, info = env.reset()
            self.reset()
            
            episode_reward = 0  # Track cumulative reward
            steps = 0
            terminated = truncated = False
            
            frames = []  # Reset frames for each episode
            
            while not (terminated or truncated) and steps < env.max_steps:
                # Validate observation
                if obs.min() < 0 or obs.max() >= VOCAB_SIZE:
                    print(f"Warning: Invalid observation in episode {episode}, step {steps}: "
                        f"min={obs.min()}, max={obs.max()}")
                    # Clip to valid range
                    obs = np.clip(obs, 0, VOCAB_SIZE - 1)
                
                # Get action
                action = self.act(obs, training=False)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)  # Updated to get reward
                
                episode_reward += reward  # Sum rewards
                steps += 1
                
                # Record frame if needed
                if visualize or save_video:
                    frame = env.render()
                    if frame is not None:
                        if save_video:
                            # Convert RGB to BGR for OpenCV
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
                            frames.append(frame_bgr)
                        if visualize:
                            cv2.imshow('Test', frame)
                            cv2.waitKey(50)  # ~20 FPS
                
                # Save video for THIS episode immediately
                if save_video and frames and (terminated or truncated or steps == env.max_steps):
                    if frames:
                        # Create video writer for this specific episode
                        h, w, _ = frames[0].shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        
                        # Use clean model name in video filename
                        video_path = f'results/videos/{clean_model_name}_ep_{episode}.mp4'
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(video_path), exist_ok=True)
                        
                        # Check if file exists and remove it (overwrite)
                        if os.path.exists(video_path):
                            os.remove(video_path)
                        
                        video_writer = cv2.VideoWriter(
                            video_path, fourcc, 20.0, (w, h)
                        )
                        
                        for frame in frames:
                            video_writer.write(frame)
                        
                        video_writer.release()
                        print(f"✓ Saved video to {video_path}")
                        
                        # Clear frames for next episode
                        frames = []
            
            rewards.append(episode_reward)  # Use cumulative reward
            success_flags.append(steps == env.max_steps)
            steps_list.append(steps)
            
            if visualize:
                cv2.waitKey(500)  # Pause between episodes
        
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