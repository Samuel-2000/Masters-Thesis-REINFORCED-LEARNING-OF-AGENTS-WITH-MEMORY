# src/training/trainer.py - OPTIMIZED PARALLEL VERSION WITH DYNAMIC COMPLEXITY
"""
Training module with optimizations, parallel execution, and dynamic complexity
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Deque
import time
from datetime import datetime
from collections import deque

from src.core.environment import GridMazeWorld
from src.core.vector_env import VectorizedMazeEnv
from src.core.agent import Agent
from src.core.utils import setup_logging, seed_everything
from .losses import PolicyLoss, AuxiliaryLoss
from .optimizers import GradientClipper, LearningRateScheduler, OptimizerFactory

import cv2

from src.core.constants import (
    OBSERVATION_SIZE, ACTION_SIZE
)
# ============================================================================
# DYNAMIC COMPLEXITY MANAGER
# ============================================================================

class ComplexityManager:
    """Manages dynamic complexity adjustment based on agent performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config['training']
        
        # Dynamic complexity settings
        self.enabled = self.training_config['dynamic_complexity']
        self.performance_window = self.training_config['performance_window']
        self.increase_threshold = self.training_config['complexity_increase_threshold']
        self.decrease_threshold = self.training_config['complexity_decrease_threshold']
        self.complexity_step = self.training_config['complexity_step']
        self.min_complexity = self.training_config['min_complexity']
        self.max_complexity = self.training_config['max_complexity']
        self.adjustment_interval = self.training_config['adjustment_interval']
        self.stagnation_switch_interval = self.training_config['stagnation_switch_interval']
        self.stagnation_termination = self.training_config['stagnation_termination']
        self.min_basic_complexity = self.training_config['min_basic_complexity']
        self.curriculum_stages = self.training_config['curriculum_stages']
        
        # Current state - each stage maintains its own complexity
        self.current_stage_idx = 0
        self.stage_complexities = {stage: 0.0 for stage in self.curriculum_stages}
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.performance_window)
        self._last_performance_score = 0.0
        self.max_rewards_by_stage = {}
        self.epochs_without_progress = 0  # Tracks stagnation
        self.last_max_reward = -float('inf')
        self.last_stagnation_check = 0  # NEW: track when we last checked for stagnation
        
        # Statistics
        self.adjustments_made = 0
        self.stage_switches = 0  # Track how many times we switch tasks

    def add_performance(self, reward: float, epoch: int):
        """Add performance metric to history"""
        self.performance_history.append(reward)
        
        # Check for progress in current stage
        current_max = self.max_rewards_by_stage.get(self.current_stage_idx, reward)
        if reward > current_max:
            self.max_rewards_by_stage[self.current_stage_idx] = reward
            self.epochs_without_progress = 0
            self.last_max_reward = reward
        else:
            self.epochs_without_progress += 1
    
    def get_current_task_class(self) -> str:
        """Get current task class based on curriculum stage"""
        if not self.enabled or self.current_stage_idx >= len(self.curriculum_stages):
            return self.config['environment']['task_class']
        return self.curriculum_stages[self.current_stage_idx]
    
    def get_current_complexity(self) -> float:
        """Get current complexity level"""
        if not self.enabled:
            return self.config['environment']['complexity_level']
        
        current_stage = self.get_current_task_class()
        return self.stage_complexities[current_stage]
    
    def should_adjust(self, epoch: int) -> bool:
        """Check if we should adjust complexity"""
        if not self.enabled:
            return False
        
        # Check if enough time has passed since last adjustment
        if epoch % self.adjustment_interval != 0:
            return False
        
        # Check if we have enough performance data
        if len(self.performance_history) < self.performance_window // 2: # at the beginning of the training
            return False
        
        return True
    
    def should_switch_stage(self, epoch: int) -> bool:
        """Check if we should switch to a different task due to stagnation"""
        if not self.enabled:
            return False
        
        # NEW: Only check stagnation every stagnation_switch_interval epochs
        if epoch - self.last_stagnation_check < self.stagnation_switch_interval:
            return False
        
        self.last_stagnation_check = epoch  # Update last check time
        
        current_stage = self.get_current_task_class()
        current_complexity = self.stage_complexities[current_stage]
        
        # Prevent switching from basic until it reaches minimum complexity
        if current_stage == 'basic' and current_complexity < self.min_basic_complexity:
            return False
        
        # Switch if we've been stuck for too long
        if self.epochs_without_progress >= self.stagnation_termination:
            return True
        
        # Optional: Check for oscillation (could be enhanced with more sophisticated detection)
        if len(self.performance_history) >= self.performance_window:
            recent_std = np.std(list(self.performance_history)[-self.performance_window:])
            recent_mean = np.mean(list(self.performance_history)[-self.performance_window:])
            if recent_std < 0.1 and recent_mean < self.decrease_threshold:
                return True  # Stuck at low performance with little variation
        
        return False

    def calculate_performance_score(self) -> float:
        if not self.performance_history:
            return self._last_performance_score

        avg_performance = np.mean(list(self.performance_history)[-self.performance_window:])
        
        # Track maximum reward seen for current stage
        stage_idx = self.current_stage_idx
        
        if stage_idx not in self.max_rewards_by_stage:
            self.max_rewards_by_stage[stage_idx] = avg_performance
        else:
            self.max_rewards_by_stage[stage_idx] = max(self.max_rewards_by_stage[stage_idx], avg_performance)
        
        max_reward = self.max_rewards_by_stage[stage_idx]
        
        # Avoid division by zero
        if max_reward < 0.1:
            return 0.0
        
        # Normalize by maximum observed reward for current stage
        score = max(0.0, min(1.0, avg_performance / max_reward))
        self._last_performance_score = score
        return score

    def switch_to_next_stage(self) -> Dict[str, Any]:
        """Switch to next task in curriculum, maintaining each task's complexity"""
        old_stage_idx = self.current_stage_idx
        old_stage = self.curriculum_stages[old_stage_idx]
        old_complexity = self.stage_complexities[old_stage]
        
        # Calculate weighted probabilities for next stage
        # Higher probability to stay in current stage or progress to next
        # But always allow some probability to return to earlier stages for consolidation
        
        stage_weights = np.zeros(len(self.curriculum_stages))
        
        # Base weight for each stage depends on how recently we visited it
        # and its current complexity level
        for i, stage in enumerate(self.curriculum_stages):
            # Recent visit penalty - avoid bouncing too quickly
            recent_penalty = 1.0
            if i == old_stage_idx:
                recent_penalty = 0.3  # Less likely to stay in same stage if stagnating
            elif i == (old_stage_idx - 1) % len(self.curriculum_stages):
                recent_penalty = 0.5  # Less likely to go back immediately
            
            # Complexity-based weight - prefer stages at moderate complexity
            stage_complexity = self.stage_complexities[stage]
            complexity_weight = 1.0 - abs(stage_complexity - 0.5)  # Prefer ~0.5 complexity
            
            # Distance weight - prefer next stage in curriculum
            distance = (i - old_stage_idx) % len(self.curriculum_stages)
            if distance == 1:
                distance_weight = 2.0  # Most likely: progress forward
            elif distance == 0:
                distance_weight = 1.0  # Same stage
            else:
                distance_weight = 0.7  # Going backward or skipping
            
            stage_weights[i] = recent_penalty * complexity_weight * distance_weight
        
        # Normalize to probabilities
        stage_probs = stage_weights / stage_weights.sum()
        
        # Sample next stage
        next_idx = np.random.choice(len(self.curriculum_stages), p=stage_probs)
        
        self.current_stage_idx = next_idx
        new_stage = self.curriculum_stages[next_idx]
        new_complexity = self.stage_complexities[new_stage]
        
        # Reset stagnation tracking
        self.epochs_without_progress = 0
        self.last_max_reward = -float('inf')
        self.performance_history.clear()
        self.stage_switches += 1
        
        adjustment_info = {
            "action": "switched_stage_due_to_stagnation",
            "old_stage": old_stage,
            "new_stage": new_stage,
            "old_complexity": old_complexity,
            "new_complexity": new_complexity,
            "reason": f"Stagnation for {self.epochs_without_progress} epochs",
            "stage_probs": stage_probs.tolist()  # For debugging
        }
        
        return adjustment_info
    
    def adjust_complexity(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Adjust complexity based on performance, with task switching for stagnation"""
        if not self.should_adjust(epoch):
            return None
        
        # First check if we should switch tasks due to stagnation
        if self.should_switch_stage(epoch):  # NEW: Pass epoch parameter
            adjustment_info = self.switch_to_next_stage()
            self.adjustments_made += 1
            return adjustment_info
        
        # Otherwise, proceed with normal complexity adjustment
        performance_score = self.calculate_performance_score()
        current_stage = self.get_current_task_class()
        old_complexity = self.stage_complexities[current_stage]

        if self.decrease_threshold <= performance_score <= self.increase_threshold:
            return None
        
        adjustment_info = {
            "performance_score": performance_score,
            "old_complexity": old_complexity,
            "old_stage": current_stage,
            "new_stage": current_stage,
        }
        
        if performance_score > self.increase_threshold:
            if old_complexity >= self.max_complexity:
                return None
            
            new_complexity = min(self.max_complexity, old_complexity + self.complexity_step)
            adjustment_info["action"] = "increased_complexity"
            self.epochs_without_progress = 0

        elif performance_score < self.decrease_threshold:
            if old_complexity <= self.min_complexity:
                return None
            
            new_complexity = max(self.min_complexity, old_complexity - self.complexity_step)
            adjustment_info["action"] = "decreased_complexity"
        else:
            raise

        adjustment_info["new_complexity"] = new_complexity
        self.stage_complexities[current_stage] = new_complexity
        self.adjustments_made += 1
        self.performance_history.clear()  # Reset performance history after adjustment
        
        return adjustment_info
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration with current complexity settings"""
        env_config = self.config['environment'].copy()
        
        if self.enabled:
            current_stage = self.get_current_task_class()
            env_config['task_class'] = current_stage
            env_config['complexity_level'] = self.stage_complexities[current_stage]
            
            # CRITICAL: Set door parameters appropriately for each stage
            if current_stage == 'basic':
                # Basic stage: no doors
                env_config['n_doors'] = 0
                env_config['n_buttons_per_door'] = 0
                env_config['button_break_probability'] = 0.0
                
            elif current_stage == 'doors':
                # Doors stage: periodic doors, no buttons
                env_config['n_doors'] = -1
                env_config['n_buttons_per_door'] = 0
                env_config['button_break_probability'] = 0.0
                
            elif current_stage == 'buttons':
                # Buttons stage: doors with buttons
                env_config['n_doors'] = -1
                env_config['n_buttons_per_door'] = -1
                env_config['button_break_probability'] = -1.0
                
            elif current_stage == 'complex':
                # Complex stage: mix of periodic and button doors
                env_config['n_doors'] = -1
                env_config['n_buttons_per_door'] = -1
                env_config['button_break_probability'] = -1.0
        
        return env_config
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of complexity manager"""
        current_stage = self.get_current_task_class()
        
        return {
            "enabled": self.enabled,
            "current_stage": current_stage,
            "current_complexity": self.stage_complexities[current_stage],
            "all_stage_complexities": self.stage_complexities,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.curriculum_stages),
            "performance_history_size": len(self.performance_history),
            "adjustments_made": self.adjustments_made,
            "stage_switches": self.stage_switches,
            "epochs_without_progress": self.epochs_without_progress,
            "last_stagnation_check": self.last_stagnation_check,
            "performance_score": self.calculate_performance_score(), # if self.performance_history else 0.0,
            "max_rewards_by_stage": self.max_rewards_by_stage,
            "basic_min_complexity_reached": self.stage_complexities['basic'] >= self.min_basic_complexity
        }


# ============================================================================
# BASE TRAINER (NO DYNAMIC COMPLEXITY)
# ============================================================================

class ParallelTrainer:
    """Base trainer with parallel execution, fixed environment (no dynamic complexity)"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 use_wandb: bool = False):
        
        self.config = config
        self.experiment_name = f"{config['model']['type']}_" \
                                f"{config['training']['batch_size']}b_" \
                                f"{config['training']['learning_rate']}lr_" \
                                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        if config['training']['auxiliary_tasks']:
            self.experiment_name += "_aux"
        
        self.use_wandb = use_wandb
        
        # Setup
        self.logger = setup_logging(self.experiment_name)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Set seed
        seed_everything(config['experiment']['seed'])
        
        # Get batch size for parallel environments
        training_config = self.config['training']
        self.batch_size = training_config['batch_size']
        
        # Create initial vectorized environment (fixed, from config)
        self.vector_env = self._create_vectorized_env()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        
        # Create loss functions
        self.policy_loss_fn = PolicyLoss(
            gamma=training_config['gamma'],
            entropy_coef=training_config['entropy_coef'],
            normalize_advantages=True
        )
        
        # Only create auxiliary loss if configured
        if self.agent.use_auxiliary:
            self.aux_loss_fn = AuxiliaryLoss(
                energy_coef=0.1,
                obs_coef=0.05,
                obs_prediction_type='classification'
            )
        else:
            self.aux_loss_fn = None
            
        self.gradient_clipper = GradientClipper(
            max_norm=training_config['max_grad_norm']
        )
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            mode='cosine',
            lr_start=training_config['learning_rate'],
            lr_min=1e-6
        )
        
        # Metrics
        self.metrics = {
            'train_rewards': [],
            'train_losses': [],
            'aux_losses': [],
            'energy_losses': [],
            'obs_losses': [],
            'test_rewards': [],
            'best_reward': -np.inf,
            'timing': {
                'collection': [],
                'training': [],
                'total': []
            }
        }


        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project="maze-rl",
                name=self.experiment_name,
                config=config
            )

    def _create_vectorized_env(self) -> VectorizedMazeEnv:
        """Create vectorized training environment from config (fixed)"""
        env_config = self.config['environment'].copy()
        return VectorizedMazeEnv(
            num_envs=self.batch_size,
            env_config=env_config
        )
    
    def _create_agent(self) -> Agent:
        """Create agent with specified network"""
        model_config = self.config['model']
        
        agent = Agent(
            network_type=model_config['type'],
            observation_size=OBSERVATION_SIZE,  # Fixed observation size
            action_size=ACTION_SIZE,  # Fixed action size
            hidden_size=model_config['hidden_size'],
            use_auxiliary=model_config['use_auxiliary'],
            device=self.device
        )
        
        # Load pretrained if specified
        if 'pretrained_path' in model_config:
            agent.load(model_config['pretrained_path'])
        
        return agent
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        training_config = self.config['training']
        
        optimizer_type = training_config['optimizer']
        lr = training_config['learning_rate']
        weight_decay = training_config['weight_decay']
        
        return OptimizerFactory.create(
            optimizer_name=optimizer_type,
            parameters=self.agent.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    
    def _collect_experiences_parallel(self) -> Dict[str, torch.Tensor]:
        """
        Collect experiences in parallel across all environments
        """
        max_steps = self.vector_env.envs[0].max_steps
        self.agent.reset()  # Reset network state once
        
        # Reset all environments
        obs_array, _ = self.vector_env.reset()
        
        # Convert to tensor
        observations = torch.tensor(obs_array, dtype=torch.long).to(self.device)
        observations = observations.unsqueeze(1)  # [B, 1, K]
        
        # Storage
        all_observations = []
        all_actions = []
        all_rewards = []
        all_energies = []          # energy before each step (target)
        all_next_obs = []          # observation after each step
        
        # Run for max_steps or until all environments are done
        active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        current_energies = [env.energy for env in self.vector_env.envs]
        
        for step in range(max_steps):
            # Store current observations
            all_observations.append(observations.clone())
            all_energies.append(torch.tensor(current_energies, dtype=torch.float32, device=self.device))
            
            # Get actions from network
            with torch.no_grad():
                # Ensure network is in eval mode for inference
                was_training = self.agent.network.training
                self.agent.network.eval()
                
                logits = self.agent.network(observations)  # [B, 1, A]
                logits = logits.squeeze(1)  # [B, A]
                
                if was_training:
                    self.agent.network.train()
                
                # Sample actions during training
                if self.agent.network.training:
                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)  # [B]
                else:
                    actions = logits.argmax(dim=-1)  # [B]
            
            # Convert to numpy for environment step
            actions_np = actions.cpu().numpy()
            
            # Step all environments in parallel
            obs_array, rewards, terminated, truncated, infos = self.vector_env.step(actions_np)
            
            # Convert to tensors, Store actions and rewards
            next_obs_tensor = torch.tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)
            all_next_obs.append(next_obs_tensor.clone())
            all_actions.append(actions)
            all_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=self.device))

            current_energies = [info.get('energy', 0.0) for info in infos]
            
            # Update active mask
            done_mask = torch.tensor(terminated | truncated, dtype=torch.bool, device=self.device)
            active_mask = active_mask & ~done_mask
            
            # Break if all environments are done
            if not active_mask.any():
                break
            
            # Prepare next observations
            observations = next_obs_tensor  # [B, 1, K]
        
        # Stack all collected data   
        return {
            'observations': torch.cat(all_observations, dim=1),            # [B,T,K]
            'actions': torch.stack(all_actions, dim=1),           # [B,T]
            'rewards': torch.stack(all_rewards, dim=1),           # [B,T]
            'mask': torch.ones_like(torch.stack(all_rewards, dim=1)),  # [B,T]
            'energy_targets': torch.stack(all_energies, dim=1),   # [B,T]
            'next_obs_targets': torch.cat(all_next_obs, dim=1),   # [B,T,K]
        }
    def _compute_loss(self, 
                     experiences: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Compute policy loss with auxiliary losses"""
        obs = experiences['observations']
        actions = experiences['actions']
        rewards = experiences['rewards']
        mask = experiences.get('mask', None)
        
        # Reset network for batch processing
        self.agent.reset()
        
        # Forward pass
        if self.aux_loss_fn and self.agent.use_auxiliary:
            # Get outputs from network with auxiliary predictions
            outputs = self.agent.network(obs, return_auxiliary=True)
            
            if isinstance(outputs, tuple):
                logits, energy_pred, obs_pred = outputs[0], outputs[1], outputs[2]
            else: # Network doesn't return auxiliary outputs
                logits = outputs
                energy_pred = obs_pred = None
            
            policy_loss, entropy = self.policy_loss_fn(logits, actions, rewards, mask)
            
            energy_target = experiences['energy_targets']               # [B,T]
            obs_target = experiences['next_obs_targets']                # [B,T,K]

            # Regression auxiliary loss (observation prediction MSE)
            aux_loss = self.aux_loss_fn(energy_pred, energy_target, obs_pred, obs_target.float(), mask)
            total_loss = policy_loss + aux_loss
            
            metrics = {
                'loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'aux_loss': aux_loss.item(),
                'energy_loss': (energy_pred - energy_target.unsqueeze(-1)).pow(2).mean().item(),
                'obs_loss': (obs_pred - obs_target.float()).pow(2).mean().item(),
                'entropy': entropy.item(),
                'reward': rewards.sum(dim=1).mean().item(),
            }
        else:
            # Standard forward pass without auxiliary tasks
            logits = self.agent.network(obs)
            policy_loss, entropy = self.policy_loss_fn(
                logits, actions, rewards, mask
            )
            
            total_loss = policy_loss
            
            metrics = {
                'loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy': entropy.item(),
                'reward': rewards.sum(dim=1).mean().item()
            }
        
        return total_loss, metrics
    
    def _train_step(self, 
                   experiences: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step"""
        self.agent.network.train()
        
        # Compute loss
        loss, metrics = self._compute_loss(experiences)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.gradient_clipper.clip(self.agent.network.parameters())
        self.optimizer.step()
        
        # Flush cache if using multi-memory
        if hasattr(self.agent.network, 'flush_cache_buffer'):
            self.agent.network.flush_cache_buffer()
        
        return metrics
    
    def get_environment_config(self):
        """Get the current environment configuration (static for base trainer)"""
        return self.config['environment'].copy()
    
    def _test_valid(self, epochs: int = 10) -> Dict[str, float]:
        """Run `epochs` test epochs, each processing `batch_size` parallel epochs."""
        self.agent.network.eval()
        env_config = self.get_environment_config()
        env_config['render_size'] = 0

        total_epochs = epochs * self.batch_size
        all_rewards = []
        all_lengths = []
        max_steps = 0

        for _ in range(epochs):
            test_env = VectorizedMazeEnv(num_envs=self.batch_size, env_config=env_config)
            max_steps = test_env.envs[0].max_steps
            obs_array, _ = test_env.reset()
            obs_t = torch.tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)

            rewards = np.zeros(self.batch_size)
            lengths = np.zeros(self.batch_size, dtype=int)

            with torch.no_grad():
                for step in range(max_steps):
                    logits = self.agent.network(obs_t).squeeze(1)
                    actions = logits.argmax(dim=-1).cpu().numpy()
                    obs_array, r, terminated, truncated, _ = test_env.step(actions)
                    obs_t = torch.tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)
                    rewards += r
                    lengths += 1
                    if (terminated | truncated).all():
                        break
            test_env.close()
            all_rewards.extend(rewards)
            all_lengths.extend(lengths)

        avg_reward = np.mean(all_rewards)
        avg_length = np.mean(all_lengths)
        success_rate = np.sum(np.array(all_lengths) == max_steps) / total_epochs * 100

        return {
            'reward': avg_reward,
            'success_rate': success_rate,
            'avg_length': avg_length
        }
    
    def _save_model(self, name: str):
        save_dir = Path(self.config['experiment']['save_dir'])
        save_dir.mkdir(exist_ok=True)

        # Save the agent model using agent.save()
        if name in ['best', 'final']:
            agent_path = save_dir / f"{self.experiment_name}_{name}.pt"
            self.agent.save(str(agent_path))          # no extra_data needed for base trainer
            self.logger.info(f"Saved agent to {agent_path}")

        # Save checkpoint (optimizer, scheduler, metrics) for resuming
        checkpoint_path = save_dir / f"{self.experiment_name}_{name}_checkpoint.pt"
        checkpoint = {
            'epoch': len(self.metrics['train_rewards']),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.lr_scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        torch.save(checkpoint, str(checkpoint_path))
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_metrics(self):
        """Save training metrics"""
        metrics_dir = Path('logs/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = metrics_dir / f"{self.experiment_name}_metrics.npz"
        
        np.savez(str(metrics_path),
                train_rewards=self.metrics['train_rewards'],
                train_losses=self.metrics['train_losses'],
                test_rewards=self.metrics['test_rewards'],
                timing_collection=self.metrics['timing']['collection'],
                timing_training=self.metrics['timing']['training'])
        
        # Plot metrics
        self._plot_metrics()


    def _plot_metrics(self):
        """Save each plot as a separate file in a folder named after the experiment."""
        import matplotlib.pyplot as plt

        # Create directory for this experiment's plots
        plots_dir = Path('results/plots') / self.experiment_name
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Helper to save a plot
        def save_plot(fig, name):
            path = plots_dir / f"{name}.png"
            fig.savefig(str(path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Saved plot to {path}")


        # ---- 1. Training Rewards (raw + smoothed) + Test Rewards ----
        fig, ax = plt.subplots(figsize=(8, 5))
        rewards = np.array(self.metrics['train_rewards'])
        epochs = np.arange(len(rewards))
        ax.plot(epochs, rewards, alpha=0.5, color='gray', linewidth=0.5, label='Train Reward (raw)')
        
        # Smoothed with expanding window until size reaches 100, then rolling window of 100
        if len(rewards) > 0:
            smoothed = np.zeros_like(rewards, dtype=float)
            window = 100
            for i in range(len(rewards)):
                if i < window:
                    # Expanding average: average all rewards up to i
                    smoothed[i] = np.mean(rewards[:i+1])
                else:
                    # Rolling window of size window
                    smoothed[i] = np.mean(rewards[i-window+1:i+1])
            ax.plot(epochs, smoothed, 'b-', linewidth=2, label=f'Train Reward (smoothed, window={window})')
        
        # Test rewards (if available)
        if 'test_rewards' in self.metrics and len(self.metrics['test_rewards']) > 0:
            test_rewards = self.metrics['test_rewards']
            test_interval = self.config['training']['test_interval']
            # Ensure x-values start at 0 and align with the epochs when tests were performed
            # First test is at epoch 0, then at test_interval, 2*test_interval, ...
            x_vals = np.arange(0, len(test_rewards) * test_interval, test_interval)
            # If test_interval is 0? Not possible. Also handle case where first test not at epoch 0? Our training loop does epoch % test_interval == 0, so epoch 0 is included.
            ax.plot(x_vals, test_rewards, 'g-o', linewidth=1.5, markersize=6, label='Test Reward')
        
        ax.set_title('Training & Test Rewards')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_plot(fig, 'rewards')

        # ---- 2. Training Losses (total + policy) ----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.metrics['train_losses'], 'r-', alpha=0.7, label='Total Loss')
        if 'policy_losses' in self.metrics and len(self.metrics['policy_losses']) > 0:
            ax.plot(self.metrics['policy_losses'], 'r--', alpha=0.5, label='Policy Loss')
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_plot(fig, 'losses')

        # ---- 2. Auxiliary Losses (if any) ----
        has_aux = 'aux_losses' in self.metrics and len(self.metrics['aux_losses']) > 0
        if has_aux:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.metrics['aux_losses'], label='Total Aux Loss', color='purple')
            if 'energy_losses' in self.metrics and len(self.metrics['energy_losses']) > 0:
                ax.plot(self.metrics['energy_losses'], label='Energy MSE', color='orange')
            if 'obs_losses' in self.metrics and len(self.metrics['obs_losses']) > 0:
                ax.plot(self.metrics['obs_losses'], label='Obs MSE', color='green')
            ax.set_title('Auxiliary Losses')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            save_plot(fig, 'aux_losses')

        # ---- 3. Complexity & Task Class Progression (if complexity enabled) ----
        has_complexity = 'complexity_history' in self.metrics and len(self.metrics['complexity_history']) > 0
        if has_complexity:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.metrics['complexity_history'], 'b-', linewidth=2, label='Complexity')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Complexity Level', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            # Task class overlay if available
            if 'task_class_history' in self.metrics and len(self.metrics['task_class_history']) > 0:
                ax2 = ax.twinx()
                stage_map = {'basic': 0.0, 'doors': 0.33, 'buttons': 0.66, 'complex': 1.0}
                task_numeric = [stage_map.get(s, 0.0) for s in self.metrics['task_class_history']]
                ax2.plot(task_numeric, 'g--', linewidth=1.5, label='Task Class', alpha=0.9)
                ax2.set_ylabel('Task Class', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                ax2.set_yticks([0.0, 0.33, 0.66, 1.0])
                ax2.set_yticklabels(['Basic', 'Doors', 'Buttons', 'Complex'])
                ax2.set_ylim(-0.1, 1.1)
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                ax.legend()
            ax.set_title('Complexity Progression')
            save_plot(fig, 'complexity')

        # ---- 4. Performance Scores (x-axis multiplied by window) ----
        has_perf_scores = 'performance_scores' in self.metrics and len(self.metrics['performance_scores']) > 0
        if has_perf_scores:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            scores = np.array(self.metrics['performance_scores'])
            # Multiply index by performance_window to get actual epoch
            window = self.complexity_manager.performance_window  # e.g., 10
            epochs = np.arange(len(scores)) * window  # [0, 10, 20, ...]
            
            # Complexity and task histories are full-length; we need values at these epochs
            full_complexities = np.array(self.metrics['complexity_history'])
            full_tasks = np.array(self.metrics['task_class_history'])
            # Index by epoch (clip to avoid out-of-bound)
            complexities = full_complexities[epochs]
            tasks = full_tasks[epochs]
            
            # Detect change points (where complexity or task changes between consecutive scores)
            change_indices = []
            for i in range(1, len(complexities)):
                if abs(complexities[i] - complexities[i-1]) > 1e-6 or tasks[i] != tasks[i-1]:
                    change_indices.append(i)
            
            # Plot segments with alternating colors
            colors = ['orange', 'blue']
            start = 0
            for idx, split in enumerate(change_indices):
                seg_x = epochs[start:split]
                seg_y = scores[start:split]
                if len(seg_x) > 0:
                    ax.plot(seg_x, seg_y, color=colors[idx % 2], linewidth=1.5)
                    ax.axvline(x=epochs[split], color='gray', linestyle=':', alpha=0.5)
                start = split
            # Last segment
            if start < len(epochs):
                ax.plot(epochs[start:], scores[start:], 
                        color=colors[len(change_indices) % 2], linewidth=1.5)
            
            # Threshold lines
            if hasattr(self, 'complexity_manager') and self.complexity_manager is not None:
                inc = self.complexity_manager.increase_threshold
                dec = self.complexity_manager.decrease_threshold
                ax.axhline(inc, color='green', linestyle='--', alpha=0.7, label='Increase')
                ax.axhline(dec, color='red', linestyle='--', alpha=0.7, label='Decrease')
            
            ax.set_title('Performance Scores (colored by config change)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            save_plot(fig, 'performance_scores')

        # ---- 5. Complexity vs Reward Correlation (if enough data) ----
        if has_complexity and len(self.metrics['train_rewards']) > 100:
            fig, ax = plt.subplots(figsize=(8, 5))
            window = min(100, len(self.metrics['train_rewards']) // 10)
            if window > 1:
                smoothed_r = np.convolve(self.metrics['train_rewards'],
                                        np.ones(window)/window, mode='valid')
                smoothed_c = np.convolve(self.metrics['complexity_history'],
                                        np.ones(window)/window, mode='valid')
                min_len = min(len(smoothed_r), len(smoothed_c))
                sc = ax.scatter(smoothed_c[:min_len], smoothed_r[:min_len],
                                c=range(min_len), cmap='viridis', alpha=0.7, s=15)
                plt.colorbar(sc, ax=ax, label='Epoch (smoothed)')
                correlation = np.corrcoef(smoothed_c[:min_len], smoothed_r[:min_len])[0, 1]
                ax.set_title(f'Complexity vs Reward (corr: {correlation:.3f})')
            else:
                ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Complexity Level')
            ax.set_ylabel('Smoothed Reward')
            ax.grid(True, alpha=0.3)
            save_plot(fig, 'complexity_vs_reward')

        # ---- 6. Reward vs Complexity per stage (smoothed, matching complexity_vs_reward) ----
        has_stage_highlight = ('task_class_history' in self.metrics and 
                            len(self.metrics['task_class_history']) > 0)
        if has_stage_highlight:
            stage_order = ['basic', 'doors', 'buttons', 'complex']
            unique_stages = [s for s in stage_order if s in self.metrics['task_class_history']]
            
            # Extract raw data
            rewards_raw = np.array(self.metrics['train_rewards'])
            complexities_raw = np.array(self.metrics['complexity_history'])
            stages_raw = np.array(self.metrics['task_class_history'])
            
            # Apply same smoothing as in global complexity_vs_reward plot
            window = min(100, len(rewards_raw) // 10)
            if window > 1:
                # Convolve with box filter of length window (valid mode)
                kernel = np.ones(window) / window
                smoothed_rewards = np.convolve(rewards_raw, kernel, mode='valid')
                smoothed_complexities = np.convolve(complexities_raw, kernel, mode='valid')
                # Map stage labels to smoothed indices: smoothed index i corresponds to raw epoch i + window - 1
                offset = window - 1
                stages_smoothed = stages_raw[offset:offset + len(smoothed_rewards)]
                # Epoch numbers for smoothed data (starting from offset)
                epochs_smoothed = np.arange(offset, offset + len(smoothed_rewards))
            else:
                # Not enough data for smoothing; use raw data
                smoothed_rewards = rewards_raw
                smoothed_complexities = complexities_raw
                stages_smoothed = stages_raw
                epochs_smoothed = np.arange(len(rewards_raw))
                window = 1
            
            # For each stage, create a plot
            for stage in unique_stages:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Mask for this stage (on smoothed indices)
                mask = (stages_smoothed == stage)
                
                # Plot all points as faint background (all smoothed points)
                ax.scatter(smoothed_complexities, smoothed_rewards,
                           c='gray', alpha=0.2, s=10, label='All epochs (smoothed)')
                
                # Highlight this stage's points with color mapped to epoch progression
                sc = ax.scatter(smoothed_complexities[mask], smoothed_rewards[mask],
                                c=epochs_smoothed[mask], cmap='viridis',
                                alpha=0.8, s=40, label=f'{stage.capitalize()} active')
                
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label('Epoch (original index)')
                
                # Optional: trend line for this stage
                if np.sum(mask) > 1:
                    x_vals = smoothed_complexities[mask]
                    y_vals = smoothed_rewards[mask]
                    # Remove any NaN or inf values
                    valid = ~(np.isnan(x_vals) | np.isnan(y_vals) | 
                              np.isinf(x_vals) | np.isinf(y_vals))
                    x_vals = x_vals[valid]
                    y_vals = y_vals[valid]
                    if len(x_vals) >= 2 and np.std(x_vals) > 1e-6:
                        try:
                            z = np.polyfit(x_vals, y_vals, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                            ax.plot(x_line, p(x_line), 'r--', linewidth=2,
                                    label=f'Trend: {z[0]:.2f}*x + {z[1]:.2f}')
                        except (np.linalg.LinAlgError, ValueError):
                            # Skip if fit fails (e.g., singular matrix)
                            pass
                
                ax.set_title(f'Reward vs Complexity – {stage.capitalize()} stage (smoothed, w={window})')
                ax.set_xlabel('Complexity Level (smoothed)')
                ax.set_ylabel('Reward (smoothed)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                save_plot(fig, f'reward_vs_complexity_stage_{stage}')

    def _print_training_summary(self, start_time: float):
        """Print training summary"""
        total_time = time.time() - start_time
        avg_collection = np.mean(self.metrics['timing']['collection'])
        avg_training = np.mean(self.metrics['timing']['training'])
        
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average epoch time: {avg_collection+avg_training:.3f}s (collection: {avg_collection:.3f}s, training: {avg_training:.3f}s)")
        print(f"Average environment steps per second: {self.batch_size/avg_collection:.1f}")
        print(f"Final best reward: {self.metrics['best_reward']:.2f}")
        print(f"Model saved as: {self.experiment_name}_best.pt")
        print(f"{'='*80}")
    
    def train(self):
        """Main training loop (no dynamic complexity)"""
        training_config = self.config['training']
        epochs = training_config['epochs']
        save_interval = training_config['save_interval']
        test_interval = training_config['test_interval']
        
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        start_time = time.time()
        
        for epoch in pbar:
            epoch_start = time.time()
            coll_start = time.time()
            experiences = self._collect_experiences_parallel()
            coll_time = time.time() - coll_start
            
            train_start = time.time()
            train_metrics = self._train_step(experiences)
            train_time = time.time() - train_start
            
            epoch_reward = train_metrics['reward']
            
            self.metrics['train_rewards'].append(epoch_reward)
            self.metrics['train_losses'].append(train_metrics['loss'])
            self.metrics['timing']['collection'].append(coll_time)
            self.metrics['timing']['training'].append(train_time)
            self.metrics['timing']['total'].append(time.time() - epoch_start)
            
            if epoch % test_interval == 0:# and epoch != 0:
                test_metrics = self._test_valid(epochs=4)
                test_reward = test_metrics['reward']
                self.metrics['test_rewards'].append(test_reward)
                if test_reward > self.metrics['best_reward']:
                    self.metrics['best_reward'] = test_reward
                    self._save_model('best')
                    self.logger.info(f"New best model with reward: {test_reward:.2f}")
            
            if epoch % save_interval == 0 and epoch != 0:
                self._save_model(f'epoch_{epoch:06d}')
            
            avg_coll = np.mean(self.metrics['timing']['collection'][-10:]) if self.metrics['timing']['collection'] else coll_time
            avg_train = np.mean(self.metrics['timing']['training'][-10:]) if self.metrics['timing']['training'] else train_time
            
            pbar.set_postfix({
                'reward': f"{train_metrics['reward']:.2f}",
                'loss': f"{train_metrics['loss']:.4f}",
                'best': f"{self.metrics['best_reward']:.2f}",
                'eps/s': f"{self.batch_size/(avg_coll+avg_train):.1f}",
            })
            
            if self.use_wandb:
                wandb.log({
                    'train/reward': train_metrics['reward'],
                    'train/loss': train_metrics['loss'],
                    'lr': self.lr_scheduler.get_lr(),
                })
                if epoch % test_interval == 0 and epoch != 0:
                    wandb.log({'test/reward': test_metrics['reward']})
            
            self.lr_scheduler.step()

        test_metrics = self._test_valid(epochs=4)
        test_reward = test_metrics['reward']
        self.metrics['test_rewards'].append(test_reward)
        if test_reward > self.metrics['best_reward']:
            self.metrics['best_reward'] = test_reward
            self._save_model('best')
            self.logger.info(f"New best model with reward: {test_reward:.2f}")

        
        self._save_model('final')
        self._save_metrics()
        self._print_training_summary(start_time)
        if self.use_wandb:
            wandb.finish()


# ============================================================================
# ADAPTIVE TRAINER (inherits from ParallelTrainer)
# ============================================================================

class AdaptiveParallelTrainer(ParallelTrainer):
    """Trainer with dynamic complexity adjustment"""
    
    def __init__(self, config: Dict[str, Any], use_wandb: bool = False):
        # Mark dynamic in experiment name
        config['training']['dynamic_complexity'] = True  # ensure flag set
        # Create complexity manager
        self.complexity_manager = ComplexityManager(config)
        
        super().__init__(config, use_wandb)
        
        
        # Override initial environment to use manager's config
        self.vector_env = self._create_vectorized_env()
        
        # Extend metrics for complexity tracking
        self.metrics['complexity_history'] = []
        self.metrics['task_class_history'] = []
        self.metrics['performance_scores'] = []
        
        # Log initial status
        self.logger.info(f"Dynamic complexity enabled: {self.complexity_manager.get_status()}")
    
    def get_environment_config(self):
        """Get dynamic environment configuration from complexity manager"""
        return self.complexity_manager.get_environment_config()
    
    def _create_vectorized_env(self) -> VectorizedMazeEnv:
        """Create environment using complexity manager's current config"""
        env_config = self.get_environment_config()
        return VectorizedMazeEnv(
            num_envs=self.batch_size,
            env_config=env_config
        )
    
    def _recreate_vectorized_env(self):
        """Recreate environment after complexity adjustment"""
        if hasattr(self, 'vector_env'):
            self.vector_env.close()
        self.vector_env = self._create_vectorized_env()
    
    def _handle_complexity_adjustment(self, adjustment: Dict[str, Any], epoch: int):
        """React to complexity change"""
        # Log adjustment
        self.logger.info(f"Epoch {epoch}: {adjustment['action']} - "
                         f"{adjustment['old_stage']} -> {adjustment['new_stage']}, "
                         f"complexity {adjustment['old_complexity']:.2f} -> {adjustment['new_complexity']:.2f}")
        # Recreate environment if stage or complexity changed significantly
        if (adjustment['old_stage'] != adjustment['new_stage'] or
            abs(adjustment['old_complexity'] - adjustment['new_complexity']) > 0.01):
            self._recreate_vectorized_env()
    

    def _visualize_current_environments(self, epoch: int):
        """Visualize a sample of current training environments"""
        print(f"\n📸 Visualizing environments at epoch {epoch}")
        status = self.complexity_manager.get_status()
        print(f"  Stage: {status['current_stage']}, Complexity: {status['current_complexity']:.2f}")
        
        num_to_show = min(4, len(self.vector_env.envs))
        cell_size = 256
        padding = 10
        cols = 2
        rows = (num_to_show + cols - 1) // cols
        total_width = cols * cell_size + (cols + 1) * padding
        total_height = rows * cell_size + (rows + 1) * padding
        combined = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        
        for i in range(num_to_show):
            env = self.vector_env.envs[i]
            # Force render by temporarily setting render_size
            original_size = env.render_size
            env.render_size = cell_size
            if hasattr(env, '_render_buffer'):
                env._render_buffer = None
            frame = super(type(env), env).render()
            env.render_size = original_size
            if frame is None:
                frame = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                cv2.putText(frame, f"Env {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            if frame.shape[:2] != (cell_size, cell_size):
                frame = cv2.resize(frame, (cell_size, cell_size))
            col = i % cols
            row = i // cols
            x = padding + col * (cell_size + padding)
            y = padding + row * (cell_size + padding)
            combined[y:y+cell_size, x:x+cell_size] = frame
        
        title = f"Epoch {epoch}: {status['current_stage']} (Complexity: {status['current_complexity']:.2f})"
        cv2.putText(combined, title, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        info = f"Doors: {len(self.vector_env.envs[0].doors)}, Buttons: {len(self.vector_env.envs[0].buttons)}"
        cv2.putText(combined, info, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
        cv2.imshow('Training Visualization', combined)
        cv2.waitKey(0)
    
    def _stage_to_numeric(self, stage: str) -> float:
        """Convert stage name to numeric value for logging"""
        stage_map = {
            "basic": 0.0,
            "doors": 0.33,
            "buttons": 0.66,
            "complex": 1.0
        }
        return stage_map.get(stage, 0.0)
    
    def _save_model(self, name: str):
        save_dir = Path(self.config['experiment']['save_dir'])
        save_dir.mkdir(exist_ok=True)

        if name in ['best', 'final']:
            agent_path = save_dir / f"{self.experiment_name}_{name}.pt"
            extra_data = {
                'complexity_status': self.complexity_manager.get_status(),
                'training_metrics': {
                    'best_reward': self.metrics['best_reward'],
                    'current_epoch': len(self.metrics['train_rewards']),
                    'current_complexity': self.complexity_manager.get_current_complexity(),
                    'current_task_class': self.complexity_manager.get_current_task_class(),
                }
            }
            self.agent.save(str(agent_path), extra_data=extra_data)
            self.logger.info(f"Saved agent to {agent_path}")

        # Also save checkpoint (same as base trainer)
        checkpoint_path = save_dir / f"{self.experiment_name}_{name}_checkpoint.pt"
        checkpoint = {
            'epoch': len(self.metrics['train_rewards']),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.lr_scheduler.state_dict(),
            'metrics': self.metrics,
            'complexity_manager_state': {
                'current_stage_idx': self.complexity_manager.current_stage_idx,
                'current_complexity': self.complexity_manager.get_current_complexity(),
                'performance_history': list(self.complexity_manager.performance_history),
                'adjustments_made': self.complexity_manager.adjustments_made
            },
            'config': self.config
        }
        torch.save(checkpoint, str(checkpoint_path))
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_metrics(self):
        """Override to include complexity history"""
        metrics_dir = Path('logs/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = metrics_dir / f"{self.experiment_name}_metrics.npz"
        
        # Convert task class history to numeric for saving
        task_class_numeric = [self._stage_to_numeric(stage) for stage in self.metrics['task_class_history']]
        
        np.savez(str(metrics_path),
                train_rewards=self.metrics['train_rewards'],
                train_losses=self.metrics['train_losses'],
                test_rewards=self.metrics['test_rewards'],
                complexity_history=self.metrics['complexity_history'],
                task_class_history=task_class_numeric,
                performance_scores=self.metrics['performance_scores'],
                timing_collection=self.metrics['timing']['collection'],
                timing_training=self.metrics['timing']['training'])
        
        # Plot metrics (uses base class method)
        self._plot_metrics()
    
    def _print_training_summary(self, start_time: float):
        """Print detailed training summary with complexity info"""
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("DYNAMIC COMPLEXITY TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Final best reward: {self.metrics['best_reward']:.2f}")
        
        print(f"\nComplexity Progression:")
        print(f"  Final stage: {self.complexity_manager.get_current_task_class()}")
        print(f"  Final complexity: {self.complexity_manager.get_current_complexity():.2f}")
        print(f"  Total adjustments made: {self.complexity_manager.adjustments_made}")
        
        if self.complexity_manager.performance_history:
            print(f"  Final performance score: {self.complexity_manager.calculate_performance_score():.2f}")
        
        print(f"\nStage Progression:")
        stages = ["basic", "doors", "buttons", "complex"]
        for stage in stages:
            if stage in self.metrics['task_class_history']:
                first_epoch = self.metrics['task_class_history'].index(stage)
                last_epoch = len(self.metrics['task_class_history']) - 1 - self.metrics['task_class_history'][::-1].index(stage)
                duration = last_epoch - first_epoch + 1
                avg_complexity = np.mean([c for c, s in zip(self.metrics['complexity_history'][first_epoch:last_epoch+1], 
                                                          self.metrics['task_class_history'][first_epoch:last_epoch+1]) 
                                        if s == stage])
                print(f"  {stage.capitalize():8s}: epochs {first_epoch:4d}-{last_epoch:4d} "
                      f"({duration:4d} epochs), avg complexity: {avg_complexity:.2f}")
        
        print(f"\nModel saved as: {self.experiment_name}_best.pt")
        print(f"{'='*80}")
    
    def train(self):
        """Main training loop with dynamic complexity adjustments"""
        training_config = self.config['training']
        epochs = training_config['epochs']
        save_interval = training_config['save_interval']
        test_interval = training_config['test_interval']
        
        print("\n🎮 Dynamic Complexity Training Controls:")
        print("  Press 'v' to visualize current environments")
        print("  Press 'q' to stop training early")
        print("=" * 50)
        
        cv2.namedWindow('Training Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Training Controls', 400, 100)
        dummy = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(dummy, "Press 'v' to visualize", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(dummy, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.imshow('Training Controls', dummy)
        cv2.waitKey(1)
        
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        start_time = time.time()
        
        for epoch in pbar:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('v'):
                self._visualize_current_environments(epoch)
                cv2.imshow('Training Controls', dummy)
            elif key == ord('q'):
                print("\n⚠️ Early stop requested.")
                self._save_model('interrupted')
                cv2.destroyAllWindows()
                break
            
            epoch_start = time.time()
            coll_start = time.time()
            experiences = self._collect_experiences_parallel()
            coll_time = time.time() - coll_start
            
            train_start = time.time()
            train_metrics = self._train_step(experiences)
            train_time = time.time() - train_start
            
            epoch_reward = train_metrics['reward']
            
            # Update complexity manager
            self.complexity_manager.add_performance(epoch_reward, epoch)
            adjustment = self.complexity_manager.adjust_complexity(epoch)
            if adjustment:
                self._handle_complexity_adjustment(adjustment, epoch)

            if epoch % self.complexity_manager.performance_window == 0:
                self.metrics['performance_scores'].append(self.complexity_manager.calculate_performance_score())
            
            # Store metrics
            self.metrics['train_rewards'].append(epoch_reward)
            self.metrics['train_losses'].append(train_metrics['loss'])
            self.metrics['timing']['collection'].append(coll_time)
            self.metrics['timing']['training'].append(train_time)
            self.metrics['timing']['total'].append(time.time() - epoch_start)
            self.metrics['complexity_history'].append(self.complexity_manager.get_current_complexity())
            self.metrics['task_class_history'].append(self.complexity_manager.get_current_task_class())
            
            
            if epoch % test_interval == 0:# and epoch != 0:
                test_metrics = self._test_valid(epochs=4)
                test_reward = test_metrics['reward']
                self.metrics['test_rewards'].append(test_reward)
                if test_reward > self.metrics['best_reward']:
                    self.metrics['best_reward'] = test_reward
                    self._save_model('best')
                    self.logger.info(f"New best model with reward: {test_reward:.2f}")
            
            if epoch % save_interval == 0 and epoch != 0:
                self._save_model(f'epoch_{epoch:06d}')
            
            avg_coll = np.mean(self.metrics['timing']['collection'][-10:]) if self.metrics['timing']['collection'] else coll_time
            avg_train = np.mean(self.metrics['timing']['training'][-10:]) if self.metrics['timing']['training'] else train_time
            
            pbar.set_postfix({
                'reward': f"{epoch_reward:.2f}",
                'loss': f"{train_metrics['loss']:.4f}",
                'best': f"{self.metrics['best_reward']:.2f}",
                'stage': self.complexity_manager.get_current_task_class(),
                'comp': f"{self.complexity_manager.get_current_complexity():.2f}",
                'perf': f"{self.complexity_manager.calculate_performance_score():.2f}",
                'adj': self.complexity_manager.adjustments_made,
                'eps/s': f"{self.batch_size/(avg_coll+avg_train):.1f}",
            })
            
            if self.use_wandb:
                wandb.log({
                    'train/reward': epoch_reward,
                    'train/loss': train_metrics['loss'],
                    'lr': self.lr_scheduler.get_lr(),
                    'complexity/current': self.complexity_manager.get_current_complexity(),
                    'complexity/stage': self._stage_to_numeric(self.complexity_manager.get_current_task_class()),
                    'complexity/performance_score': self.complexity_manager.calculate_performance_score(),
                })
                if adjustment:
                    wandb.log({
                        'complexity/adjustment_action': adjustment.get('action', 'none'),
                        'complexity/old_complexity': adjustment.get('old_complexity', 0.0),
                        'complexity/new_complexity': adjustment.get('new_complexity', 0.0),
                    })
                if epoch % test_interval == 0 and epoch != 0:
                    wandb.log({'test/reward': test_metrics['reward']})
            
            self.lr_scheduler.step()

        test_metrics = self._test_valid(epochs=4)
        test_reward = test_metrics['reward']
        self.metrics['test_rewards'].append(test_reward)
        if test_reward > self.metrics['best_reward']:
            self.metrics['best_reward'] = test_reward
            self._save_model('best')
            self.logger.info(f"New best model with reward: {test_reward:.2f}")

        
        self._save_model('final')
        self._save_metrics()
        self._print_training_summary(start_time)
        if self.use_wandb:
            wandb.finish()


# ============================================================================
# FACTORY: Return appropriate trainer based on config
# ============================================================================

def Trainer(config: Dict[str, Any], use_wandb: bool = False):
    """Factory function returning either ParallelTrainer or AdaptiveParallelTrainer"""
    if config['training'].get('dynamic_complexity', False):
        return AdaptiveParallelTrainer(config, use_wandb)
    else:
        return ParallelTrainer(config, use_wandb)