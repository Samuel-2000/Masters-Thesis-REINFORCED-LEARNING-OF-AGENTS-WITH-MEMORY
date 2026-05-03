# src/training/trainer.py - OPTIMIZED PARALLEL VERSION WITH DYNAMIC COMPLEXITY
"""
Training module with optimizations, parallel execution, and dynamic complexity
"""

import torch
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
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
# STANDALONE PLOTTING FUNCTION (with safe array length and exact test epochs)
# ============================================================================
def generate_plots_from_metrics(metrics: Dict[str, Any], plots_dir: Path, increase_threshold: float, decrease_threshold: float):
    """Generate all training plots from a metrics dictionary."""
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    def save_plot(fig, name):
        path = plots_dir / f"{name}.png"
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ---- 1. Training Rewards (raw) + Test Rewards (exact epochs) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    rewards = np.array(metrics['train_rewards'])
    epochs = np.arange(len(rewards))
    ax.plot(epochs, rewards, 'b-', alpha=0.7, linewidth=1, label='Train Reward (raw)')
    if 'test_rewards' in metrics and len(metrics['test_rewards']) > 0:
        test_rewards = metrics['test_rewards']
        if 'test_epochs' in metrics and len(metrics['test_epochs']) == len(test_rewards):
            test_epochs = metrics['test_epochs']
        else:
            # fallback for old checkpoints
            test_interval = max(1, len(rewards) // len(test_rewards))
            test_epochs = np.arange(0, len(test_rewards) * test_interval, test_interval)
        ax.plot(test_epochs, test_rewards, 'g-o', linewidth=1.5, markersize=6, label='Test Reward')
    ax.set_title('Training & Test Rewards (raw)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, 'rewards')

    # ---- 2. Training Losses (total + policy) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics['train_losses'], 'r-', alpha=0.7, label='Total Loss')
    if 'policy_losses' in metrics and len(metrics['policy_losses']) > 0:
        ax.plot(metrics['policy_losses'], 'r--', alpha=0.5, label='Policy Loss')
    ax.set_title('Training Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'losses')

    # ---- 3. Auxiliary Losses ----
    if 'aux_losses' in metrics and len(metrics['aux_losses']) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(metrics['aux_losses'], label='Total Aux Loss', color='purple')
        if 'energy_losses' in metrics and len(metrics['energy_losses']) > 0:
            ax.plot(metrics['energy_losses'], label='Energy MSE', color='orange')
        if 'obs_losses' in metrics and len(metrics['obs_losses']) > 0:
            ax.plot(metrics['obs_losses'], label='Obs MSE', color='green')
        ax.set_title('Auxiliary Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_plot(fig, 'aux_losses')

    # ---- 4. Complexity & Task Class Progression ----
    if 'complexity_history' in metrics and len(metrics['complexity_history']) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(metrics['complexity_history'], 'b-', linewidth=1, label='Complexity')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Complexity Level', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        if 'task_class_history' in metrics and len(metrics['task_class_history']) > 0:
            ax2 = ax.twinx()
            stage_map = {'basic': 0.0, 'doors': 0.33, 'buttons': 0.66, 'complex': 1.0}
            task_numeric = []
            for t in metrics['task_class_history']:
                if isinstance(t, str):
                    task_numeric.append(stage_map.get(t, 0.0))
                else:
                    task_numeric.append(float(t))
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
        ax.set_title('Complexity Progression (raw)')
        save_plot(fig, 'complexity')

    # ---- 5. Performance Scores (colored by config change) ----
    if 'performance_scores' in metrics and len(metrics['performance_scores']) > 0 and 'complexity_history' in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        scores = np.array(metrics['performance_scores'])
        window = max(1, len(metrics['train_rewards']) // len(scores))
        perf_epochs = np.arange(len(scores)) * window
        complexities = np.array(metrics['complexity_history'])
        # Ensure indices are within bounds
        valid_mask = perf_epochs < len(complexities)
        perf_epochs = perf_epochs[valid_mask]
        scores = scores[:len(perf_epochs)]
        if len(scores) > 0:
            tasks = metrics.get('task_class_history', ['basic'] * len(complexities))
            if isinstance(tasks, list):
                tasks = np.array(tasks)
            complexities_at_perf = complexities[perf_epochs]
            tasks_at_perf = tasks[perf_epochs]
            change_indices = []
            for i in range(1, len(complexities_at_perf)):
                if abs(complexities_at_perf[i] - complexities_at_perf[i-1]) > 1e-6 or tasks_at_perf[i] != tasks_at_perf[i-1]:
                    change_indices.append(i)
            colors = ['orange', 'blue']
            start = 0
            for idx, split in enumerate(change_indices):
                seg_x = perf_epochs[start:split]
                seg_y = scores[start:split]
                if len(seg_x) > 0:
                    ax.plot(seg_x, seg_y, color=colors[idx % 2], linewidth=1.5)
                    ax.axvline(x=perf_epochs[split], color='gray', linestyle=':', alpha=0.5)
                start = split
            if start < len(perf_epochs):
                ax.plot(perf_epochs[start:], scores[start:], color=colors[len(change_indices) % 2], linewidth=1.5)
            # Threshold lines
            ax.axhline(increase_threshold, color='green', linestyle='--', alpha=0.7, label=f'Increase ({increase_threshold})')
            ax.axhline(decrease_threshold, color='red', linestyle='--', alpha=0.7, label=f'Decrease ({decrease_threshold})')
            ax.legend()
            ax.set_title('Performance Scores (colored by config change)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            save_plot(fig, 'performance_scores')

    # ---- 6. Complexity vs Reward (raw) ----
    if 'complexity_history' in metrics and len(metrics['train_rewards']) > 10:
        fig, ax = plt.subplots(figsize=(8, 5))
        complexities = np.array(metrics['complexity_history'])
        rewards_raw = np.array(metrics['train_rewards'])
        # Ensure same length
        min_len = min(len(complexities), len(rewards_raw))
        complexities = complexities[:min_len]
        rewards_raw = rewards_raw[:min_len]
        sc = ax.scatter(complexities, rewards_raw, c=range(min_len), cmap='viridis', alpha=0.7, s=10)
        plt.colorbar(sc, ax=ax, label='Epoch')
        if min_len > 1:
            corr = np.corrcoef(complexities, rewards_raw)[0, 1]
            ax.set_title(f'Complexity vs Reward (raw, corr: {corr:.3f})')
        else:
            ax.set_title('Complexity vs Reward (raw)')
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        save_plot(fig, 'complexity_vs_reward')

    # ---- 7. Reward vs Complexity per stage ----
    if 'task_class_history' in metrics and len(metrics['task_class_history']) > 0:
        stage_order = ['basic', 'doors', 'buttons', 'complex']
        unique_stages = [s for s in stage_order if s in metrics['task_class_history']]
        rewards_raw = np.array(metrics['train_rewards'])
        complexities_raw = np.array(metrics['complexity_history'])
        stages_raw = np.array(metrics['task_class_history'])
        min_len = min(len(rewards_raw), len(complexities_raw), len(stages_raw))
        rewards_raw = rewards_raw[:min_len]
        complexities_raw = complexities_raw[:min_len]
        stages_raw = stages_raw[:min_len]
        for stage in unique_stages:
            fig, ax = plt.subplots(figsize=(8, 5))
            mask = (stages_raw == stage)
            ax.scatter(complexities_raw, rewards_raw, c='gray', alpha=0.2, s=10, label='All epochs')
            epochs_of_stage = np.arange(min_len)[mask]
            if len(epochs_of_stage) > 0:
                sc = ax.scatter(complexities_raw[mask], rewards_raw[mask], c=epochs_of_stage, cmap='viridis', alpha=0.8, s=30, label=f'{stage.capitalize()} active')
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label('Epoch')
            if np.sum(mask) > 1:
                x_vals = complexities_raw[mask]
                y_vals = rewards_raw[mask]
                valid = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isinf(x_vals) | np.isinf(y_vals))
                x_vals = x_vals[valid]
                y_vals = y_vals[valid]
                if len(x_vals) >= 2 and np.std(x_vals) > 1e-6:
                    try:
                        z = np.polyfit(x_vals, y_vals, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend: {z[0]:.2f}*x + {z[1]:.2f}')
                    except np.linalg.LinAlgError:
                        pass
            ax.set_title(f'Reward vs Complexity – {stage.capitalize()} stage (raw data)')
            ax.set_xlabel('Complexity Level')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
            ax.legend()
            save_plot(fig, f'reward_vs_complexity_stage_{stage}')


# ============================================================================
# DYNAMIC COMPLEXITY MANAGER (unchanged)
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
        self.epochs_without_progress = 0
        self.last_complexity_increase_epoch = 0
        self.last_max_reward = -float('inf')
        
        # Stage switching tracking
        self.stage_selection_counts = {stage: 0 for stage in self.curriculum_stages}
        self.total_switches = 0
        self.linear_cycle_complete = False
        
        # Statistics
        self.adjustments_made = 0
        self.stage_switches = 0

    def add_performance(self, reward: float):
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
        if len(self.performance_history) < self.performance_window // 2:
            return False
        
        return True

    def should_switch_stage(self, epoch: int) -> bool:
        """Check if we should switch to a different task due to stagnation"""
        if not self.enabled:
            return False
        
        current_stage = self.get_current_task_class()
        current_complexity = self.stage_complexities[current_stage]
        
        # Prevent switching from basic until it reaches minimum complexity
        if current_stage == 'basic' and current_complexity < self.min_basic_complexity:
            return False
        
        # Complexity stagnation detection
        if epoch - self.last_complexity_increase_epoch >= self.stagnation_switch_interval:
            return True
        
        # Optional: Check for oscillation (low performance, low variation)
        if len(self.performance_history) >= self.performance_window:
            recent_std = np.std(list(self.performance_history)[-self.performance_window:])
            recent_mean = np.mean(list(self.performance_history)[-self.performance_window:])
            if recent_std < 0.1 and recent_mean < self.decrease_threshold:
                return True
        
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
        
        if max_reward < 0.1:
            return 0.0
        
        score = max(0.0, min(1.0, avg_performance / max_reward))
        self._last_performance_score = score
        return score

    def switch_to_next_stage(self, epoch: int) -> Dict[str, Any]:
        """
        Switch stage:
        - First cycle: strictly linear progression through curriculum.
        - After cycle: probabilistic based on low complexity and low selection frequency.
        """
        old_stage_idx = self.current_stage_idx
        old_stage = self.curriculum_stages[old_stage_idx]
        old_complexity = self.stage_complexities[old_stage]

        # ---------- FIRST CYCLE: LINEAR PROGRESSION ----------
        if not self.linear_cycle_complete:
            # Move to next stage in order (cyclic)
            next_idx = (old_stage_idx + 1) % len(self.curriculum_stages)
            self.current_stage_idx = next_idx
            new_stage = self.curriculum_stages[next_idx]
            new_complexity = self.stage_complexities[new_stage]

            # If we just completed one full cycle (wrapped back to start)
            if next_idx == 0:
                self.linear_cycle_complete = True

            reason = "Linear progression (first cycle)"
            probs = None

        # ---------- AFTER CYCLE: PROBABILISTIC ----------
        else:
            epsilon = 0.1   # prevents zero weights
            weights = []
            for stage in self.curriculum_stages:
                complexity = self.stage_complexities[stage]
                freq = self.stage_selection_counts[stage] / (self.total_switches + 1)
                weight = (1.0 - complexity + epsilon) * (1.0 - freq + epsilon)
                weights.append(weight)
            probs = np.array(weights) / np.sum(weights)
            next_idx = np.random.choice(len(self.curriculum_stages), p=probs)
            self.current_stage_idx = next_idx
            new_stage = self.curriculum_stages[next_idx]
            new_complexity = self.stage_complexities[new_stage]
            reason = f"Probabilistic (complexity={new_complexity:.2f}, freq={self.stage_selection_counts[new_stage]/(self.total_switches+1):.2f})"

        # --- Update selection counts and reset stagnation ---
        self.stage_selection_counts[new_stage] += 1
        self.total_switches += 1

        self.epochs_without_progress = 0
        self.last_complexity_increase_epoch = epoch
        self.last_max_reward = -float('inf')
        self.performance_history.clear()
        self.stage_switches += 1

        adjustment_info = {
            "action": "switched_stage",
            "old_stage": old_stage,
            "new_stage": new_stage,
            "old_complexity": old_complexity,
            "new_complexity": new_complexity,
            "reason": reason,
            "stage_probs": probs.tolist() if probs is not None else None
        }
        return adjustment_info

    def adjust_complexity(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Adjust complexity based on performance, with task switching for stagnation"""
        if not self.should_adjust(epoch):
            return None
        
        # First check if we should switch tasks due to stagnation
        if self.should_switch_stage(epoch):
            adjustment_info = self.switch_to_next_stage(epoch)
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
            self.last_complexity_increase_epoch = epoch

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
        self.performance_history.clear()
        
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
                env_config['n_doors'] = 0
                env_config['n_buttons_per_door'] = 0
                env_config['button_break_probability'] = 0.0
                
            elif current_stage == 'doors':
                env_config['n_doors'] = None
                env_config['n_buttons_per_door'] = 0
                env_config['button_break_probability'] = 0.0
                
            elif current_stage == 'buttons':
                env_config['n_doors'] = None
                env_config['n_buttons_per_door'] = None
                env_config['button_break_probability'] = None
                
            elif current_stage == 'complex':
                env_config['n_doors'] = None
                env_config['n_buttons_per_door'] = None
                env_config['button_break_probability'] = None
        
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
            "performance_score": self.calculate_performance_score(),
            "max_rewards_by_stage": self.max_rewards_by_stage,
            "basic_min_complexity_reached": self.stage_complexities['basic'] >= self.min_basic_complexity,
            "linear_cycle_complete": self.linear_cycle_complete,
            "stage_selection_counts": self.stage_selection_counts,
            "total_switches": self.total_switches
        }


# ============================================================================
# BASE TRAINER (NO DYNAMIC COMPLEXITY)
# ============================================================================

class ParallelTrainer:
    """Base trainer with parallel execution, fixed environment (no dynamic complexity)"""
    
    def __init__(self, config: Dict[str, Any]):
        
        self.config = config
        self.base_seed = config['experiment']['seed']
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed
        seed_everything(config['experiment']['seed'])
        
        # Get batch size for parallel environments
        training_config = self.config['training']
        self.batch_size = training_config['batch_size']
        
        # ---------- HANDLE RESUME BEFORE BUILDING EXPERIMENT NAME AND DIRECTORIES ----------
        resume_path = config['experiment'].get('resume')
        if resume_path and Path(resume_path).exists():
            # Load checkpoint temporarily to extract experiment name and config
            temp_checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            if 'config' in temp_checkpoint:
                orig_config = temp_checkpoint['config']
                # Override current config with checkpoint's settings
                config['environment'] = orig_config.get('environment', config['environment'])
                config['model'] = orig_config.get('model', config['model'])
                if 'training' in orig_config and 'dynamic_complexity' in orig_config['training']:
                    config['training']['dynamic_complexity'] = orig_config['training']['dynamic_complexity']
                # Use original base name and date subfolder
                if 'experiment' in orig_config:
                    self.base_name = orig_config['experiment'].get('base_name', None)
                    self.date_subfolder = orig_config['experiment'].get('date_subfolder', None)
                # If not found in config, fallback to parsing the path
                if not self.base_name or not self.date_subfolder:
                    parent = Path(resume_path).parent
                    self.date_subfolder = parent.name
                    self.base_name = parent.parent.name
            else:
                parent = Path(resume_path).parent
                self.date_subfolder = parent.name
                self.base_name = parent.parent.name
        else:
            # Build new base name (without timestamp) and date subfolder
            self.base_name = config['experiment']['name'] or f"{config['model']['type']}_" \
                              f"{config['training']['batch_size']}b_" \
                              f"{config['training']['learning_rate']}lr"
            if config['training']['auxiliary_tasks']:
                self.base_name += "_aux"
            self.date_subfolder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Full experiment name for logging (for backward compatibility)
        self.experiment_name = f"{self.base_name}_{self.date_subfolder}"
        
        # Now set up logging
        self.logger = setup_logging(self.experiment_name)
        
        # Build structured directories: models/{network_type}/{aux_or_no_aux}/{base_name}/{date_subfolder}/
        network_type = config['model']['type']
        use_aux = config['model']['use_auxiliary']
        aux_str = 'aux' if use_aux else 'no_aux'
        
        self.experiment_dir = Path(config['experiment']['save_dir']) / network_type / aux_str / self.base_name / self.date_subfolder
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path('results/plots') / network_type / aux_str / self.base_name / self.date_subfolder
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_dir = Path('logs/metrics') / network_type / aux_str / self.base_name
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.metrics_dir / "metrics.npz"
        
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

        self.consecutive_episodes = config['training']['consecutive_episodes']
        self.grid_change_prob = config['training']['grid_change_prob']
        self.update_per_episode = config['training']['update_per_episode']
        self._grid_pool = []   # stores grid configurations (tuples)
        
        # Metrics
        self.metrics = {
            'train_rewards': [],
            'train_losses': [],
            'aux_losses': [],
            'energy_losses': [],
            'obs_losses': [],
            'test_epochs': [],          # store epoch at which test was performed
            'test_rewards': [],
            'best_reward': -np.inf,
            'timing': {
                'collection': [],
                'training': [],
                'total': []
            }
        }

        # Load checkpoint if provided (to restore metrics, optimizer, etc.)
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(resume_path)
        
        # Create vectorized environment (after config may have been restored from checkpoint)
        self.vector_env = self._create_vectorized_env()

    def _create_vectorized_env(self) -> VectorizedMazeEnv:
        """Create vectorized training environment from config (fixed)"""
        env_config = self.config['environment'].copy()
        # Ensure n_doors is not None (should be set by config)
        if env_config.get('n_doors') is None and env_config.get('task_class') != 'basic':
            env_config['n_doors'] = -1  # trigger auto-configuration
        return VectorizedMazeEnv(
            num_envs=self.batch_size,
            env_config=env_config,
            base_seed=self.base_seed
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


    def _generate_grid_config(self) -> tuple:
        """Generate a new random grid configuration based on current environment settings."""
        env_config = self.get_environment_config()
        # Use a new random seed (different from any previous)
        seed = random.randint(0, 2**31 - 1)
        return (seed,
                env_config['task_class'],
                env_config['complexity_level'],
                env_config.get('n_doors'),
                env_config.get('n_buttons_per_door'),
                env_config.get('button_break_probability'))

    def _apply_grid_config(self, config: tuple, reset_hidden: bool = False):
        """Recreate vectorized environment with given grid config. Optionally reset agent hidden state."""
        seed, task_class, complexity, n_doors, n_buttons, break_prob = config
        env_config = self.get_environment_config()
        env_config.update({
            'task_class': task_class,
            'complexity_level': complexity,
            'n_doors': n_doors,
            'n_buttons_per_door': n_buttons,
            'button_break_probability': break_prob
        })
        if hasattr(self, 'vector_env'):
            self.vector_env.close()
        self.vector_env = VectorizedMazeEnv(
            num_envs=self.batch_size,
            env_config=env_config,
            base_seed=seed
        )
        if reset_hidden:
            self.agent.reset()


    def _collect_experiences_parallel(self, full_reset: bool = True) -> Dict[str, torch.Tensor]:
        """
        Collect experiences in parallel across all environments
        
        Args:
            full_reset: If True, do full environment reset (new random grid) and reset agent hidden state.
                    If False, do soft reset (keep grid layout, only reset agent position/food) and 
                    keep agent hidden state (allows memory across episodes on same grid).
        """
        max_steps = self.vector_env.envs[0].max_steps
        
        # Decide reset type
        if full_reset:
            self.agent.reset()  # Reset network state for new grid
            # Reset all environments (new random grids)
            obs_array, _ = self.vector_env.reset()
        else:
            # Soft reset: keep grid layout, only reset agent position and food
            # Do NOT reset agent hidden state
            obs_array, _ = self.vector_env.soft_reset_all()
        
        # Convert to tensor
        observations = torch.tensor(obs_array, dtype=torch.long).to(self.device)
        observations = observations.unsqueeze(1)  # [B, 1, K]
        
        # Storage
        all_observations, all_actions, all_rewards, all_energies, all_next_obs = [], [], [], [], []
        
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
        
    def _merge_experiences(self, exp1, exp2):
        """Concatenate two episode batches along time dimension."""
        merged = {}
        for key in exp1:
            merged[key] = torch.cat([exp1[key], exp2[key]], dim=1)
        return merged

    def _post_epoch_hook(self, epoch: int):
        """Hook called after each epoch; can be overridden by subclasses."""
        pass

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
        """Run `epochs` test epochs, each processing `batch_size` parallel epochs.
        Always tests on complex stage at maximum complexity (1.0) using fixed seeds.
        """
        self.agent.network.eval()

        # Build test environment configuration: always complex, complexity=1.0
        test_env_config = self.config['environment'].copy()
        test_env_config['task_class'] = 'complex'
        test_env_config['complexity_level'] = 1.0
        # Set door/button parameters for complex stage
        test_env_config['n_doors'] = -1
        test_env_config['n_buttons_per_door'] = -1
        test_env_config['button_break_probability'] = -1.0

        test_env_config['render_size'] = 0

        # Use a fixed seed offset to guarantee identical environments each test call
        test_seed = self.base_seed + 12345  # any constant offset works

        total_epochs = epochs * self.batch_size
        all_rewards = []
        all_lengths = []
        max_steps = 0

        for _ in range(epochs):
            test_env = VectorizedMazeEnv(
                num_envs=self.batch_size,
                env_config=test_env_config,
                base_seed=test_seed   # same seed every time
            )
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
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training state from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.metrics = checkpoint['metrics']
        
        # Migrate old checkpoints (without test_epochs)
        if 'test_epochs' not in self.metrics:
            test_interval = self.config['training'].get('test_interval', 100)
            num_tests = len(self.metrics['test_rewards'])
            self.metrics['test_epochs'] = []
            for i in range(1, num_tests + 1):
                epoch = i * test_interval
                if epoch < len(self.metrics['train_rewards']):
                    self.metrics['test_epochs'].append(epoch)
                else:
                    break
            if len(self.metrics['test_epochs']) != len(self.metrics['test_rewards']):
                self.metrics['test_epochs'] = []
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
        if 'model_state_dict' in checkpoint:
            self.agent.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded model weights from checkpoint {checkpoint_path}")
        
        if hasattr(self, 'complexity_manager') and 'complexity_manager_state' in checkpoint:
            cm_state = checkpoint['complexity_manager_state']
            self.complexity_manager.current_stage_idx = cm_state['current_stage_idx']
            self.complexity_manager.performance_history = deque(
                cm_state['performance_history'],
                maxlen=self.complexity_manager.performance_window
            )
            self.complexity_manager.adjustments_made = cm_state['adjustments_made']
            if 'stage_complexities' in cm_state:
                self.complexity_manager.stage_complexities = cm_state['stage_complexities']
            if 'max_rewards_by_stage' in cm_state:
                self.complexity_manager.max_rewards_by_stage = cm_state['max_rewards_by_stage']
            if 'epochs_without_progress' in cm_state:
                self.complexity_manager.epochs_without_progress = cm_state['epochs_without_progress']
            if 'last_complexity_increase_epoch' in cm_state:
                self.complexity_manager.last_complexity_increase_epoch = cm_state['last_complexity_increase_epoch']
            if 'last_max_reward' in cm_state:
                self.complexity_manager.last_max_reward = cm_state['last_max_reward']
        
        start_epoch = len(self.metrics['train_rewards'])
        self.logger.info(f"Resumed training from checkpoint at epoch {start_epoch}")

    def _save_model(self, name: str):
        # Save agent model (separate file)
        if name in ['best', 'final']:
            agent_path = self.experiment_dir / f"{name}.pt"
            self.agent.save(str(agent_path))
            self.logger.info(f"Saved agent to {agent_path}")

        # Save checkpoint (with model weights and config for resume/testing)
        checkpoint_path = self.experiment_dir / f"{name}_checkpoint.pt"
        checkpoint = {
            'epoch': len(self.metrics['train_rewards']),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.lr_scheduler.state_dict(),
            'metrics': self.metrics,
            'model_state_dict': self.agent.network.state_dict(),
            'model_config': {
                'network_type': self.agent.network_type,
                'use_auxiliary': self.agent.use_auxiliary,
                'hidden_size': self.agent.network.hidden_size,
                'observation_size': OBSERVATION_SIZE,
                'action_size': ACTION_SIZE
            },
            'config': self.config.copy()
        }
        # Store base_name and date_subfolder for resume
        checkpoint['config']['experiment']['base_name'] = self.base_name
        checkpoint['config']['experiment']['date_subfolder'] = self.date_subfolder
        
        if hasattr(self, 'complexity_manager'):
            checkpoint['complexity_manager_state'] = {
                'current_stage_idx': self.complexity_manager.current_stage_idx,
                'performance_history': list(self.complexity_manager.performance_history),
                'adjustments_made': self.complexity_manager.adjustments_made,
                'stage_complexities': self.complexity_manager.stage_complexities,
                'max_rewards_by_stage': self.complexity_manager.max_rewards_by_stage,
                'epochs_without_progress': self.complexity_manager.epochs_without_progress,
                'last_complexity_increase_epoch': self.complexity_manager.last_complexity_increase_epoch,
                'last_max_reward': self.complexity_manager.last_max_reward,
            }
        torch.save(checkpoint, str(checkpoint_path))
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _save_metrics(self):
        """Save training metrics"""
        metrics_dir = Path('logs/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save thresholds for later plotting
        increase_threshold = self.config['training'].get('complexity_increase_threshold', 0.65)
        decrease_threshold = self.config['training'].get('complexity_decrease_threshold', 0.4)
        
        np.savez(str(self.metrics_path),
                train_rewards=self.metrics['train_rewards'],
                train_losses=self.metrics['train_losses'],
                test_epochs=self.metrics['test_epochs'],
                test_rewards=self.metrics['test_rewards'],
                timing_collection=self.metrics['timing']['collection'],
                timing_training=self.metrics['timing']['training'],
                increase_threshold=increase_threshold,
                decrease_threshold=decrease_threshold)
        
        # Plot metrics
        self._plot_metrics()

    def _plot_metrics(self):
        increase_threshold = self.config['training'].get('complexity_increase_threshold', 0.65)
        decrease_threshold = self.config['training'].get('complexity_decrease_threshold', 0.4)
        generate_plots_from_metrics(self.metrics, self.plots_dir, increase_threshold, decrease_threshold)

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
        print(f"Model saved as: {self.base_name}_best.pt")
        print(f"{'='*80}")
        
    def _visualize_current_environments(self, epoch: int):
        """Visualize a sample of current training environments"""
        print(f"\n📸 Visualizing environments at epoch {epoch}")
        
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
        
        info = f"Doors: {len(self.vector_env.envs[0].doors)}, Buttons: {len(self.vector_env.envs[0].buttons)}"
        cv2.putText(combined, info, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
        cv2.imshow('Training Visualization', combined)
        cv2.waitKey(0)
    


    def train(self):
        training_config = self.config['training']
        epochs = training_config['epochs']
        save_interval = training_config['save_interval']
        test_interval = training_config['test_interval']

        start_epoch = len(self.metrics.get('epoch_rewards', []))
        episode_counter = len(self.metrics['train_rewards'])

        # Initial test
        if len(self.metrics['test_epochs']) == 0:
            test_metrics = self._test_valid(epochs=4)
            self.metrics['test_epochs'].append(0)
            self.metrics['test_rewards'].append(test_metrics['reward'])
            if test_metrics['reward'] > self.metrics['best_reward']:
                self.metrics['best_reward'] = test_metrics['reward']
                self._save_model('best')

        # ---------- Visualisation setup ----------
        print("\n🎮 Visualisation Controls:")
        print("  Press 'v' to visualise current environments")
        print("  Press 'q' to stop training early")
        print("=" * 50)
        cv2.namedWindow('Training Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Training Controls', 400, 100)
        dummy = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(dummy, "Press 'v' to visualise", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(dummy, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.imshow('Training Controls', dummy)
        cv2.waitKey(1)   # non‑blocking

        pbar = tqdm(range(start_epoch, epochs), desc="Epochs", unit="epoch")
        start_time = time.time()

        try:
            for epoch in pbar:
                # ---- Start of epoch: generate first grid, reset everything ----
                current_config = self._generate_grid_config()
                self._grid_pool = [current_config]
                self._apply_grid_config(current_config, reset_hidden=True)

                batched_experiences = None
                epoch_rewards = []

                # ---- Consecutive episodes loop ----
                for ep_idx in range(self.consecutive_episodes):
                    if ep_idx == 0:
                        full_reset = True               # first episode always full reset
                    else:
                        if np.random.random() < self.grid_change_prob:
                            new_config = self._generate_grid_config()
                            self._grid_pool.append(new_config)
                            current_config = new_config
                            full_reset = True
                        else:
                            chosen_config = random.choice(self._grid_pool)
                            if chosen_config == current_config:
                                full_reset = False
                            else:
                                current_config = chosen_config
                                full_reset = True
                        if full_reset:
                            self._apply_grid_config(current_config, reset_hidden=False)

                    # Collect one episode
                    experiences = self._collect_experiences_parallel(full_reset=full_reset)

                    if self.update_per_episode:
                        train_metrics = self._train_step(experiences)
                        episode_reward = train_metrics['reward']
                        self.metrics['train_rewards'].append(episode_reward)
                        self.metrics['train_losses'].append(train_metrics['loss'])
                        # store auxiliary losses if needed
                        episode_counter += 1
                        epoch_rewards.append(episode_reward)
                        self.lr_scheduler.step()
                    else:
                        if batched_experiences is None:
                            batched_experiences = experiences
                        else:
                            batched_experiences = self._merge_experiences(batched_experiences, experiences)

                # If not updating per episode, do one update for the whole epoch
                if not self.update_per_episode:
                    train_metrics = self._train_step(batched_experiences)
                    episode_reward = train_metrics['reward']
                    self.metrics['train_rewards'].append(episode_reward)
                    self.metrics['train_losses'].append(train_metrics['loss'])
                    episode_counter += 1
                    epoch_rewards = [episode_reward]
                    self.lr_scheduler.step()

                avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0
                self.metrics.setdefault('epoch_rewards', []).append(avg_epoch_reward)

                # ---------- Key handling ----------
                key = cv2.waitKey(1) & 0xFF
                if key == ord('v'):
                    self._visualize_current_environments(epoch)
                    # Re‑show control window after visualisation (if it was closed)
                    cv2.imshow('Training Controls', dummy)
                elif key == ord('q'):
                    print("\n⚠️ Early stop requested.")
                    self._save_model('interrupted')
                    cv2.destroyAllWindows()
                    break

                # Hook for adaptive trainer (complexity adjustment, stagnation check)
                self._post_epoch_hook(epoch)

                # Testing and saving
                if epoch % test_interval == 0 and epoch > 0:
                    test_metrics = self._test_valid(epochs=4)
                    self.metrics['test_epochs'].append(epoch)
                    self.metrics['test_rewards'].append(test_metrics['reward'])
                    if test_metrics['reward'] > self.metrics['best_reward']:
                        self.metrics['best_reward'] = test_metrics['reward']
                        self._save_model('best')

                if epoch % save_interval == 0 and epoch > 0:
                    self._save_model(f'epoch_{epoch:06d}')

                pbar.set_postfix({
                    'reward': f"{avg_epoch_reward:.2f}",
                    'best': f"{self.metrics['best_reward']:.2f}",
                    'episodes': episode_counter,
                    'pool': len(self._grid_pool),
                })

        except StopIteration:
            # This is raised by AdaptiveParallelTrainer._post_epoch_hook on stagnation
            print("\n🛑 Training stopped due to stagnation (no progress for many epochs).")
            cv2.destroyAllWindows()
            self._save_metrics()
            self._print_training_summary(start_time)
            return

        cv2.destroyAllWindows()

        # Final test and save
        if epochs > 0 and epochs-1 not in self.metrics['test_epochs']:
            test_metrics = self._test_valid(epochs=4)
            self.metrics['test_epochs'].append(epochs-1)
            self.metrics['test_rewards'].append(test_metrics['reward'])
            if test_metrics['reward'] > self.metrics['best_reward']:
                self.metrics['best_reward'] = test_metrics['reward']
                self._save_model('best')

        self._save_model('final')
        self._save_metrics()
        self._print_training_summary(start_time)


# ============================================================================
# ADAPTIVE TRAINER (inherits from ParallelTrainer)
# ============================================================================

class AdaptiveParallelTrainer(ParallelTrainer):
    """Trainer with dynamic complexity adjustment"""
    
    def __init__(self, config: Dict[str, Any]):
        config['training']['dynamic_complexity'] = True
        self.complexity_manager = ComplexityManager(config)
        super().__init__(config)
        
        # If we resumed a checkpoint that already contains these metrics, do not reset them.
        if 'complexity_history' not in self.metrics:
            self.metrics['complexity_history'] = []
            self.metrics['task_class_history'] = []
            self.metrics['performance_scores'] = []
        
        # Recreate environment with complexity manager's config (overrides the base environment)
        if hasattr(self, 'vector_env'):
            self.vector_env.close()
        self.vector_env = self._create_vectorized_env()
        
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
            env_config=env_config,
            base_seed=self.base_seed
        )
    
    def _recreate_vectorized_env(self):
        """Recreate environment after complexity adjustment"""
        if hasattr(self, 'vector_env'):
            self.vector_env.close()
        self.vector_env = self._create_vectorized_env()
    
    def _post_epoch_hook(self, epoch: int):
        if not self.complexity_manager.enabled:
            return

        # Get average performance for this epoch
        if 'epoch_rewards' in self.metrics and self.metrics['epoch_rewards']:
            avg_reward = self.metrics['epoch_rewards'][-1]
            self.complexity_manager.add_performance(avg_reward)

        adjustment = self.complexity_manager.adjust_complexity(epoch)
        if adjustment:
            self.logger.info(f"Epoch {epoch}: {adjustment['action']} - "
                            f"{adjustment['old_stage']} -> {adjustment['new_stage']}, "
                            f"complexity {adjustment['old_complexity']:.2f} -> {adjustment['new_complexity']:.2f}")
            self._recreate_vectorized_env()
            self._grid_pool = []

        # Record complexity history
        self.metrics['complexity_history'].append(self.complexity_manager.get_current_complexity())
        self.metrics['task_class_history'].append(self.complexity_manager.get_current_task_class())
        self.metrics['performance_scores'].append(self.complexity_manager.calculate_performance_score())

        # Stagnation termination
        if self.complexity_manager.epochs_without_progress >= self.complexity_manager.stagnation_termination:
            self.logger.info(f"Stagnation termination triggered at epoch {epoch}")
            self._save_model('stagnation_stop')
            raise StopIteration("Stagnation termination")


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
        # Save agent model with extra data
        if name in ['best', 'final']:
            agent_path = self.experiment_dir / f"{name}.pt"
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
        checkpoint_path = self.experiment_dir / f"{name}_checkpoint.pt"
        checkpoint = {
            'epoch': len(self.metrics['train_rewards']),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.lr_scheduler.state_dict(),
            'metrics': self.metrics,
            'model_state_dict': self.agent.network.state_dict(),
            'model_config': {
                'network_type': self.agent.network_type,
                'use_auxiliary': self.agent.use_auxiliary,
                'hidden_size': self.agent.network.hidden_size,
                'observation_size': OBSERVATION_SIZE,
                'action_size': ACTION_SIZE
            },
            'config': self.config.copy(),
            'complexity_manager_state': {
                'current_stage_idx': self.complexity_manager.current_stage_idx,
                'performance_history': list(self.complexity_manager.performance_history),
                'adjustments_made': self.complexity_manager.adjustments_made,
                'stage_complexities': self.complexity_manager.stage_complexities,
                'max_rewards_by_stage': self.complexity_manager.max_rewards_by_stage,
                'epochs_without_progress': self.complexity_manager.epochs_without_progress,
                'last_complexity_increase_epoch': self.complexity_manager.last_complexity_increase_epoch,
                'last_max_reward': self.complexity_manager.last_max_reward,
            }
        }
        checkpoint['config']['experiment']['base_name'] = self.base_name
        checkpoint['config']['experiment']['date_subfolder'] = self.date_subfolder
        torch.save(checkpoint, str(checkpoint_path))
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_metrics(self):
        """Override to include complexity history and thresholds"""
        metrics_dir = Path('logs/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert task class history to numeric for saving
        task_class_numeric = [self._stage_to_numeric(stage) for stage in self.metrics['task_class_history']]
        
        # Get thresholds
        increase_threshold = self.complexity_manager.increase_threshold
        decrease_threshold = self.complexity_manager.decrease_threshold
        
        np.savez(str(self.metrics_path),
                train_rewards=self.metrics['train_rewards'],
                train_losses=self.metrics['train_losses'],
                test_epochs=self.metrics['test_epochs'],
                test_rewards=self.metrics['test_rewards'],
                complexity_history=self.metrics['complexity_history'],
                task_class_history=task_class_numeric,
                performance_scores=self.metrics['performance_scores'],
                timing_collection=self.metrics['timing']['collection'],
                timing_training=self.metrics['timing']['training'],
                increase_threshold=increase_threshold,
                decrease_threshold=decrease_threshold)
        
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
        
        print(f"\nModel saved as: {self.base_name}_best.pt")
        print(f"{'='*80}")
    
    """
    def train(self):
        #Main training loop with dynamic complexity adjustments
        training_config = self.config['training']
        epochs = training_config['epochs']
        save_interval = training_config['save_interval']
        test_interval = training_config['test_interval']
        
        start_epoch = len(self.metrics['train_rewards'])
        
        # Perform initial test at epoch 0 if never tested before (fresh run)
        if len(self.metrics['test_epochs']) == 0 and len(self.metrics['test_rewards']) == 0:
            self.logger.info("Running initial test at epoch 0...")
            test_metrics = self._test_valid(epochs=4)
            self.metrics['test_epochs'].append(0)
            self.metrics['test_rewards'].append(test_metrics['reward'])
            if test_metrics['reward'] > self.metrics['best_reward']:
                self.metrics['best_reward'] = test_metrics['reward']
                self._save_model('best')
                self.logger.info(f"Initial best model with reward: {test_metrics['reward']:.2f}")

        # If already at or beyond target, skip training
        if start_epoch >= epochs:
            self.logger.info(f"Already at epoch {start_epoch} >= {epochs}, no training performed.")
            self._save_metrics()
            self._print_training_summary(0)
            return
        
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
        
        pbar = tqdm(range(start_epoch, epochs), desc="Training", unit="epoch", initial=start_epoch, total=epochs)
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

            if self.complexity_manager.epochs_without_progress >= self.complexity_manager.stagnation_termination:
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
            self.complexity_manager.add_performance(epoch_reward)
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
            
            # Test with exact epoch tracking
            if epoch % test_interval == 0 and epoch != 0:
                if epoch not in self.metrics['test_epochs']:  # avoid duplicate after resume
                    test_metrics = self._test_valid(epochs=4)
                    test_reward = test_metrics['reward']
                    self.metrics['test_epochs'].append(epoch)
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
            
            self.lr_scheduler.step()

        # Final test only if we actually trained new epochs and the last epoch wasn't already a test epoch
        if len(self.metrics['train_rewards']) > start_epoch:
            last_epoch = len(self.metrics['train_rewards']) - 1
            if last_epoch not in self.metrics['test_epochs']:
                test_metrics = self._test_valid(epochs=4)
                test_reward = test_metrics['reward']
                self.metrics['test_epochs'].append(last_epoch)
                self.metrics['test_rewards'].append(test_reward)
                if test_reward > self.metrics['best_reward']:
                    self.metrics['best_reward'] = test_reward
                    self._save_model('best')
                    self.logger.info(f"New best model with reward: {test_reward:.2f}")
            else:
                self.logger.info(f"Epoch {last_epoch} already tested, skipping final test.")

        self._save_model('final')
        self._save_metrics()
        self._print_training_summary(start_time)

        """

# ============================================================================
# FACTORY: Return appropriate trainer based on config
# ============================================================================

def Trainer(config: Dict[str, Any]):
    """Factory function returning either ParallelTrainer or AdaptiveParallelTrainer"""
    if config['training'].get('dynamic_complexity', False):
        return AdaptiveParallelTrainer(config)
    else:
        return ParallelTrainer(config)