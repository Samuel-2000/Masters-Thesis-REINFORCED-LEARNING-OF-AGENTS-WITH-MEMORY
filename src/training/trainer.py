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
from .optimizers import GradientClipper, LearningRateScheduler

import cv2


class ComplexityManager:
    """Manages dynamic complexity adjustment based on agent performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config['training']
        
        # Dynamic complexity settings
        self.enabled = self.training_config.get('dynamic_complexity', False)
        self.performance_window = self.training_config.get('performance_window', 100)
        self.increase_threshold = self.training_config.get('complexity_increase_threshold', 0.95)
        self.decrease_threshold = self.training_config.get('complexity_decrease_threshold', 0.7)
        self.complexity_step = self.training_config.get('complexity_step', 0.05)
        self.min_complexity = self.training_config.get('min_complexity', 0.0)
        self.max_complexity = self.training_config.get('max_complexity', 1.0)
        self.adjustment_interval = self.training_config.get('adjustment_interval', 500)
        self.stagnation_threshold = self.training_config.get('stagnation_threshold', 10)  # epochs without progress
        self.stagnation_check_interval = self.training_config.get('stagnation_check_interval', 100)  # NEW: check every 100 episodes
        self.min_basic_complexity = self.training_config.get('min_basic_complexity', 0.3)  # NEW: minimum before switching from basic
        self.curriculum_stages = self.training_config.get('curriculum_stages', 
                                                         ["basic", "doors", "buttons", "complex"])
        
        # Current state - each stage maintains its own complexity
        self.current_stage_idx = 0
        self.stage_complexities = {stage: config['environment'].get('complexity_level', 0.0) 
                                  for stage in self.curriculum_stages}
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.performance_window)
        self.max_rewards_by_stage = {}
        self.epochs_without_progress = 0  # Tracks stagnation
        self.last_max_reward = -float('inf')
        self.last_stagnation_check = 0  # NEW: track when we last checked for stagnation
        
        # Statistics
        self.adjustments_made = 0
        self.last_adjustment_epoch = 0
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
            return self.config['environment'].get('task_class', 'basic')
        return self.curriculum_stages[self.current_stage_idx]
    
    def get_current_complexity(self) -> float:
        """Get current complexity level"""
        if not self.enabled:
            return self.config['environment'].get('complexity_level', 0.0)
        
        current_stage = self.get_current_task_class()
        return self.stage_complexities[current_stage]
    
    def should_adjust(self, epoch: int) -> bool:
        """Check if we should adjust complexity"""
        if not self.enabled:
            return False
        
        # Check if enough time has passed since last adjustment
        if epoch - self.last_adjustment_epoch < self.adjustment_interval:
            return False
        
        # Check if we have enough performance data
        if len(self.performance_history) < self.performance_window // 2:
            return False
        
        return True
    
    def should_switch_stage(self, epoch: int) -> bool:
        """Check if we should switch to a different task due to stagnation"""
        if not self.enabled:
            return False
        
        # NEW: Only check stagnation every stagnation_check_interval episodes
        if epoch - self.last_stagnation_check < self.stagnation_check_interval:
            return False
        
        self.last_stagnation_check = epoch  # Update last check time
        
        current_stage = self.get_current_task_class()
        current_complexity = self.stage_complexities[current_stage]
        
        # NEW: Prevent switching from basic until it reaches minimum complexity
        if current_stage == 'basic' and current_complexity < self.min_basic_complexity:
            return False
        
        # Switch if we've been stuck for too long
        if self.epochs_without_progress >= self.stagnation_threshold:
            return True
        
        # Optional: Check for oscillation (could be enhanced with more sophisticated detection)
        if len(self.performance_history) >= self.performance_window:
            recent_std = np.std(list(self.performance_history)[-self.performance_window//2:])
            recent_mean = np.mean(list(self.performance_history)[-self.performance_window//2:])
            if recent_std < 0.1 and recent_mean < self.decrease_threshold:
                return True  # Stuck at low performance with little variation
        
        return False

    def calculate_performance_score(self) -> float:
        if not self.performance_history:
            return 0.0
        
        avg_performance = np.mean(list(self.performance_history))
        
        # Track maximum reward seen for current stage
        current_stage = self.get_current_task_class()
        stage_idx = self.current_stage_idx
        
        if stage_idx not in self.max_rewards_by_stage:
            self.max_rewards_by_stage[stage_idx] = avg_performance
        else:
            self.max_rewards_by_stage[stage_idx] = max(
                self.max_rewards_by_stage[stage_idx],
                avg_performance
            )
        
        max_reward = self.max_rewards_by_stage[stage_idx]
        
        # Avoid division by zero
        if max_reward < 0.1:
            return 0.0
        
        # Normalize by maximum observed reward for current stage
        normalized = avg_performance / max_reward
        return max(0.0, min(1.0, normalized))

    def switch_to_next_stage(self) -> Dict[str, Any]:
        """Switch to next task in curriculum, maintaining each task's complexity"""
        old_stage_idx = self.current_stage_idx
        old_stage = self.curriculum_stages[old_stage_idx]
        old_complexity = self.stage_complexities[old_stage]
        
        # Choose next stage (round robin, but could be random or based on other criteria)
        next_idx = (old_stage_idx + 1) % len(self.curriculum_stages)
        
        # NEW: Ensure we don't switch back to basic unless all other tasks are at high complexity
        if next_idx == 0 and old_stage_idx > 0:
            # Check if other stages are mastered enough to return to basic
            other_stages_mastered = all(
                self.stage_complexities[stage] >= 0.8 
                for stage in self.curriculum_stages[1:]
            )
            
            if not other_stages_mastered:
                # Skip basic and go to the next non-basic stage
                next_idx = 1 if len(self.curriculum_stages) > 1 else 0
        
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
            "adjusted": True,
            "reason": f"Stagnation for {self.epochs_without_progress} epochs"
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
            self.last_adjustment_epoch = epoch
            return adjustment_info
        
        # Otherwise, proceed with normal complexity adjustment
        performance_score = self.calculate_performance_score()
        current_stage = self.get_current_task_class()
        old_complexity = self.stage_complexities[current_stage]
        
        adjustment_info = {
            "old_complexity": old_complexity,
            "old_stage": current_stage,
            "performance_score": performance_score,
            "adjusted": False
        }
        
        # NEW: Special rule for basic stage - focus on reaching minimum complexity
        if current_stage == 'basic' and old_complexity < self.min_basic_complexity:
            # Force complexity increase for basic until we reach minimum
            if performance_score > 0.8:  # Lower threshold for basic when building up
                new_complexity = min(self.min_basic_complexity, 
                                   old_complexity + self.complexity_step)
                self.stage_complexities[current_stage] = new_complexity
                adjustment_info["action"] = "forced_increase_basic"
                adjustment_info["new_complexity"] = new_complexity
                adjustment_info["new_stage"] = current_stage
                adjustment_info["adjusted"] = True
                self.epochs_without_progress = 0
        # Normal complexity adjustment for other stages or basic after minimum
        elif performance_score > self.increase_threshold:
            # Increase complexity for current stage
            new_complexity = min(self.max_complexity, 
                               old_complexity + self.complexity_step)
            if new_complexity > old_complexity:
                self.stage_complexities[current_stage] = new_complexity
                adjustment_info["action"] = "increased_complexity"
                adjustment_info["new_complexity"] = new_complexity
                adjustment_info["new_stage"] = current_stage
                adjustment_info["adjusted"] = True
                self.epochs_without_progress = 0  # Reset stagnation counter
        
        elif performance_score < self.decrease_threshold:
            # Decrease complexity for current stage
            new_complexity = max(self.min_complexity,
                               old_complexity - self.complexity_step)
            if new_complexity < old_complexity:
                self.stage_complexities[current_stage] = new_complexity
                adjustment_info["action"] = "decreased_complexity"
                adjustment_info["new_complexity"] = new_complexity
                adjustment_info["new_stage"] = current_stage
                adjustment_info["adjusted"] = True
        
        if adjustment_info["adjusted"]:
            self.adjustments_made += 1
            self.last_adjustment_epoch = epoch
            self.performance_history.clear()  # Reset performance history after adjustment
        
        return adjustment_info if adjustment_info["adjusted"] else None
    
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
                env_config['door_periodic'] = False
                env_config['button_break_probability'] = 0.0
                
            elif current_stage == 'doors':
                # Doors stage: periodic doors, no buttons
                env_config['n_doors'] = -1
                env_config['n_buttons_per_door'] = 0
                env_config['door_periodic'] = True
                env_config['button_break_probability'] = 0.0
                
            elif current_stage == 'buttons':
                # Buttons stage: doors with buttons
                env_config['n_doors'] = -1
                env_config['n_buttons_per_door'] = -1
                env_config['door_periodic'] = False
                env_config['button_break_probability'] = -1.0
                
            elif current_stage == 'complex':
                # Complex stage: mix of periodic and button doors
                env_config['n_doors'] = -1
                env_config['n_buttons_per_door'] = -1
                env_config['door_periodic'] = True  # Some doors periodic
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
            "performance_score": self.calculate_performance_score() if self.performance_history else 0.0,
            "max_rewards_by_stage": self.max_rewards_by_stage,
            "basic_min_complexity_reached": self.stage_complexities['basic'] >= self.min_basic_complexity
        }


class AdaptiveParallelTrainer:
    """Main trainer class with parallel execution and dynamic complexity"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 use_wandb: bool = False):
        
        self.config = config
        self.experiment_name = f"{config["model"]["type"]}_" \
                                f"{config["training"]["batch_size"]}b_" \
                                f"{config["training"]["learning_rate"]}lr_" \
                                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        if config["training"].get("auxiliary_tasks", None):
            self.experiment_name += "_aux"
        
        if config["training"].get("dynamic_complexity", False):
            self.experiment_name += "_dynamic"
        
        self.use_wandb = use_wandb
        
        # Setup
        self.logger = setup_logging(self.experiment_name)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Set seed
        seed_everything(config.get('seed', 42))
        
        # Get batch size for parallel environments
        training_config = self.config['training']
        self.batch_size = training_config['batch_size']
        
        # Create complexity manager
        self.complexity_manager = ComplexityManager(config)
        
        # Create initial vectorized environment
        self.vector_env = self._create_vectorized_env()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        
        # Create loss functions
        self.policy_loss_fn = PolicyLoss(
            gamma=training_config.get('gamma', 0.97),
            entropy_coef=training_config.get('entropy_coef', 0.01),
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
            max_norm=training_config.get('max_grad_norm', 1.0)
        )
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            mode='cosine',
            lr_start=training_config.get('learning_rate', 0.0005),
            lr_min=1e-6
        )
        
        # Metrics
        self.metrics = {
            'train_rewards': [],
            'train_losses': [],
            'test_rewards': [],
            'best_reward': -np.inf,
            'timing': {
                'collection': [],
                'training': [],
                'total': []
            },
            'complexity_history': [],  # Track complexity over time
            'task_class_history': [],  # Track task class over time
            'performance_scores': []   # Track performance scores
        }
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project="maze-rl-dynamic",
                name=self.experiment_name,
                config=config
            )


    def _visualize_current_environments(self, epoch: int):
        """Visualize a sample of current training environments"""
        print(f"\n📸 DEBUG: Visualizing environments at epoch {epoch}")
        
        # Get current complexity status
        status = self.complexity_manager.get_status()
        print(f"  Stage: {status['current_stage']}")
        print(f"  Complexity: {status['current_complexity']:.2f}")
        print(f"  n_doors parameter: {self.vector_env.envs[0].n_doors}")
        print(f"  Actual doors created: {len(self.vector_env.envs[0].doors)}")
        print(f"  Actual buttons created: {len(self.vector_env.envs[0].buttons)}")
        
        # Visualize first N environments (e.g., 4)
        num_to_show = min(4, len(self.vector_env.envs))
        
        # Create a grid of environments
        cell_size = 256
        padding = 10
        cols = 2
        rows = (num_to_show + cols - 1) // cols
        
        total_width = cols * cell_size + (cols + 1) * padding
        total_height = rows * cell_size + (rows + 1) * padding
        
        combined_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        
        for i in range(num_to_show):
            env = self.vector_env.envs[i]
            
            # Print environment info
            print(f"\n  Environment {i}:")
            print(f"    Grid size: {env.grid_size}")
            if hasattr(env, 'agent_pos') and env.agent_pos is not None:
                print(f"    Agent position: ({env.agent_pos[0]}, {env.agent_pos[1]})")
            print(f"    Doors: {len(env.doors)}")
            print(f"    Buttons: {len(env.buttons)}")
            if env.doors:
                for j, door in enumerate(env.doors[:3]):  # Show first 3 doors
                    print(f"      Door {j}: ({door.y},{door.x}), open={door.is_open}, needs_button={door.requires_button}")
            if env.buttons:
                for j, button in enumerate(env.buttons[:3]):  # Show first 3 buttons
                    print(f"      Button {j}: ({button.y},{button.x}), broken={button.is_broken}, door_idx={button.door_idx}")
            
            # Get the rendered frame
            # Temporarily enable rendering by calling the parent class's render method
            try:
                # Store original render size
                original_render_size = env.render_size
                
                # Temporarily set render size to enable rendering
                env.render_size = cell_size
                
                # Clear the render buffer to force re-render
                if hasattr(env, '_render_buffer'):
                    env._render_buffer = None
                
                # Call the parent class's render method (GridMazeWorld.render)
                # We need to call it through super()
                frame = super(type(env), env).render()
                
                # Restore original render size
                env.render_size = original_render_size
                
                if frame is None:
                    frame = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Env {i}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as e:
                print(f"    Warning: Could not render environment {i}: {e}")
                frame = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                cv2.putText(frame, f"Error: {str(e)[:20]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Resize to cell size if needed
            if frame.shape[0] != cell_size or frame.shape[1] != cell_size:
                frame = cv2.resize(frame, (cell_size, cell_size))
            
            # Calculate position
            col = i % cols
            row = i // cols
            
            x_start = padding + col * (cell_size + padding)
            y_start = padding + row * (cell_size + padding)
            
            # Place frame in combined image
            combined_image[y_start:y_start+cell_size, x_start:x_start+cell_size] = frame
        
        # Add title/header to the image
        title = f"Epoch {epoch}: {status['current_stage']} (Complexity: {status['current_complexity']:.2f})"
        cv2.putText(combined_image, title, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info = f"Doors: {len(self.vector_env.envs[0].doors)}, Buttons: {len(self.vector_env.envs[0].buttons)}"
        cv2.putText(combined_image, info, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show the image
        cv2.imshow('Training Visualization', combined_image)
        
        # Wait for key press to continue
        print("\n👀 Press any key to continue training...")
        cv2.waitKey(0)
        
    
    def _create_vectorized_env(self) -> VectorizedMazeEnv:
        """Create vectorized training environment with current complexity"""
        env_config = self.complexity_manager.get_environment_config()
        # DEBUG: Print current environment config
        current_stage = self.complexity_manager.get_current_task_class()
        current_complexity = self.complexity_manager.get_current_complexity()

        #print(f"\n🚪 DEBUG: Creating vectorized env - Stage: {current_stage}, Complexity: {current_complexity:.2f}")
        #print(f"   n_doors in config: {env_config.get('n_doors', 0)}")
        #print(f"   task_class in config: {env_config.get('task_class', 'basic')}")

        return VectorizedMazeEnv(
            num_envs=self.batch_size,
            env_config=env_config
        )
    
    def _recreate_vectorized_env(self):
        """Recreate vectorized environment with updated complexity"""
        # Close old environment
        if hasattr(self, 'vector_env'):
            self.vector_env.close()
        
        # Create new environment
        self.vector_env = self._create_vectorized_env()
        #self.logger.info(f"Recreated environment with task_class: {self.complexity_manager.get_current_task_class()}, "
        #                f"complexity: {self.complexity_manager.get_current_complexity():.2f}")
    
    def _create_agent(self) -> Agent:
        """Create agent with specified network"""
        model_config = self.config['model']
        
        agent = Agent(
            network_type=model_config['type'],
            observation_size=10,  # Fixed observation size
            action_size=6,  # Fixed action size
            hidden_size=model_config.get('hidden_size', 512),
            use_auxiliary=model_config.get('use_auxiliary', False),
            device=self.device
        )
        
        # Load pretrained if specified
        if 'pretrained_path' in model_config:
            agent.load(model_config['pretrained_path'])
        
        return agent
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        training_config = self.config['training']
        
        optimizer_type = training_config.get('optimizer', 'adam')
        lr = training_config.get('learning_rate', 0.0005)
        weight_decay = training_config.get('weight_decay', 0.0)
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.agent.network.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=1e-8
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.agent.network.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def train(self):
        """Main training loop with parallel execution and dynamic complexity"""
        print("\n🎮 Training Controls:")
        print("  Press 'v' during training to visualize current environments")
        print("  Press 'q' to stop training early")
        print("=" * 50)

        training_config = self.config['training']
        epochs = training_config.get('epochs', 10000)
        save_interval = training_config.get('save_interval', 1000)
        test_interval = training_config.get('test_interval', 500)
        
        # Create progress bar
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        
        start_time = time.time()
        
        # Log initial complexity status
        complexity_status = self.complexity_manager.get_status()
        self.logger.info(f"Initial complexity status: {complexity_status}")


        cv2.namedWindow('Training Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Training Controls', 400, 100)
        cv2.putText(np.zeros((100, 400, 3), dtype=np.uint8), "Press 'v' to visualize", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(np.zeros((100, 400, 3), dtype=np.uint8), "Press 'q' to quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Training Controls', np.zeros((100, 400, 3), dtype=np.uint8))
        cv2.waitKey(1)  # Force window creation
        
        for epoch in pbar:

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('v'):  # 'v' key pressed
                print(f"\n📸 Key 'v' detected! Visualizing environments at epoch {epoch}")
                self._visualize_current_environments(epoch)
                # Force window focus back to control window
                cv2.imshow('Training Controls', np.zeros((100, 400, 3), dtype=np.uint8))
                
            elif key == ord('q'):  # 'q' key pressed
                print("\n⚠️  User pressed 'q' - early stop requested.")
                self._save_model('interrupted')
                cv2.destroyAllWindows()
                break
            
            elif key != 255:  # 255 means no key pressed
                print(f"Key pressed: {key} (char: {chr(key) if key < 128 else key})")

            epoch_start = time.time()
            
            # Training phase with timing
            coll_start = time.time()
            experiences = self._collect_experiences_parallel()
            coll_time = time.time() - coll_start
            
            train_start = time.time()
            train_metrics = self._train_step(experiences)
            train_time = time.time() - train_start
            
            # Get epoch reward (average across batch)
            epoch_reward = train_metrics['reward']
            
            # Update performance history for complexity adjustment
            self.complexity_manager.add_performance(epoch_reward, epoch)
            
            # Check and adjust complexity if needed
            adjustment = self.complexity_manager.adjust_complexity(epoch)
            if adjustment:
                self._handle_complexity_adjustment(adjustment, epoch)
            
            # Update metrics
            self.metrics['train_rewards'].append(epoch_reward)
            self.metrics['train_losses'].append(train_metrics['loss'])
            self.metrics['timing']['collection'].append(coll_time)
            self.metrics['timing']['training'].append(train_time)
            self.metrics['timing']['total'].append(time.time() - epoch_start)
            
            # Track complexity history
            self.metrics['complexity_history'].append(self.complexity_manager.get_current_complexity())
            self.metrics['task_class_history'].append(self.complexity_manager.get_current_task_class())
            self.metrics['performance_scores'].append(self.complexity_manager.calculate_performance_score())
            
            # Test phase
            if epoch % test_interval == 0 and epoch != 0:
                test_metrics = self._test_epoch(episodes=10)
                test_reward = test_metrics['reward']
                self.metrics['test_rewards'].append(test_reward)
                
                # Update best model
                if test_reward > self.metrics['best_reward']:
                    self.metrics['best_reward'] = test_reward
                    self._save_model('best')
                    self.logger.info(f"New best model with reward: {test_reward:.2f}")
            
            # Save checkpoint
            if epoch % save_interval == 0 and epoch != 0:
                self._save_model(f'epoch_{epoch:06d}')
            
            # Update progress bar
            avg_coll_time = np.mean(self.metrics['timing']['collection'][-10:]) if len(self.metrics['timing']['collection']) > 10 else coll_time
            avg_train_time = np.mean(self.metrics['timing']['training'][-10:]) if len(self.metrics['timing']['training']) > 10 else train_time
            
            # Get current complexity info
            current_stage = self.complexity_manager.get_current_task_class()
            current_complexity = self.complexity_manager.get_current_complexity()

            perf_score = self.complexity_manager.calculate_performance_score()
            
            pbar.set_postfix({
                'reward': f"{train_metrics['reward']:.2f}",
                'loss': f"{train_metrics['loss']:.4f}",
                'best': f"{self.metrics['best_reward']:.2f}",
                'stage': current_stage,
                'comp': f"{current_complexity:.2f}",
                'perf': f"{perf_score:.2f}",
                'adj': self.complexity_manager.adjustments_made,
                'eps/s': f"{self.batch_size/(avg_coll_time+avg_train_time):.1f}",
            })
            
            # Log to wandb
            if self.use_wandb:
                log_data = {
                    'train/reward': train_metrics['reward'],
                    'train/loss': train_metrics['loss'],
                    'train/entropy': train_metrics.get('entropy', 0.0),
                    'lr': self.lr_scheduler.get_lr(),
                    'timing/collection': coll_time,
                    'timing/training': train_time,
                    'timing/env_steps_per_sec': self.batch_size / coll_time,
                    'complexity/current': current_complexity,
                    'complexity/stage': self._stage_to_numeric(current_stage),
                    'complexity/adjustments': self.complexity_manager.adjustments_made,
                    'complexity/performance_score': self.complexity_manager.calculate_performance_score(),
                }
                
                if adjustment:
                    log_data.update({
                        'complexity/adjustment_action': adjustment.get('action', 'none'),
                        'complexity/old_complexity': adjustment.get('old_complexity', 0.0),
                        'complexity/new_complexity': adjustment.get('new_complexity', 0.0),
                    })
                
                wandb.log(log_data)
                
                if epoch % test_interval == 0 and epoch != 0:
                    wandb.log({
                        'test/reward': test_metrics['reward'],
                        'test/success_rate': test_metrics['success_rate'],
                    })
            
            # Update learning rate
            self.lr_scheduler.step()
        
        # Save final model
        self._save_model('final')
        
        # Save training metrics
        self._save_metrics()
        
        # Print training summary
        self._print_training_summary(start_time)
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
    
    def _handle_complexity_adjustment(self, adjustment: Dict[str, Any], epoch: int):
        """Handle complexity adjustment"""
        action = adjustment.get('action', 'unknown')
        old_complexity = adjustment.get('old_complexity', 0.0)
        new_complexity = adjustment.get('new_complexity', 0.0)
        old_stage = adjustment.get('old_stage', 'basic')
        new_stage = adjustment.get('new_stage', 'basic')
        
        # Log the adjustment
        #self.logger.info(f"Complexity adjustment at epoch {epoch}:")
        #self.logger.info(f"  Action: {action}")
        #self.logger.info(f"  Stage: {old_stage} -> {new_stage}")
        #self.logger.info(f"  Complexity: {old_complexity:.2f} -> {new_complexity:.2f}")
        #self.logger.info(f"  Performance score: {adjustment.get('performance_score', 0.0):.2f}")
        
        # Recreate environment with new complexity
        if old_stage != new_stage or abs(old_complexity - new_complexity) > 0.01:
            self._recreate_vectorized_env()
    
    def _stage_to_numeric(self, stage: str) -> float:
        """Convert stage name to numeric value for logging"""
        stage_map = {
            "basic": 0.0,
            "doors": 0.33,
            "buttons": 0.66,
            "complex": 1.0
        }
        return stage_map.get(stage, 0.0)
    
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
        
        # Run for max_steps or until all environments are done
        active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(max_steps):
            # Store current observations
            all_observations.append(observations.clone())
            
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
            obs_array, rewards, terminated, truncated, _ = self.vector_env.step(actions_np)
            
            # Convert to tensors
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            terminated_tensor = torch.tensor(terminated, dtype=torch.bool, device=self.device)
            truncated_tensor = torch.tensor(truncated, dtype=torch.bool, device=self.device)
            
            # Store actions and rewards
            all_actions.append(actions)
            all_rewards.append(rewards_tensor)
            
            # Update active mask
            done_mask = terminated_tensor | truncated_tensor
            active_mask = active_mask & ~done_mask
            
            # Break if all environments are done
            if not active_mask.any():
                break
            
            # Prepare next observations
            observations = torch.tensor(obs_array, dtype=torch.long, device=self.device)
            observations = observations.unsqueeze(1)  # [B, 1, K]
        
        # Stack all collected data
        T = len(all_observations)
        
        observations_tensor = torch.cat(all_observations, dim=1)  # [B, T, K]
        actions_tensor = torch.stack(all_actions, dim=1)  # [B, T]
        rewards_tensor = torch.stack(all_rewards, dim=1)  # [B, T]
        
        # Create mask for valid steps
        mask = torch.ones_like(rewards_tensor, dtype=torch.float32)
        
        return {
            'observations': observations_tensor,
            'actions': actions_tensor,
            'rewards': rewards_tensor,
            'mask': mask
        }
    
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
                if len(outputs) == 4:
                    logits, energy_pred, obs_pred, _ = outputs
                else:
                    logits, energy_pred, obs_pred = outputs
            else:
                # Network doesn't return auxiliary outputs
                logits = outputs
                energy_pred = None
                obs_pred = None
            
            # Policy loss
            policy_loss, entropy = self.policy_loss_fn(
                logits, actions, rewards, mask
            )
            
            total_loss = policy_loss
            
            metrics = {
                'loss': policy_loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy': entropy.item(),
                'reward': rewards.sum(dim=1).mean().item()
            }
            
            # Add auxiliary loss if available
            if energy_pred is not None and obs_pred is not None:
                # Note: We don't have energy targets in basic implementation
                # You'd need to collect these during experience collection
                pass
            
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
    
    def _test_epoch(self, episodes: int = 10) -> Dict[str, float]:
        """Test agent performance with current complexity"""
        self.agent.network.eval()
        
        total_reward = 0.0
        success_count = 0
        episode_lengths = []
        
        with torch.no_grad():
            for _ in range(episodes):
                # Create test environment with current complexity
                env_config = self.complexity_manager.get_environment_config()
                env_config['render_size'] = 0  # No rendering for testing
                
                test_env = GridMazeWorld(**env_config)
                obs, info = test_env.reset()
                self.agent.reset()
                
                episode_reward = 0.0
                steps = 0
                terminated = truncated = False
                
                while not (terminated or truncated) and steps < test_env.max_steps:
                    action = self.agent.act(obs, training=False)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                
                total_reward += episode_reward  # Use cumulative reward
                episode_lengths.append(steps)
                
                # Consider episode successful if agent survives to end
                if steps == test_env.max_steps:
                    success_count += 1
        
        avg_reward = total_reward / episodes
        success_rate = success_count / episodes * 100
        avg_length = np.mean(episode_lengths)
        
        return {
            'reward': avg_reward,
            'success_rate': success_rate,
            'avg_length': avg_length
        }
    
    def _save_model(self, name: str):
        """Save model checkpoint with complexity information"""
        save_dir = Path(self.config.get('save_dir', 'models'))
        save_dir.mkdir(exist_ok=True)
        
        # For 'best' and 'final', save the agent file
        if name in ['best', 'final']:
            agent_path = save_dir / f"{self.experiment_name}_{name}.pt"
            
            # Save additional complexity information
            agent_state = {
                'state_dict': self.agent.network.state_dict(),
                'config': self.agent.network.get_config() if hasattr(self.agent.network, 'get_config') else {},
                'complexity_status': self.complexity_manager.get_status(),
                'training_metrics': {
                    'best_reward': self.metrics['best_reward'],
                    'current_epoch': len(self.metrics['train_rewards']),
                    'current_complexity': self.complexity_manager.get_current_complexity(),
                    'current_task_class': self.complexity_manager.get_current_task_class()
                }
            }
            
            torch.save(agent_state, str(agent_path))
            self.logger.info(f"Saved agent to {agent_path}")
        
        # Save checkpoint (for resuming training)
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
                'adjustments_made': self.complexity_manager.adjustments_made,
                'last_adjustment_epoch': self.complexity_manager.last_adjustment_epoch
            },
            'config': self.config
        }
        
        torch.save(checkpoint, str(checkpoint_path))
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_metrics(self):
        """Save training metrics including complexity history"""
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
        
        # Plot metrics
        self._plot_metrics()
    
    def _plot_metrics(self):
        """Plot training metrics including complexity progression"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # 1. Raw and smoothed rewards
        ax = axes[0, 0]
        rewards = self.metrics['train_rewards']
        ax.plot(rewards, alpha=0.3, color='gray', linewidth=0.5, label='Raw')
        
        # Add smoothed line
        if len(rewards) >= 100:
            window = 100
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, 
                   'r-', linewidth=2, label=f'Smoothed (window={window})')
        
        ax.set_title('Training Rewards')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Complexity progression
        ax = axes[0, 1]
        complexity = self.metrics['complexity_history']
        task_class_numeric = [self._stage_to_numeric(stage) for stage in self.metrics['task_class_history']]
        
        # Plot task class as background shading
        for i in range(len(task_class_numeric)-1):
            if task_class_numeric[i] != task_class_numeric[i+1]:
                ax.axvline(x=i, color='gray', alpha=0.3, linestyle='--')
        
        ax.plot(complexity, 'b-', linewidth=2, label='Complexity')
        ax.set_title('Complexity Progression')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Complexity Level')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add task class labels
        y_pos = 1.05
        for stage in ["basic", "doors", "buttons", "complex"]:
            numeric_value = self._stage_to_numeric(stage)
            if numeric_value in task_class_numeric:
                first_occurrence = task_class_numeric.index(numeric_value)
                ax.text(first_occurrence, y_pos, stage, fontsize=8, alpha=0.7)
        
        # 3. Training losses
        ax = axes[1, 0]
        ax.plot(self.metrics['train_losses'], alpha=0.7, linewidth=1)
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # 4. Performance scores
        ax = axes[1, 1]
        if self.metrics['performance_scores']:
            ax.plot(self.metrics['performance_scores'], 'g-', linewidth=1.5, label='Performance Score')
            ax.axhline(y=self.complexity_manager.increase_threshold, color='r', 
                      linestyle='--', alpha=0.5, label='Increase Threshold')
            ax.axhline(y=self.complexity_manager.decrease_threshold, color='orange', 
                      linestyle='--', alpha=0.5, label='Decrease Threshold')
            ax.set_title('Performance Scores')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        # 5. Test rewards
        if self.metrics['test_rewards']:
            ax = axes[2, 0]
            test_rewards = self.metrics['test_rewards']
            test_interval = len(self.metrics['train_rewards']) // len(test_rewards)
            x_vals = np.arange(test_interval, len(test_rewards)*test_interval + 1, test_interval)
            ax.plot(x_vals, test_rewards, 'o-', linewidth=2, markersize=6)
            ax.set_title('Test Rewards')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
        
        # 6. Complexity vs Reward correlation
        ax = axes[2, 1]
        if len(self.metrics['train_rewards']) > 100:
            # Use smoothed rewards for correlation
            window = min(100, len(self.metrics['train_rewards']) // 10)
            smoothed_rewards = np.convolve(self.metrics['train_rewards'], 
                                          np.ones(window)/window, mode='valid')
            smoothed_complexity = np.convolve(self.metrics['complexity_history'],
                                            np.ones(window)/window, mode='valid')
            
            # Align arrays
            min_len = min(len(smoothed_rewards), len(smoothed_complexity))
            ax.scatter(smoothed_complexity[:min_len], smoothed_rewards[:min_len],
                      alpha=0.5, s=10, c=range(min_len), cmap='viridis')
            ax.set_title('Complexity vs Reward Correlation')
            ax.set_xlabel('Complexity Level')
            ax.set_ylabel('Smoothed Reward')
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            if min_len > 10:
                correlation = np.corrcoef(smoothed_complexity[:min_len], 
                                         smoothed_rewards[:min_len])[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Training Progress: {self.experiment_name}', fontsize=16)
        plt.tight_layout()
        
        plot_path = Path('results/plots') / f"{self.experiment_name}_metrics.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create dedicated complexity progression plot
        self._create_complexity_progression_plot()
        
        self.logger.info(f"Saved metrics plot to {plot_path}")
    
    def _create_complexity_progression_plot(self):
        """Create a detailed complexity progression plot"""
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(12, 10))
        
        # Create 4 subplots
        ax1 = plt.subplot(3, 2, 1)
        ax2 = plt.subplot(3, 2, 2)
        ax3 = plt.subplot(3, 2, 3)
        ax4 = plt.subplot(3, 2, 4)
        ax5 = plt.subplot(3, 2, (5, 6))
        
        # 1. Complexity progression
        ax1.plot(self.metrics['complexity_history'], 'b-', linewidth=2)
        ax1.set_title('Complexity Level Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Complexity')
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3)
        
        # Mark adjustment points
        adjustments = []
        for i in range(1, len(self.metrics['complexity_history'])):
            if abs(self.metrics['complexity_history'][i] - self.metrics['complexity_history'][i-1]) > 0.01:
                adjustments.append(i)
        
        if adjustments:
            ax1.scatter(adjustments, [self.metrics['complexity_history'][i] for i in adjustments],
                       color='red', s=50, zorder=5, label='Adjustments')
            ax1.legend()
        
        # 2. Task class progression
        task_class_numeric = [self._stage_to_numeric(stage) for stage in self.metrics['task_class_history']]
        ax2.plot(task_class_numeric, 'g-', linewidth=2)
        ax2.set_title('Task Class Progression')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Task Class')
        ax2.set_yticks([0.0, 0.33, 0.66, 1.0])
        ax2.set_yticklabels(['Basic', 'Doors', 'Buttons', 'Complex'])
        ax2.set_ylim([-0.1, 1.1])
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance scores
        ax3.plot(self.metrics['performance_scores'], 'orange', linewidth=2)
        ax3.axhline(y=self.complexity_manager.increase_threshold, 
                   color='green', linestyle='--', alpha=0.7, label='Increase threshold')
        ax3.axhline(y=self.complexity_manager.decrease_threshold,
                   color='red', linestyle='--', alpha=0.7, label='Decrease threshold')
        ax3.set_title('Performance Scores')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_ylim([0, 1.1])
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # 4. Reward vs Complexity phase
        window = min(50, len(self.metrics['train_rewards']) // 10)
        if window > 1:
            smoothed_rewards = np.convolve(self.metrics['train_rewards'], 
                                          np.ones(window)/window, mode='valid')
            smoothed_complexity = np.convolve(self.metrics['complexity_history'],
                                            np.ones(window)/window, mode='valid')
            
            min_len = min(len(smoothed_rewards), len(smoothed_complexity))
            sc = ax4.scatter(smoothed_complexity[:min_len], smoothed_rewards[:min_len],
                           c=range(min_len), cmap='viridis', alpha=0.7, s=30)
            ax4.set_title('Reward vs Complexity (Smoothed)')
            ax4.set_xlabel('Complexity Level')
            ax4.set_ylabel('Reward')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax4, label='Epoch (smoothed)')
        
        # 5. Combined view
        epochs = range(len(self.metrics['train_rewards']))
        color_map = plt.cm.viridis
        
        # Plot rewards (scaled)
        scaled_rewards = np.array(self.metrics['train_rewards'])
        if scaled_rewards.max() > scaled_rewards.min():
            scaled_rewards = (scaled_rewards - scaled_rewards.min()) / (scaled_rewards.max() - scaled_rewards.min())
        
        ax5.plot(epochs, scaled_rewards, 'b-', alpha=0.5, linewidth=1, label='Reward (scaled)')
        ax5.plot(epochs, self.metrics['complexity_history'], 'r-', linewidth=2, label='Complexity')
        ax5.plot(epochs, task_class_numeric, 'g--', linewidth=1.5, label='Task Class')
        
        # Fill between for task classes
        for i, stage in enumerate(["basic", "doors", "buttons", "complex"]):
            numeric_value = self._stage_to_numeric(stage)
            if numeric_value in task_class_numeric:
                # Find epochs where this stage is active
                stage_epochs = [j for j, val in enumerate(task_class_numeric) if val == numeric_value]
                if stage_epochs:
                    start = min(stage_epochs)
                    end = max(stage_epochs)
                    ax5.axvspan(start, end, alpha=0.1, color=color_map(i/4))
                    ax5.text((start+end)/2, 1.05, stage, 
                            ha='center', fontsize=9, alpha=0.7)
        
        ax5.set_title('Training Progression Overview')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Value (scaled)')
        ax5.set_ylim([0, 1.1])
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper right')
        
        plt.suptitle(f'Dynamic Complexity Training: {self.experiment_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        complexity_plot_path = Path('results/plots') / f"{self.experiment_name}_complexity_progression.png"
        plt.savefig(str(complexity_plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved complexity progression plot to {complexity_plot_path}")
    
    def _print_training_summary(self, start_time: float):
        """Print detailed training summary"""
        total_time = time.time() - start_time
        avg_collection = np.mean(self.metrics['timing']['collection'])
        avg_training = np.mean(self.metrics['timing']['training'])
        avg_total = np.mean(self.metrics['timing']['total'])
        
        final_complexity = self.complexity_manager.get_current_complexity()
        final_stage = self.complexity_manager.get_current_task_class()
        
        print(f"\n{'='*80}")
        print("DYNAMIC COMPLEXITY TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average epoch time: {avg_total:.3f}s (collection: {avg_collection:.3f}s, training: {avg_training:.3f}s)")
        print(f"Average environment steps per second: {self.batch_size/avg_collection:.1f}")
        print(f"Final best reward: {self.metrics['best_reward']:.2f}")
        
        print(f"\nComplexity Progression:")
        print(f"  Final stage: {final_stage}")
        print(f"  Final complexity: {final_complexity:.2f}")
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
        print(f"Metrics saved to: logs/metrics/{self.experiment_name}_metrics.npz")
        print(f"Plots saved to: results/plots/{self.experiment_name}_*.png")
        print(f"{'='*80}")


# For backward compatibility
Trainer = AdaptiveParallelTrainer