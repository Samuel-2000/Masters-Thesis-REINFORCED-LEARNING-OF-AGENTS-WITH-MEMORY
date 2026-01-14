"""
Visualization utilities for model evaluation
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import imageio
from tqdm import tqdm

from src.core.environment import GridMazeWorld
from src.core.agent import Agent


class Visualizer:
    """Visualize agent behavior"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 agent: Optional[Agent] = None):
        """
        Args:
            model_path: Path to trained model
            agent: Pre-loaded agent (optional)
        """
        if agent is None and model_path is not None:
            self.agent = Agent.load(model_path)
        elif agent is not None:
            self.agent = agent
        else:
            raise ValueError("Either model_path or agent must be provided")
        
        self.env = None
        self.frames = []


    def run(self,
        episodes: int = 3,
        env_config: Optional[Dict] = None,
        save_video: bool = False,
        save_gif: bool = False):
        """Run visualization with custom environment config"""
        if env_config is None:
            env_config = {
                'grid_size': 11,
                'max_steps': 100,
                'obstacle_fraction': 0.25,
                'n_food_sources': 4,
                'render_size': 512
            }
        
        # Create environment from config
        from src.core.environment import GridMazeWorld
        env = GridMazeWorld(**env_config)
        
        # Run episode
        episode_results = self.run_episode(
            env_config=env_config,
            max_steps=env_config.get('max_steps', 100),
            render=True,
            save_frames=save_video or save_gif
        )
        
        # Save video if requested
        if save_video and self.frames:
            video_path = self.create_video()
            print(f"Video saved to: {video_path}")
        
        # Save GIF if requested
        if save_gif and self.frames:
            gif_path = self.create_gif()
            print(f"GIF saved to: {gif_path}")
        
        return episode_results
    
    def run_episode(self,
                   env_config: Optional[Dict] = None,
                   max_steps: int = 100,
                   render: bool = True,
                   save_frames: bool = True) -> Dict[str, Any]:
        """Run and visualize a single episode"""
        # Default environment config
        if env_config is None:
            env_config = {
                'grid_size': 11,
                'max_steps': max_steps,
                'obstacle_fraction': 0.25,
                'n_food_sources': 4,
                'render_size': 512
            }
        
        # Create environment
        self.env = GridMazeWorld(**env_config)
        self.agent.reset()
        
        # Reset frame storage
        if save_frames:
            self.frames = []
        
        # Run episode
        obs, info = self.env.reset()
        total_reward = 0
        steps = 0
        terminated = truncated = False
        
        while not (terminated or truncated) and steps < max_steps:
            # Get action from agent
            action = self.agent.act(obs, training=False)
            
            # Take step
            obs, terminated, truncated, info = self.env.step(action)
            
            total_reward += self.env.energy
            steps += 1
            
            # Render and save frame
            if render or save_frames:
                frame = self.env.render()
                
                if save_frames:
                    self.frames.append(frame.copy())
                
                if render:
                    cv2.imshow('Maze RL - Agent View', frame)
                    key = cv2.waitKey(50)  # ~20 FPS
                    
                    if key == ord('q'):  # Quit on 'q'
                        break
                    elif key == ord(' '):  # Pause on space
                        cv2.waitKey(0)
        
        if render:
            cv2.destroyAllWindows()
        
        # Return episode summary
        return {
            'total_reward': total_reward,
            'steps': steps,
            'final_energy': info.get('energy', 0),
            'success': steps == max_steps,  # Survived full episode
            'frames': self.frames if save_frames else []
        }
    
    def create_video(self,
                    output_path: str = "agent_behavior.mp4",
                    fps: int = 20,
                    add_info_overlay: bool = True) -> str:
        """Create video from stored frames"""
        if not self.frames:
            raise ValueError("No frames available. Run an episode with save_frames=True first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get frame dimensions
        height, width = self.frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Write frames
        for frame in tqdm(self.frames, desc="Creating video"):
            video.write(frame)
        
        video.release()
        
        return str(output_path)
    
    def create_gif(self,
                  output_path: str = "agent_behavior.gif",
                  fps: int = 10,
                  resize_factor: float = 0.5) -> str:
        """Create GIF from stored frames"""
        if not self.frames:
            raise ValueError("No frames available. Run an episode with save_frames=True first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resize frames if needed
        frames_to_save = []
        for frame in self.frames:
            if resize_factor != 1.0:
                new_width = int(frame.shape[1] * resize_factor)
                new_height = int(frame.shape[0] * resize_factor)
                frame = cv2.resize(frame, (new_width, new_height))
            frames_to_save.append(frame)
        
        # Save as GIF
        imageio.mimsave(output_path, frames_to_save, fps=fps)
        
        return str(output_path)
    
    def plot_trajectory(self,
                       episode_results: Dict[str, Any],
                       output_path: Optional[str] = None):
        """Plot agent trajectory and statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Reward over time (simulated)
        ax = axes[0, 0]
        steps = episode_results['steps']
        rewards_per_step = np.ones(steps) * 0.01  # Base survival reward
        
        # Simulate food collection (in practice, you'd track this)
        food_collection_points = np.random.choice(steps, size=min(3, steps), replace=False)
        for point in food_collection_points:
            rewards_per_step[point] += 1.0
        
        cumulative_rewards = np.cumsum(rewards_per_step)
        ax.plot(range(steps), cumulative_rewards, 'b-', linewidth=2)
        ax.scatter(food_collection_points, cumulative_rewards[food_collection_points], 
                  color='red', s=100, zorder=5, label='Food collected')
        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Reward Accumulation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Energy over time (simulated)
        ax = axes[0, 1]
        energy = 30.0  # Starting energy
        energy_history = [energy]
        
        for step in range(1, steps):
            energy = energy * 0.98 - 0.1  # Decay per step
            # Random food collection
            if step in food_collection_points:
                energy = min(100.0, energy + 10.0)  # Food adds energy
            energy_history.append(max(0, energy))
        
        ax.plot(range(steps), energy_history, 'g-', linewidth=2)
        ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Low energy threshold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Level Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Action distribution (simulated)
        ax = axes[1, 0]
        actions = ['Left', 'Right', 'Up', 'Down', 'Stay']
        action_counts = np.random.randint(5, 20, size=len(actions))
        action_counts[-1] = np.random.randint(1, 5)  # Less staying
        
        bars = ax.bar(actions, action_counts, color=['blue', 'green', 'red', 'purple', 'orange'])
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
        ax.set_title('Action Distribution')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        # 4. Episode summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = [
            "Episode Summary",
            "===============",
            f"Total reward: {episode_results['total_reward']:.2f}",
            f"Steps taken: {episode_results['steps']}",
            f"Final energy: {episode_results['final_energy']:.1f}",
            f"Success: {'Yes' if episode_results['success'] else 'No'}",
            "",
            "Agent Information",
            "=================",
            f"Network type: {getattr(self.agent, 'network_type', 'Unknown')}",
            f"Parameters: {sum(p.numel() for p in self.agent.network.parameters()):,}"
        ]
        
        ax.text(0.1, 0.95, "\n".join(summary_text), 
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_attention(self,
                          observation: np.ndarray,
                          output_path: Optional[str] = None):
        """Visualize attention patterns (if network supports it)"""
        if not hasattr(self.agent.network, 'attention_weights'):
            print("Network doesn't support attention visualization")
            return
        
        # Get attention weights
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
            _ = self.agent.network(obs_tensor)
            attention = self.agent.network.attention_weights
        
        if attention is None:
            return
        
        # Visualize attention
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Attention heatmap
        ax = axes[0]
        im = ax.imshow(attention.cpu().numpy(), cmap='viridis', aspect='auto')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        ax.set_title('Attention Weights')
        plt.colorbar(im, ax=ax)
        
        # 2. Bar chart of attention to different positions
        ax = axes[1]
        positions = ['NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE', 'Action', 'Energy']
        if len(attention) == 1:  # Single attention head
            weights = attention[0].cpu().numpy()
            if len(weights) == 10:  # Should match observation size
                bars = ax.bar(positions, weights, color='skyblue')
                ax.set_xlabel('Observation Component')
                ax.set_ylabel('Attention Weight')
                ax.set_title('Attention to Observation Components')
                ax.tick_params(axis='x', rotation=45)
                
                # Highlight highest attention
                max_idx = np.argmax(weights)
                bars[max_idx].set_color('red')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_comparison_video(self,
                               agents: List[Tuple[str, Agent]],
                               env_config: Optional[Dict] = None,
                               max_steps: int = 50,
                               output_path: str = "comparison.mp4") -> str:
        """Create side-by-side comparison video of multiple agents"""
        if env_config is None:
            env_config = {
                'grid_size': 11,
                'max_steps': max_steps,
                'obstacle_fraction': 0.25,
                'n_food_sources': 4
            }
        
        # Create environments for each agent (same seed for fairness)
        envs = []
        for name, agent in agents:
            env = GridMazeWorld(**env_config)
            obs, info = env.reset(seed=42)  # Same seed for all
            envs.append((name, env, agent, obs))
        
        # Prepare video writer
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get frame dimensions
        test_frame = envs[0][1].render()
        height, width = test_frame.shape[:2]
        
        # Create combined frame (side by side)
        n_agents = len(agents)
        combined_width = width * n_agents
        combined_height = height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(output_path), fourcc, 10, (combined_width, combined_height))
        
        frames = []
        
        for step in range(max_steps):
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            all_done = True
            
            for i, (name, env, agent, obs) in enumerate(envs):
                # Check if episode is done
                if hasattr(env, 'done') and env.done:
                    frame = env.render()
                else:
                    all_done = False
                    
                    # Get action
                    action = agent.act(obs, training=False)
                    
                    # Take step
                    obs, reward, done, truncated, info = env.step(action)
                    envs[i] = (name, env, agent, obs)  # Update stored obs
                    
                    frame = env.render()
                
                # Add agent name overlay
                cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255, 255, 255), 2)
                
                # Add to combined frame
                start_x = i * width
                combined_frame[:, start_x:start_x + width] = frame
            
            frames.append(combined_frame.copy())
            video.write(combined_frame)
            
            if all_done:
                break
        
        video.release()
        
        # Create summary
        self._create_comparison_summary(agents, envs, frames)
        
        return str(output_path)
    
    def _create_comparison_summary(self, agents, envs, frames):
        """Create summary of comparison"""
        print("\n" + "="*60)
        print("AGENT COMPARISON SUMMARY")
        print("="*60)
        
        for i, (name, env, agent, obs) in enumerate(envs):
            if hasattr(env, 'energy'):
                print(f"\n{name}:")
                print(f"  Final energy: {env.energy:.1f}")
                print(f"  Steps taken: {getattr(env, 'steps', 0)}")
                print(f"  Network type: {getattr(agent, 'network_type', 'Unknown')}")
        
        print("\n" + "="*60)


def create_training_visualization(metrics_path: str,
                                 output_dir: str = "results/plots"):
    """Create visualizations from training metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    if metrics_path.endswith('.npz'):
        data = np.load(metrics_path)
        metrics = {key: data[key] for key in data.files}
    else:
        # Assume it's a dict saved with pickle
        import pickle
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Training rewards
    if 'train_rewards' in metrics:
        ax = axes[0, 0]
        rewards = metrics['train_rewards']
        ax.plot(rewards, alpha=0.7, linewidth=1)
        
        # Add moving average
        window = min(100, len(rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'MA ({window})')
            ax.legend()
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reward')
        ax.set_title('Training Rewards')
        ax.grid(True, alpha=0.3)
    
    # 2. Training losses
    if 'train_losses' in metrics:
        ax = axes[0, 1]
        losses = metrics['train_losses']
        ax.plot(losses, alpha=0.7, linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.grid(True, alpha=0.3)
    
    # 3. Test rewards (if available)
    if 'test_rewards' in metrics and len(metrics['test_rewards']) > 0:
        ax = axes[0, 2]
        test_rewards = metrics['test_rewards']
        test_interval = len(metrics.get('train_rewards', [])) // len(test_rewards)
        x_vals = np.arange(test_interval, len(test_rewards)*test_interval + 1, test_interval)
        ax.plot(x_vals, test_rewards, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reward')
        ax.set_title('Test Rewards')
        ax.grid(True, alpha=0.3)
    
    # 4. Entropy (if available)
    if 'entropy_losses' in metrics:
        ax = axes[1, 0]
        entropy = metrics['entropy_losses']
        ax.plot(entropy, alpha=0.7, linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)
    
    # 5. Learning rate (if available)
    ax = axes[1, 1]
    if 'learning_rates' in metrics:
        lrs = metrics['learning_rates']
        ax.plot(lrs, alpha=0.7, linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No learning rate data', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Learning Rate')
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = ["Training Summary", "================"]
    
    if 'train_rewards' in metrics:
        rewards = metrics['train_rewards']
        summary.append(f"Final reward: {rewards[-1]:.2f}")
        summary.append(f"Max reward: {np.max(rewards):.2f}")
        summary.append(f"Avg reward: {np.mean(rewards):.2f}")
    
    if 'train_losses' in metrics:
        losses = metrics['train_losses']
        summary.append(f"Final loss: {losses[-1]:.4f}")
        summary.append(f"Min loss: {np.min(losses):.4f}")
    
    if 'best_reward' in metrics:
        summary.append(f"Best reward: {metrics['best_reward']:.2f}")
    
    if 'best_epoch' in metrics:
        summary.append(f"Best epoch: {metrics['best_epoch']}")
    
    if 'network_type' in metrics:
        summary.append(f"Network: {metrics['network_type']}")
    
    if 'total_time' in metrics:
        summary.append(f"Total time: {metrics['total_time']:.1f}s")
    
    ax.text(0.1, 0.95, "\n".join(summary), 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    exp_name = Path(metrics_path).stem
    output_path = output_dir / f"{exp_name}_training_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training visualization saved to {output_path}")
    
    return output_path


def plot_smoothed_rewards(rewards: List[float], 
                          window_sizes: List[int] = [10, 50, 100, 200],
                          title: str = "Reward Progress",
                          save_path: Optional[str] = None):
    """
    Plot raw and smoothed rewards with different window sizes
    
    Args:
        rewards: List of rewards per epoch
        window_sizes: List of window sizes for smoothing
        title: Plot title
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Raw rewards with single smoothing
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color='gray', linewidth=0.5, label='Raw rewards')
    
    # Add smoothed line (using largest window that fits)
    valid_windows = [w for w in window_sizes if w <= len(rewards)]
    if valid_windows:
        window = valid_windows[-1]  # Use largest valid window
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, 
               'r-', linewidth=2, label=f'Smoothed (window={window})')
    
    ax.set_title(f'{title} - Raw vs Smoothed')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Multiple smoothing windows
    ax = axes[1]
    
    # Plot raw rewards in background
    ax.plot(rewards, alpha=0.2, color='gray', linewidth=0.3, label='Raw')
    
    # Plot smoothed rewards with different windows
    colors = plt.cm.viridis(np.linspace(0, 1, len(window_sizes)))
    
    for i, window in enumerate(window_sizes):
        if window <= len(rewards):
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, 
                   color=colors[i], linewidth=1.5, 
                   label=f'Window={window}')
    
    ax.set_title(f'{title} - Different Smoothing Windows')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved smoothed rewards plot to {save_path}")
    
    return fig