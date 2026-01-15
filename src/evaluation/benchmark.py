"""
Benchmarking utilities for model evaluation
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import warnings

from src.core.environment import GridMazeWorld
from src.core.agent import Agent
from src.core.utils import setup_logging


class Benchmark:
    """Benchmark multiple models"""
    
    def __init__(self, 
                 models_dir: str = "models",
                 output_dir: str = "results/benchmarks"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging("benchmark")
    
    def run(self,
            episodes_per_model: int = 20,
            env_config: Optional[Dict] = None,
            verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Run benchmark on all models in directory
        
        Returns:
            DataFrame with benchmark results, or None if no models found
        """
        # Default environment config
        if env_config is None:
            env_config = {
                'grid_size': 11,
                'max_steps': 100,
                'obstacle_fraction': 0.25,
                'n_food_sources': 4,
                'render_size': 0  # No rendering for faster benchmarking
            }
        
        # Find all model files
        model_files = self._find_model_files()
        
        if len(model_files) == 0:
            self.logger.warning(f"No model files found in {self.models_dir}")
            if verbose:
                print(f"No model files found in {self.models_dir}")
                print("Try training a model first: python run.py train")
            return None
        
        if verbose:
            print(f"Found {len(model_files)} models to benchmark")
        
        results = []
        
        for model_file in tqdm(model_files, desc="Benchmarking", disable=not verbose):
            try:
                model_results = self._benchmark_single(
                    model_path=model_file,
                    episodes=episodes_per_model,
                    env_config=env_config
                )
                
                model_results['model_file'] = model_file.name
                results.append(model_results)
                
                if verbose:
                    print(f"✓ {model_file.name}: "
                          f"reward={model_results['avg_reward']:.2f} ± {model_results['std_reward']:.2f}, "
                          f"success={model_results['success_rate']:.1f}%")
            
            except Exception as e:
                self.logger.error(f"Failed to benchmark {model_file.name}: {e}")
                if verbose:
                    print(f"✗ {model_file.name}: Failed - {e}")
        
        if not results:
            self.logger.warning("No models were successfully benchmarked")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            # Sort by average reward (descending)
            df = df.sort_values('avg_reward', ascending=False)
            
            # Save results
            self.save_results(df)
            
            # Generate plots
            self.plot_results(df)
        
        return df
    
    def _find_model_files(self) -> List[Path]:
        """Find all model files in directory and subdirectories"""
        model_files = []
        
        # Look for .pt files
        for ext in ['*.pt', '*.pth']:
            model_files.extend(list(self.models_dir.rglob(ext)))
        
        # Filter out checkpoints if we have best models
        best_models = []
        checkpoint_models = []
        final_models = []
        
        for model_file in model_files:
            if 'checkpoint' in model_file.name or 'epoch' in model_file.name:
                checkpoint_models.append(model_file)
            elif 'best' in model_file.name:
                best_models.append(model_file)
            elif 'final' in model_file.name:
                # Skip final.pt files
                continue
            else:
                # Other model files
                checkpoint_models.append(model_file)
        
        # Return best.pt files if found, otherwise checkpoints
        if best_models:
            return best_models
        else:
            return checkpoint_models
    

    def _benchmark_single(self,
                        model_path: Path,
                        episodes: int,
                        env_config: Dict) -> Dict[str, Any]:
        """Benchmark a single model"""
        # Load agent with error handling
        try:
            agent = Agent.load(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load agent: {e}")
        
        # Create environment using GridMazeWorld directly
        from src.core.environment import GridMazeWorld
        env = GridMazeWorld(**env_config)
        
        # Run episodes
        rewards = []
        success_flags = []
        steps_list = []
        energies = []
        
        for episode in range(episodes):
            obs, info = env.reset()
            agent.reset()
            
            episode_reward = 0
            steps = 0
            terminated = truncated = False
            
            while not (terminated or truncated):
                action = agent.act(obs, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)  # Use cumulative reward
            success_flags.append(not terminated and steps == env.max_steps)  # Survived full episode
            steps_list.append(steps)
            energies.append(info.get('energy', 0))
        
        # Compute statistics
        results = {
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'success_rate': float(np.mean(success_flags) * 100),
            'avg_steps': float(np.mean(steps_list)),
            'std_steps': float(np.std(steps_list)),
            'avg_final_energy': float(np.mean(energies)),
            'num_episodes': episodes
        }
        
        # Add network info if available
        if hasattr(agent, 'network_type'):
            results['network_type'] = agent.network_type
        if hasattr(agent, 'use_auxiliary'):
            results['use_auxiliary'] = agent.use_auxiliary
        
        return results
    
    def save_results(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}"
        
        # Save as CSV
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON for easier reading
        json_path = self.output_dir / f"{filename}.json"
        df.to_json(json_path, orient='records', indent=2)
        
        # Save summary markdown
        md_path = self.output_dir / f"{filename}.md"
        self._create_markdown_report(df, md_path)
        
        self.logger.info(f"Results saved to {csv_path}")
        print(f"Results saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
    
    def plot_results(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Generate visualization plots from benchmark results"""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_plots_{timestamp}"
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # Define subplots
        gs = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(gs[0, 0])  # Reward distribution
        ax2 = fig.add_subplot(gs[0, 1])  # Reward vs success rate
        ax3 = fig.add_subplot(gs[0, 2])  # Network type comparison
        ax4 = fig.add_subplot(gs[1, 0])  # Reward histogram
        ax5 = fig.add_subplot(gs[1, 1])  # Steps vs reward
        ax6 = fig.add_subplot(gs[1, 2])  # Auxiliary task effect
        ax7 = fig.add_subplot(gs[2, :])  # Summary table
        
        # 1. Reward distribution
        if len(df) > 1:
            sns.boxplot(data=df, y='avg_reward', ax=ax1, color='skyblue')
        else:
            ax1.bar([0], df['avg_reward'].values, color='skyblue')
        ax1.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Reward')
        ax1.grid(True, alpha=0.3)
        
        # 2. Success rate vs reward
        scatter = ax2.scatter(df['avg_reward'], df['success_rate'], 
                            alpha=0.7, s=100, c=df.index, cmap='viridis')
        ax2.set_xlabel('Average Reward', fontsize=11)
        ax2.set_ylabel('Success Rate (%)', fontsize=11)
        ax2.set_title('Reward vs Success Rate', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add model names as annotations for top performers
        top_n = min(5, len(df))
        for i in range(top_n):
            row = df.iloc[i]
            model_name = row['model_file'].replace('.pt', '')[:15]
            ax2.annotate(model_name, 
                        (row['avg_reward'], row['success_rate']),
                        fontsize=8, alpha=0.8, xytext=(5, 5),
                        textcoords='offset points')
        
        # 3. Network type comparison
        if 'network_type' in df.columns and len(df['network_type'].unique()) > 1:
            network_data = []
            for network_type in df['network_type'].unique():
                network_df = df[df['network_type'] == network_type]
                if not network_df.empty:
                    network_data.append({
                        'network': network_type,
                        'mean_reward': network_df['avg_reward'].mean(),
                        'std_reward': network_df['avg_reward'].std(),
                        'count': len(network_df)
                    })
            
            if network_data:
                network_df = pd.DataFrame(network_data)
                x = range(len(network_df))
                bars = ax3.bar(x, network_df['mean_reward'], 
                              yerr=network_df['std_reward'], 
                              capsize=5, color=['#3498db', '#2ecc71', '#e74c3c'])
                ax3.set_xticks(x)
                ax3.set_xticklabels(network_df['network'], rotation=45, ha='right')
                ax3.set_title('Performance by Network Type', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Average Reward')
                ax3.grid(True, alpha=0.3)
                
                # Add count labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    count = network_df.iloc[i]['count']
                    ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                            f"n={count}", ha='center', va='center', 
                            color='white', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No network type data available', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=11)
            ax3.set_title('Network Type Comparison', fontsize=12, fontweight='bold')
        
        # 4. Reward histogram
        ax4.hist(df['avg_reward'], bins=min(10, len(df)), 
                alpha=0.7, edgecolor='black', color='#9b59b6')
        ax4.set_xlabel('Average Reward', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Reward Distribution Histogram', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Steps vs reward
        ax5.scatter(df['avg_steps'], df['avg_reward'], alpha=0.6, s=80)
        ax5.set_xlabel('Average Steps', fontsize=11)
        ax5.set_ylabel('Average Reward', fontsize=11)
        ax5.set_title('Steps vs Reward', fontsize=12, fontweight='bold')
        
        # Add trend line if we have enough data
        if len(df) > 1:
            z = np.polyfit(df['avg_steps'], df['avg_reward'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df['avg_steps'].min(), df['avg_steps'].max(), 100)
            ax5.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
        
        ax5.grid(True, alpha=0.3)
        
        # 6. Auxiliary task effect
        if 'use_auxiliary' in df.columns:
            aux_data = df.groupby('use_auxiliary').agg({
                'avg_reward': ['mean', 'std', 'count'],
                'success_rate': 'mean'
            }).round(2)
            
            if len(aux_data) > 1:
                x = ['No Auxiliary', 'With Auxiliary']
                means = [aux_data.loc[False, ('avg_reward', 'mean')], 
                        aux_data.loc[True, ('avg_reward', 'mean')]]
                stds = [aux_data.loc[False, ('avg_reward', 'std')], 
                       aux_data.loc[True, ('avg_reward', 'std')]]
                
                bars = ax6.bar(x, means, yerr=stds, capsize=5,
                              color=['#e74c3c', '#2ecc71'])
                ax6.set_ylabel('Average Reward')
                ax6.set_title('Effect of Auxiliary Tasks', fontsize=12, fontweight='bold')
                ax6.grid(True, alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom')
            else:
                ax6.text(0.5, 0.5, 'Insufficient auxiliary task data', 
                        ha='center', va='center', transform=ax6.transAxes,
                        fontsize=11)
                ax6.set_title('Auxiliary Task Effect', fontsize=12, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No auxiliary task data', 
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=11)
            ax6.set_title('Auxiliary Task Effect', fontsize=12, fontweight='bold')
        
        # 7. Summary table
        ax7.axis('off')
        
        # Create summary text
        summary_lines = [
            "SUMMARY STATISTICS",
            "=" * 50,
            f"Total models benchmarked: {len(df)}",
            f"Average reward (all): {df['avg_reward'].mean():.2f} ± {df['avg_reward'].std():.2f}",
            f"Best reward: {df['avg_reward'].max():.2f} ({df.loc[df['avg_reward'].idxmax(), 'model_file']})",
            f"Worst reward: {df['avg_reward'].min():.2f} ({df.loc[df['avg_reward'].idxmin(), 'model_file']})",
            f"Average success rate: {df['success_rate'].mean():.1f}%",
            "",
            "TOP 3 MODELS:"
        ]
        
        for i in range(min(3, len(df))):
            model_name = df.iloc[i]['model_file'].replace('.pt', '')
            reward = df.iloc[i]['avg_reward']
            success = df.iloc[i]['success_rate']
            steps = df.iloc[i]['avg_steps']
            
            network_info = ""
            if 'network_type' in df.columns:
                network_info = f", {df.iloc[i]['network_type']}"
            if 'use_auxiliary' in df.columns:
                aux_info = " +aux" if df.iloc[i]['use_auxiliary'] else ""
                network_info += aux_info
            
            summary_lines.append(f"{i+1}. {model_name}{network_info}")
            summary_lines.append(f"   Reward: {reward:.2f}, Success: {success:.1f}%, Steps: {steps:.1f}")
        
        summary_lines.append("")
        summary_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        ax7.text(0.05, 0.95, "\n".join(summary_lines), 
                transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Model Benchmark Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plots saved to {plot_path}")
        print(f"Plots saved to: {plot_path}")
        
        return plot_path
    
    def _create_markdown_report(self, df: pd.DataFrame, path: Path):
        """Create markdown report from benchmark results"""
        with open(path, 'w') as f:
            f.write("# Benchmark Results\n\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of models: {len(df)}\n")
            f.write(f"Episodes per model: {df['num_episodes'].iloc[0] if len(df) > 0 else 0}\n\n")
            
            f.write("## Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## Summary\n\n")
            
            if 'network_type' in df.columns:
                f.write("### By Network Type\n\n")
                network_stats = df.groupby('network_type').agg({
                    'avg_reward': ['count', 'mean', 'max'],
                    'success_rate': 'max'
                }).round(2)
                
                for network_type in df['network_type'].unique():
                    type_df = df[df['network_type'] == network_type]
                    if len(type_df) > 0:
                        f.write(f"#### {network_type.upper()}\n")
                        f.write(f"- Count: {len(type_df)}\n")
                        f.write(f"- Average reward: {type_df['avg_reward'].mean():.2f}\n")
                        f.write(f"- Best reward: {type_df['avg_reward'].max():.2f}\n")
                        f.write(f"- Best success rate: {type_df['success_rate'].max():.1f}%\n\n")
            
            if 'use_auxiliary' in df.columns:
                f.write("### With vs Without Auxiliary Tasks\n\n")
                aux_groups = df.groupby('use_auxiliary')
                for aux, aux_df in aux_groups:
                    label = "With Auxiliary Tasks" if aux else "Without Auxiliary Tasks"
                    f.write(f"#### {label}\n")
                    f.write(f"- Count: {len(aux_df)}\n")
                    f.write(f"- Average reward: {aux_df['avg_reward'].mean():.2f}\n")
                    f.write(f"- Average success rate: {aux_df['success_rate'].mean():.1f}%\n\n")
            
            f.write("### Top 3 Models\n\n")
            for i, row in df.head(3).iterrows():
                f.write(f"{i+1}. **{row['model_file']}**\n")
                f.write(f"   - Reward: {row['avg_reward']:.2f} ± {row['std_reward']:.2f}\n")
                f.write(f"   - Success rate: {row['success_rate']:.1f}%\n")
                f.write(f"   - Steps: {row['avg_steps']:.1f} ± {row['std_steps']:.1f}\n")
                if 'network_type' in row:
                    f.write(f"   - Network type: {row['network_type']}\n")
                f.write("\n")
    
    def compare_architectures(self,
                             architectures: List[str] = None,
                             episodes_per_arch: int = 10,
                             trials: int = 3) -> pd.DataFrame:
        """
        Compare different architectures from scratch
        
        Args:
            architectures: List of architecture names to compare
            episodes_per_arch: Episodes per architecture per trial
            trials: Number of trials per architecture
        
        Returns:
            DataFrame with comparison results
        """
        if architectures is None:
            architectures = ['lstm', 'transformer', 'multimemory']
        
        results = []
        
        for arch in architectures:
            for trial in range(trials):
                print(f"Testing {arch} architecture, trial {trial+1}/{trials}...")
                
                # Create fresh agent with this architecture
                agent = Agent(
                    network_type=arch,
                    observation_size=10,
                    action_size=6,
                    hidden_size=128,  # Smaller for quick comparison
                    use_auxiliary=False
                )
                
                # Create environment
                env = GridMazeWorld(grid_size=11, max_steps=50)
                
                # Run evaluation
                rewards = []
                for episode in range(episodes_per_arch):
                    obs, info = env.reset()
                    agent.reset()
                    
                    episode_reward = 0
                    terminated = truncated = False
                    
                    while not (terminated or truncated):
                        action = agent.act(obs, training=False)
                        obs, reward, terminated, truncated, info = env.step(action)
                    
                    rewards.append(env.energy)
                
                # Record results
                results.append({
                    'architecture': arch,
                    'trial': trial + 1,
                    'avg_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'min_reward': np.min(rewards),
                    'max_reward': np.max(rewards),
                    'episodes': episodes_per_arch
                })
        
        df = pd.DataFrame(results)
        
        # Save comparison results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"architecture_comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate comparison plot
        self._plot_architecture_comparison(df)
        
        return df
    
    def _plot_architecture_comparison(self, df: pd.DataFrame):
        """Plot architecture comparison results"""
        if df.empty:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Group by architecture
        grouped = df.groupby('architecture')
        
        # Create box plot
        data_to_plot = [group['avg_reward'].values for name, group in grouped]
        labels = [name.upper() for name in grouped.groups.keys()]
        
        box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Set colors
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add individual points
        for i, (name, group) in enumerate(grouped):
            x = np.random.normal(i + 1, 0.04, size=len(group))
            plt.scatter(x, group['avg_reward'], alpha=0.6, color='black', s=50)
        
        plt.title('Architecture Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Architecture', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = []
        for i, (name, group) in enumerate(grouped):
            mean = group['avg_reward'].mean()
            std = group['avg_reward'].std()
            stats_text.append(f"{name.upper()}: {mean:.2f} ± {std:.2f} (n={len(group)})")
        
        plt.figtext(0.02, 0.02, "\n".join(stats_text), 
                   fontsize=10, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"architecture_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Architecture comparison plot saved to {plot_path}")
        print(f"Architecture comparison plot saved to: {plot_path}")


def create_benchmark_summary(results_dir: str = "results/benchmarks"):
    """Create a summary of all benchmark runs"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find all benchmark result files
    json_files = list(results_dir.glob("benchmark_*.json"))
    
    if not json_files:
        print("No benchmark results found")
        return
    
    print(f"\nFound {len(json_files)} benchmark result files")
    print("=" * 60)
    
    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
            
            if isinstance(results, list) and len(results) > 0:
                # New format (list of records)
                df = pd.DataFrame(results)
                timestamp = json_file.stem.replace("benchmark_", "")
                print(f"\n{timestamp}:")
                print(f"  Models: {len(df)}")
                print(f"  Best reward: {df['avg_reward'].max():.2f}")
                print(f"  Best model: {df.loc[df['avg_reward'].idxmax(), 'model_file']}")
        except:
            continue


if __name__ == "__main__":
    # Example usage
    benchmark = Benchmark()
    
    # Benchmark existing models
    results = benchmark.run(episodes_per_model=10, verbose=True)
    
    # Or compare architectures from scratch
    # results = benchmark.compare_architectures()