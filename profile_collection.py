# profile_train_scale.py
import torch
import numpy as np
import time
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import ParallelTrainer
from src.core.utils import load_config, seed_everything
import psutil
import GPUtil
from threading import Thread
import json


class ResourceMonitor:
    """Monitor CPU, GPU, and memory usage"""
    
    def __init__(self, interval=0.1):
        self.interval = interval
        self.cpu_percents = []
        self.memory_percents = []
        self.gpu_usages = []
        self.gpu_memories = []
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = Thread(target=self._monitor)
        self.thread.start()
    
    def _monitor(self):
        while self.running:
            # CPU usage
            self.cpu_percents.append(psutil.cpu_percent())
            
            # Memory usage
            self.memory_percents.append(psutil.virtual_memory().percent)
            
            # GPU usage (if available)
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    self.gpu_usages.append(gpu.load * 100)
                    self.gpu_memories.append(gpu.memoryUtil * 100)
            except:
                pass
            
            time.sleep(self.interval)
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def get_stats(self):
        stats = {
            'cpu_mean': np.mean(self.cpu_percents) if self.cpu_percents else 0,
            'cpu_max': np.max(self.cpu_percents) if self.cpu_percents else 0,
            'memory_mean': np.mean(self.memory_percents) if self.memory_percents else 0,
            'memory_max': np.max(self.memory_percents) if self.memory_percents else 0,
        }
        
        if self.gpu_usages:
            stats.update({
                'gpu_usage_mean': np.mean(self.gpu_usages),
                'gpu_usage_max': np.max(self.gpu_usages),
                'gpu_memory_mean': np.mean(self.gpu_memories),
                'gpu_memory_max': np.max(self.gpu_memories),
            })
        
        return stats


def create_full_scale_trainer():
    """Create trainer with actual training configuration"""
    config = {
        'experiment': {
            'name': 'full_scale_profiling',
            'seed': 42
        },
        'environment': {
            'grid_size': 11,
            'max_steps': 100,
            'obstacle_fraction': 0.25,
            'n_food_sources': 4,
            'food_energy': 10.0,
            'initial_energy': 30.0,
            'energy_decay': 0.98,
            'energy_per_step': 0.1
        },
        'model': {
            'type': 'lstm',  # default
            'hidden_size': 512,  # default
            'use_auxiliary': False
        },
        'training': {
            'epochs': 10000,  # default
            'batch_size': 64,  # ACTUAL BATCH SIZE
            'learning_rate': 0.0005,  # default
            'gamma': 0.97,
            'entropy_coef': 0.01,
            'max_grad_norm': 1.0,
            'save_interval': 1000,
            'test_interval': 500,
            'optimizer': 'adam'
        }
    }
    
    return ParallelTrainer(config, use_wandb=False)


def simulate_training_epoch(trainer):
    """Simulate a full training epoch including collection and training"""
    print("\nSimulating full training epoch...")
    
    # Start resource monitoring
    monitor = ResourceMonitor(interval=0.05)
    monitor.start()
    
    epoch_metrics = {}
    
    # Phase 1: Experience Collection
    print("  Phase 1: Experience Collection")
    collection_times = []
    
    for i in range(3):  # Collect multiple times like in training
        start_time = time.time()
        experiences = trainer._collect_experiences_parallel()
        collection_time = time.time() - start_time
        collection_times.append(collection_time)
        
        # Get experience stats
        B, T, K = experiences['observations'].shape
        total_steps = B * T
        
        print(f"    Collection {i+1}: {collection_time:.3f}s, "
              f"Steps: {total_steps:,}, Steps/sec: {total_steps/collection_time:.1f}")
    
    epoch_metrics['collection'] = {
        'times': collection_times,
        'avg_time': np.mean(collection_times),
        'avg_steps_per_sec': total_steps / np.mean(collection_times),
        'batch_size': B,
        'sequence_length': T,
        'observation_size': K
    }
    
    # Phase 2: Training Step
    print("\n  Phase 2: Training Step")
    training_times = []
    backward_times = []
    
    for i in range(3):  # Multiple training steps
        # Training step
        trainer.agent.network.train()
        
        # Forward pass
        forward_start = time.time()
        loss, metrics = trainer._compute_loss(experiences)
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.gradient_clipper.clip(trainer.agent.network.parameters())
        trainer.optimizer.step()
        backward_time = time.time() - backward_start
        
        training_time = forward_time + backward_time
        training_times.append(training_time)
        backward_times.append(backward_time)
        
        print(f"    Training {i+1}: {training_time:.3f}s "
              f"(Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s)")
    
    epoch_metrics['training'] = {
        'times': training_times,
        'avg_time': np.mean(training_times),
        'forward_times': [forward_time],  # Would need to track per iteration
        'backward_times': backward_times,
        'avg_backward_time': np.mean(backward_times)
    }
    
    # Stop monitoring and get stats
    monitor.stop()
    resource_stats = monitor.get_stats()
    
    epoch_metrics['resources'] = resource_stats
    epoch_metrics['total_epoch_time'] = np.mean(collection_times) + np.mean(training_times)
    
    return epoch_metrics


def analyze_bottlenecks(metrics):
    """Analyze where the bottlenecks are"""
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)
    
    collection_time = metrics['collection']['avg_time']
    training_time = metrics['training']['avg_time']
    total_time = metrics['total_epoch_time']
    
    print(f"\nTime breakdown:")
    print(f"  Experience Collection: {collection_time:.3f}s ({collection_time/total_time*100:.1f}%)")
    print(f"  Training Step:        {training_time:.3f}s ({training_time/total_time*100:.1f}%)")
    
    # Collection bottleneck analysis
    print(f"\nCollection performance:")
    print(f"  Steps per second: {metrics['collection']['avg_steps_per_sec']:.1f}")
    print(f"  Batch steps per second: {64 * metrics['collection']['avg_steps_per_sec']:.1f}")
    
    # Training bottleneck analysis
    if 'backward_times' in metrics['training']:
        avg_backward = np.mean(metrics['training']['backward_times'])
        print(f"\nTraining performance:")
        print(f"  Backward pass: {avg_backward:.3f}s ({avg_backward/training_time*100:.1f}% of training)")
    
    # Resource utilization
    resources = metrics.get('resources', {})
    if resources:
        print(f"\nResource utilization:")
        print(f"  CPU: {resources.get('cpu_mean', 0):.1f}% (max: {resources.get('cpu_max', 0):.1f}%)")
        print(f"  Memory: {resources.get('memory_mean', 0):.1f}% (max: {resources.get('memory_max', 0):.1f}%)")
        
        if 'gpu_usage_mean' in resources:
            print(f"  GPU Usage: {resources['gpu_usage_mean']:.1f}% (max: {resources['gpu_usage_max']:.1f}%)")
            print(f"  GPU Memory: {resources['gpu_memory_mean']:.1f}% (max: {resources['gpu_memory_max']:.1f}%)")
            
            if resources['gpu_usage_mean'] < 50:
                print("  ⚠️  GPU underutilized - consider larger batch size")
            if resources['gpu_memory_mean'] > 80:
                print("  ⚠️  GPU memory near limit - consider smaller batch size")
    
    # Identify bottlenecks
    bottleneck_threshold = 0.6  # 60% of total time
    
    if collection_time / total_time > bottleneck_threshold:
        print("\n🚨 MAJOR BOTTLENECK: Experience Collection")
        print("   Optimization suggestions:")
        print("   1. Use vectorized environment operations")
        print("   2. Implement JIT compilation for environment")
        print("   3. Use asynchronous data collection")
        print("   4. Profile individual environment steps")
    
    if training_time / total_time > bottleneck_threshold:
        print("\n🚨 MAJOR BOTTLENECK: Training Step")
        print("   Optimization suggestions:")
        print("   1. Use mixed precision (FP16) training")
        print("   2. Reduce model size (smaller hidden_dim)")
        print("   3. Use gradient accumulation for larger effective batch")
        print("   4. Profile forward/backward passes separately")
    
    if collection_time / total_time > 0.4 and training_time / total_time > 0.4:
        print("\n⚠️  BALANCED WORKLOAD: Both collection and training take significant time")
        print("   Consider overlapping them with asynchronous execution")


def profile_collection_detailed(trainer):
    """Detailed profiling of just the collection phase"""
    print("\n" + "=" * 80)
    print("DETAILED COLLECTION PROFILING")
    print("=" * 80)
    
    # Measure multiple runs for statistics
    num_runs = 10
    component_times = {
        'total': [],
        'env_reset': [],
        'network_inference': [],
        'env_step': [],
        'tensor_conversions': [],
        'data_storage': []
    }
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Time environment reset
        reset_start = time.time()
        obs_array, _ = trainer.vector_env.reset()
        reset_time = time.time() - reset_start
        component_times['env_reset'].append(reset_time)
        
        # Initialize
        trainer.agent.reset()
        max_steps = trainer.vector_env.envs[0].max_steps
        batch_size = trainer.batch_size
        
        # Convert initial observations
        conv_start = time.time()
        observations = torch.tensor(obs_array, dtype=torch.long).to(trainer.device)
        observations = observations.unsqueeze(1)
        conv_time = time.time() - conv_start
        
        total_start = time.time()
        steps_completed = 0
        
        for step in range(max_steps):
            step_start = time.time()
            
            # Network inference
            net_start = time.time()
            with torch.no_grad():
                trainer.agent.network.eval()
                logits = trainer.agent.network(observations)
                logits = logits.squeeze(1)
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)
            net_time = time.time() - net_start
            
            # Convert actions
            conv2_start = time.time()
            actions_np = actions.cpu().numpy()
            conv2_time = time.time() - conv2_start
            
            # Environment step
            env_start = time.time()
            obs_array, rewards, terminated, truncated, _ = trainer.vector_env.step(actions_np)
            env_time = time.time() - env_start
            
            # Update observations
            conv3_start = time.time()
            observations = torch.tensor(obs_array, dtype=torch.long, device=trainer.device)
            observations = observations.unsqueeze(1)
            conv3_time = time.time() - conv3_start
            
            # Accumulate times
            component_times['network_inference'].append(net_time)
            component_times['env_step'].append(env_time)
            component_times['tensor_conversions'].append(conv_time + conv2_time + conv3_time)
            
            steps_completed += 1
            
            # Check if all environments are done
            if all(terminated) or all(truncated):
                break
        
        total_time = time.time() - total_start
        component_times['total'].append(total_time)
        
        print(f"  Steps: {steps_completed}, Time: {total_time:.3f}s, "
              f"Steps/sec: {steps_completed/total_time:.1f}")
    
    # Print detailed statistics
    print("\nComponent timing statistics (mean ± std in milliseconds):")
    for component, times in component_times.items():
        if times:
            mean_ms = np.mean(times) * 1000
            std_ms = np.std(times) * 1000
            print(f"  {component:20s}: {mean_ms:6.2f} ± {std_ms:5.2f} ms")
    
    # Calculate overhead
    avg_step_time = np.mean(component_times['env_step']) + \
                   np.mean(component_times['network_inference']) + \
                   np.mean(component_times['tensor_conversions'])
    
    print(f"\nAverage per-step time: {avg_step_time*1000:.2f} ms")
    print(f"Maximum theoretical steps/sec: {1/avg_step_time:.1f}")
    print(f"Actual steps/sec: {steps_completed/np.mean(component_times['total']):.1f}")
    print(f"Efficiency: {(steps_completed/np.mean(component_times['total']))/(1/avg_step_time)*100:.1f}%")
    
    return component_times


def save_profiling_results(metrics, detailed_times):
    """Save profiling results to file"""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': 64,
            'hidden_size': 512,
            'grid_size': 11,
            'max_steps': 100
        },
        'summary_metrics': metrics,
        'detailed_times': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in detailed_times.items()}
    }
    
    os.makedirs('profiling_results', exist_ok=True)
    filename = f'profiling_results/full_scale_profile_{int(time.time())}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProfiling results saved to: {filename}")
    return filename


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)
    
    print("=" * 80)
    print("FULL-SCALE TRAINING PROFILER (Batch Size: 64)")
    print("=" * 80)
    
    print(f"\nSystem information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    
    print(f"\nMemory information:")
    print(f"  System RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if torch.cuda.is_available():
        print(f"  GPU RAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Create trainer with full scale configuration
    print("\nCreating trainer with batch size 64...")
    trainer = create_full_scale_trainer()
    
    # Warmup
    print("Warming up (3 collections)...")
    for _ in range(3):
        _ = trainer._collect_experiences_parallel()
    
    # Simulate full training epoch
    epoch_metrics = simulate_training_epoch(trainer)
    
    # Detailed collection profiling
    detailed_times = profile_collection_detailed(trainer)
    
    # Analyze bottlenecks
    analyze_bottlenecks(epoch_metrics)
    
    # Save results
    results_file = save_profiling_results(epoch_metrics, detailed_times)
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    
    # Summary
    total_epoch_time = epoch_metrics['total_epoch_time']
    steps_per_sec = epoch_metrics['collection']['avg_steps_per_sec']
    
    print(f"\nEstimated training performance with batch size 64:")
    print(f"  Collection time per epoch: {epoch_metrics['collection']['avg_time']:.2f}s")
    print(f"  Training time per epoch: {epoch_metrics['training']['avg_time']:.2f}s")
    print(f"  Total epoch time: {total_epoch_time:.2f}s")
    print(f"  Estimated epochs per hour: {3600/total_epoch_time:.0f}")
    print(f"  Estimated environment steps per hour: {steps_per_sec * 3600:,.0f}")
    print(f"  Estimated training time for 10k epochs: {10000 * total_epoch_time / 3600:.1f} hours")
    
    print(f"\nFor comparison with actual training:")
    print("  Run: python run.py train --batch-size 64 --epochs 100 --save-dir test_profile")
    print("  Then compare the timing with these profiling results.")