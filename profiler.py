#!/usr/bin/env python3
"""
profiler.py - Detailed profiling of training with cProfile and torch.profiler
python profiler.py --network-type lstm --batch-size 64 --task-class doors --complexity-level 0.7 --n-doors 5 --n-buttons-per-door 4
"""

import torch
import numpy as np
import cProfile
import pstats
import io
from pathlib import Path
import sys
import os
import json
import time
import argparse
from pstats import SortKey

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import AdaptiveParallelTrainer
from src.core.utils import seed_everything


class DetailedProfiler:
    """Detailed profiler using cProfile and torch.profiler"""
    
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create config based on args
        self.config = self._create_config()
        
        # Results storage
        self.results = {}
        
        print("=" * 100)
        print("DETAILED PROFILER")
        print("=" * 100)
        print(f"Configuration:")
        print(f"  Network type: {args.network_type}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Task class: {args.task_class}")
        print(f"  Complexity level: {args.complexity_level}")
        print(f"  Number of doors: {args.n_doors}")
        print(f"  Buttons per door: {args.n_buttons_per_door}")
        print(f"  Device: {self.device}")
        print("=" * 100)
    
    def _create_config(self):
        """Create training configuration from args"""
        config = {
            'experiment': {
                'name': f'profile_{self.args.task_class}_comp{self.args.complexity_level}',
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
                'energy_per_step': 0.1,
                'task_class': self.args.task_class,
                'complexity_level': self.args.complexity_level,
                'n_doors': self.args.n_doors,
                'door_open_duration': 10,
                'door_close_duration': 20,
                'n_buttons_per_door': self.args.n_buttons_per_door,
                'button_break_probability': 0.0,
            },
            'model': {
                'type': self.args.network_type,
                'hidden_size': 512,
                'use_auxiliary': False
            },
            'training': {
                'epochs': 10,  # Only profile a few epochs
                'batch_size': self.args.batch_size,
                'learning_rate': 0.0005,
                'gamma': 0.97,
                'entropy_coef': 0.01,
                'max_grad_norm': 1.0,
                'save_interval': 1000,
                'test_interval': 500,
                'optimizer': 'adam',
                'dynamic_complexity': False
            }
        }
        return config
    
    def run_cprofile_analysis(self, trainer):
        """Run cProfile analysis on the training process"""
        print("\n" + "=" * 100)
        print("cProfile ANALYSIS - Function-level timing")
        print("=" * 100)
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Run training with profiler
        profiler.enable()
        
        # Run a few training steps manually so we can profile specific functions
        self._run_training_steps(trainer, num_steps=5)
        
        profiler.disable()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
        
        print("\nTop 50 functions by cumulative time:")
        print("-" * 100)
        ps.print_stats(50)
        
        # Save to file
        self._save_cprofile_results(profiler, "cumulative")
        
        # Also analyze by internal time
        print("\n" + "=" * 100)
        print("Top 50 functions by internal time:")
        print("-" * 100)
        ps.sort_stats(SortKey.TIME)
        ps.print_stats(50)
        self._save_cprofile_results(profiler, "internal")
        
        # Save raw stats
        self._save_raw_stats(ps)
        
        return ps
    
    def _run_training_steps(self, trainer, num_steps=5):
        """Run specific training steps for profiling"""
        print(f"\nRunning {num_steps} training steps for profiling...")
        
        for step in range(num_steps):
            print(f"  Step {step + 1}/{num_steps}")
            
            # Reset for each step
            trainer.vector_env.reset()
            trainer.agent.reset()
            
            # Collect experiences
            experiences = trainer._collect_experiences_parallel()
            
            # Training step
            trainer._train_step(experiences)
            
            # Flush cache if using multi-memory
            if hasattr(trainer.agent.network, 'flush_cache_buffer'):
                trainer.agent.network.flush_cache_buffer()
    
    def run_pytorch_profiler(self, trainer):
        """Run PyTorch profiler for CUDA/GPU analysis"""
        if not torch.cuda.is_available():
            print("\n⚠️  CUDA not available. Skipping PyTorch profiler.")
            return None
        
        print("\n" + "=" * 100)
        print("PYTORCH PROFILER - GPU/CUDA analysis")
        print("=" * 100)
        
        # Create PyTorch profiler
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,  # Skip first iteration
                warmup=1,  # Warmup for next iteration
                active=3,  # Profile 3 iterations
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f'./profiler_traces/{self.args.task_class}_doors{self.args.n_doors}'
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        )
        
        profiler.start()
        
        # Run training steps with profiler
        for step in range(5):  # Run 5 steps total
            print(f"  Profiler step {step + 1}/5")
            
            # Reset for each step
            trainer.vector_env.reset()
            trainer.agent.reset()
            
            # Collect experiences
            with torch.profiler.record_function("experience_collection"):
                experiences = trainer._collect_experiences_parallel()
            
            # Training step
            with torch.profiler.record_function("training_step"):
                trainer._train_step(experiences)
            
            # Step the profiler
            profiler.step()
            
            # Flush cache
            if hasattr(trainer.agent.network, 'flush_cache_buffer'):
                trainer.agent.network.flush_cache_buffer()
        
        profiler.stop()
        
        # Print profiler results
        self._analyze_pytorch_profiler(profiler)
        
        return profiler
    
    def _analyze_pytorch_profiler(self, profiler):
        """Analyze and display PyTorch profiler results"""
        print("\nPyTorch Profiler Summary:")
        print("-" * 100)
        
        # Get key metrics
        try:
            # CPU/GPU time breakdown
            print("\nTime Spent (ms):")
            print(f"{'Operation':<30} {'CPU Total':>12} {'CPU %':>8} {'CUDA Total':>12} {'CUDA %':>8}")
            print("-" * 80)
            
            key_averages = profiler.key_averages()
            
            # Group by operator
            op_times = {}
            for item in key_averages:
                op_name = item.key
                cpu_time = item.cpu_time_total / 1000  # Convert to ms
                cuda_time = item.cuda_time_total / 1000 if item.cuda_time_total else 0
                
                if op_name not in op_times:
                    op_times[op_name] = {'cpu': 0, 'cuda': 0, 'count': 0}
                
                op_times[op_name]['cpu'] += cpu_time
                op_times[op_name]['cuda'] += cuda_time
                op_times[op_name]['count'] += item.count
            
            # Sort by total time
            sorted_ops = sorted(
                op_times.items(),
                key=lambda x: x[1]['cpu'] + x[1]['cuda'],
                reverse=True
            )
            
            total_cpu = sum(v['cpu'] for v in op_times.values())
            total_cuda = sum(v['cuda'] for v in op_times.values())
            
            for op_name, times in sorted_ops[:20]:  # Top 20
                cpu_pct = (times['cpu'] / total_cpu * 100) if total_cpu > 0 else 0
                cuda_pct = (times['cuda'] / total_cuda * 100) if total_cuda > 0 else 0
                print(f"{op_name[:30]:<30} {times['cpu']:>12.2f} {cpu_pct:>7.1f}% {times['cuda']:>12.2f} {cuda_pct:>7.1f}%")
            
            print("-" * 80)
            print(f"{'TOTAL':<30} {total_cpu:>12.2f} {'100.0':>8}% {total_cuda:>12.2f} {'100.0':>8}%")
            
            # Memory usage
            print(f"\nMemory Usage (MB):")
            print(f"{'Operation':<30} {'CPU Mem':>12} {'CUDA Mem':>12}")
            print("-" * 80)
            
            for item in key_averages:
                if hasattr(item, 'cpu_memory_usage') and item.cpu_memory_usage > 0:
                    cpu_mb = item.cpu_memory_usage / 1024**2
                    cuda_mb = item.cuda_memory_usage / 1024**2 if item.cuda_memory_usage else 0
                    if cpu_mb > 10 or cuda_mb > 10:  # Only show significant usage
                        print(f"{item.key[:30]:<30} {cpu_mb:>12.2f} {cuda_mb:>12.2f}")
            
            # FLOPs (if available)
            print(f"\nFLOPS Analysis:")
            total_flops = sum(item.flops for item in key_averages if hasattr(item, 'flops'))
            if total_flops > 0:
                print(f"Total FLOPS: {total_flops:.2e}")
                print(f"FLOPS per training step: {total_flops / 5:.2e}")
            
        except Exception as e:
            print(f"Error analyzing PyTorch profiler: {e}")
    
    def profile_specific_functions(self, trainer):
        """Profile specific functions with detailed timing"""
        print("\n" + "=" * 100)
        print("DETAILED FUNCTION PROFILING")
        print("=" * 100)
        
        # Reset environments
        trainer.vector_env.reset()
        trainer.agent.reset()
        
        # Profile individual components
        function_timings = {}
        
        # 1. Environment reset
        print("\n1. Environment Reset Timing:")
        reset_times = []
        for i in range(5):
            start = time.perf_counter()
            trainer.vector_env.reset()
            reset_time = time.perf_counter() - start
            reset_times.append(reset_time)
            print(f"  Reset {i+1}: {reset_time*1000:.2f} ms")
        
        function_timings['environment_reset'] = {
            'mean_ms': np.mean(reset_times) * 1000,
            'std_ms': np.std(reset_times) * 1000,
            'min_ms': np.min(reset_times) * 1000,
            'max_ms': np.max(reset_times) * 1000
        }
        
        # 2. Experience collection
        print("\n2. Experience Collection Timing:")
        coll_times = []
        for i in range(5):
            start = time.perf_counter()
            experiences = trainer._collect_experiences_parallel()
            coll_time = time.perf_counter() - start
            coll_times.append(coll_time)
            print(f"  Collection {i+1}: {coll_time:.3f} s")
            
            # Store experience stats
            if i == 0:
                B, T, K = experiences['observations'].shape
                total_steps = B * T
                print(f"    Batch size: {B}, Sequence length: {T}, Total steps: {total_steps}")
                print(f"    Steps per second: {total_steps/coll_time:.1f}")
        
        function_timings['experience_collection'] = {
            'mean_s': np.mean(coll_times),
            'std_s': np.std(coll_times),
            'steps_per_second': total_steps / np.mean(coll_times) if coll_times else 0
        }
        
        # 3. Training step
        print("\n3. Training Step Timing:")
        train_times = []
        for i in range(5):
            experiences = trainer._collect_experiences_parallel()  # Get fresh experiences
            start = time.perf_counter()
            trainer._train_step(experiences)
            train_time = time.perf_counter() - start
            train_times.append(train_time)
            print(f"  Training {i+1}: {train_time:.3f} s")
        
        function_timings['training_step'] = {
            'mean_s': np.mean(train_times),
            'std_s': np.std(train_times)
        }
        
        # 4. Network forward/backward
        print("\n4. Network Forward/Backward Timing:")
        if hasattr(trainer.agent.network, '_forward_lstm'):
            # Profile LSTM separately
            forward_times = []
            for i in range(10):
                # Create dummy input
                dummy_input = torch.randint(0, 19, (self.args.batch_size, 1, 10), device=trainer.device)
                trainer.agent.network.eval()
                
                start = time.perf_counter()
                with torch.no_grad():
                    _ = trainer.agent.network(dummy_input)
                forward_time = time.perf_counter() - start
                forward_times.append(forward_time)
            
            function_timings['network_forward'] = {
                'mean_ms': np.mean(forward_times) * 1000,
                'std_ms': np.std(forward_times) * 1000
            }
            print(f"  Forward pass: {np.mean(forward_times)*1000:.2f} ± {np.std(forward_times)*1000:.2f} ms")
        
        # 5. Door/button logic (environment complexity)
        print("\n5. Environment Complexity Impact:")
        env = trainer.vector_env.envs[0]
        print(f"  Grid size: {env.grid_size}")
        print(f"  Doors: {len(env.doors)}")
        print(f"  Buttons: {len(env.buttons)}")
        
        # Time door updates
        if env.doors:
            door_update_times = []
            for i in range(100):
                start = time.perf_counter()
                env._update_door_states()
                door_update_times.append(time.perf_counter() - start)
            
            function_timings['door_updates'] = {
                'mean_ms': np.mean(door_update_times) * 1000 * 1000,  # Convert to microseconds
                'std_ms': np.std(door_update_times) * 1000 * 1000
            }
            print(f"  Door updates: {np.mean(door_update_times)*1e6:.1f} ± {np.std(door_update_times)*1e6:.1f} µs")
        
        # 6. Template matching
        print("\n6. Template Matching Performance:")
        if hasattr(env, 'template_tree'):
            template_times = []
            grid = env.grid
            
            # Test template matching at random positions
            for i in range(100):
                y = np.random.randint(1, env.grid_size - 1)
                x = np.random.randint(1, env.grid_size - 1)
                
                start = time.perf_counter()
                env.template_tree.matches_any_template(grid, y, x)
                template_times.append(time.perf_counter() - start)
            
            function_timings['template_matching'] = {
                'mean_ms': np.mean(template_times) * 1000 * 1000,  # Convert to microseconds
                'std_ms': np.std(template_times) * 1000 * 1000
            }
            print(f"  Template matching: {np.mean(template_times)*1e6:.1f} ± {np.std(template_times)*1e6:.1f} µs per check")
        
        # Save detailed timings
        self._save_detailed_timings(function_timings)
        
        return function_timings
    
    def analyze_bottlenecks(self, cprofile_stats, function_timings, pytorch_profiler=None):
        """Analyze and identify bottlenecks"""
        print("\n" + "=" * 100)
        print("BOTTLENECK ANALYSIS")
        print("=" * 100)
        
        # Get timing breakdown
        total_epoch_time = function_timings.get('experience_collection', {}).get('mean_s', 0) + \
                          function_timings.get('training_step', {}).get('mean_s', 0)
        
        if total_epoch_time > 0:
            coll_percentage = function_timings['experience_collection']['mean_s'] / total_epoch_time * 100
            train_percentage = function_timings['training_step']['mean_s'] / total_epoch_time * 100
            
            print(f"\nTime breakdown per epoch:")
            print(f"  Experience Collection: {function_timings['experience_collection']['mean_s']:.3f}s ({coll_percentage:.1f}%)")
            print(f"  Training Step:         {function_timings['training_step']['mean_s']:.3f}s ({train_percentage:.1f}%)")
            print(f"  Total epoch time:      {total_epoch_time:.3f}s")
        
        # Identify bottlenecks
        bottleneck_threshold = 0.6  # 60% of total time
        
        if coll_percentage > bottleneck_threshold * 100:
            print(f"\n🚨 MAJOR BOTTLENECK: Experience Collection ({coll_percentage:.1f}% of time)")
            print("   Most time-consuming functions in collection:")
            self._print_top_functions_by_pattern(cprofile_stats, ['collect', 'env', 'step', 'reset', 'observation'])
            print("\n   Optimization suggestions:")
            print("   1. Reduce batch size (current: {})".format(self.args.batch_size))
            print("   2. Simplify environment complexity")
            print("   3. Optimize door/button logic")
            print("   4. Use JIT compilation for environment")
        
        if train_percentage > bottleneck_threshold * 100:
            print(f"\n🚨 MAJOR BOTTLENECK: Training Step ({train_percentage:.1f}% of time)")
            print("   Most time-consuming functions in training:")
            self._print_top_functions_by_pattern(cprofile_stats, ['train', 'loss', 'backward', 'optimizer', 'network'])
            print("\n   Optimization suggestions:")
            print("   1. Reduce model size (hidden_size)")
            print("   2. Use mixed precision (FP16) training")
            print("   3. Increase batch size (current: {})".format(self.args.batch_size))
            print("   4. Use gradient accumulation")
        
        # GPU utilization analysis
        if pytorch_profiler:
            print(f"\nGPU Utilization Analysis:")
            # This would require parsing PyTorch profiler output
            print("   Check TensorBoard traces for detailed GPU analysis:")
            print(f"   tensorboard --logdir=./profiler_traces/{self.args.task_class}_doors{self.args.n_doors}")
        
        # Memory analysis
        print(f"\nMemory Analysis:")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory allocated: {allocated:.2f} GB")
            print(f"  GPU Memory reserved:  {reserved:.2f} GB")
        
        # Environment complexity impact
        print(f"\nEnvironment Complexity Impact:")
        print(f"  Task class: {self.args.task_class}")
        print(f"  Complexity level: {self.args.complexity_level}")
        print(f"  Doors: {self.args.n_doors}")
        print(f"  Buttons per door: {self.args.n_buttons_per_door}")
        
        if 'door_updates' in function_timings:
            print(f"  Door update overhead: {function_timings['door_updates']['mean_ms']:.1f} µs per update")
        
        if 'template_matching' in function_timings:
            print(f"  Template matching overhead: {function_timings['template_matching']['mean_ms']:.1f} µs per check")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if coll_percentage > 70:
            print(f"1. PRIMARY: Focus on optimizing environment/collection")
            print(f"   - Reduce batch size from {self.args.batch_size} to 32")
            print(f"   - Simplify environment: reduce doors from {self.args.n_doors} to 3")
        elif train_percentage > 70:
            print(f"1. PRIMARY: Focus on optimizing training")
            print(f"   - Reduce model hidden_size from 512 to 256")
            print(f"   - Enable mixed precision training")
        else:
            print(f"1. SYSTEM: Well balanced, consider overall optimization")
        
        print(f"2. Next steps for actual training:")
        print(f"   Expected time for 10,000 epochs: {(total_epoch_time * 10000) / 3600:.1f} hours")
    
    def _print_top_functions_by_pattern(self, stats, patterns):
        """Print top functions matching patterns"""
        for func in stats.stats.keys():
            func_name = func[2]  # Function name is at index 2
            for pattern in patterns:
                if pattern in func_name.lower():
                    total_time = stats.stats[func][3]  # Cumulative time at index 3
                    print(f"    {func_name}: {total_time:.3f}s")
                    break
    
    def _save_cprofile_results(self, profiler, sort_method):
        """Save cProfile results to file"""
        results_dir = Path("profiler_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.args.task_class}_doors{self.args.n_doors}_cprofile_{sort_method}_{timestamp}"
        
        # Save stats
        stats_file = results_dir / f"{filename}.stats"
        profiler.dump_stats(str(stats_file))
        
        # Save human-readable output
        txt_file = results_dir / f"{filename}.txt"
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        
        if sort_method == "cumulative":
            ps.sort_stats(SortKey.CUMULATIVE)
        else:
            ps.sort_stats(SortKey.TIME)
        
        ps.print_stats(100)  # Top 100 functions
        
        with open(txt_file, 'w') as f:
            f.write(s.getvalue())
        
        print(f"\nSaved cProfile results to:")
        print(f"  Stats file: {stats_file}")
        print(f"  Text file: {txt_file}")
        
        # Can be viewed with: python -m pstats filename.stats
    
    def _save_raw_stats(self, ps):
        """Save raw profiling statistics"""
        results_dir = Path("profiler_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.args.task_class}_doors{self.args.n_doors}_raw_stats_{timestamp}.json"
        
        raw_data = []
        for func, (cc, nc, tt, ct, callers) in ps.stats.items():
            raw_data.append({
                'file': func[0],
                'line': func[1],
                'function': func[2],
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct
            })
        
        # Sort by cumulative time
        raw_data.sort(key=lambda x: x['cumulative_time'], reverse=True)
        
        with open(results_dir / filename, 'w') as f:
            json.dump(raw_data[:100], f, indent=2)  # Top 100 functions
        
        print(f"  Raw stats: {results_dir / filename}")
    
    def _save_detailed_timings(self, timings):
        """Save detailed timing results"""
        results_dir = Path("profiler_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.args.task_class}_doors{self.args.n_doors}_detailed_timings_{timestamp}.json"
        
        with open(results_dir / filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'args': vars(self.args),
                'timings': timings,
                'system_info': {
                    'pytorch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'device': str(self.device),
                    'cpu_count': os.cpu_count()
                }
            }, f, indent=2)
        
        print(f"\nSaved detailed timings to: {results_dir / filename}")
    
    def run(self):
        """Run complete profiling"""
        try:
            # Set seed for reproducibility
            seed_everything(42)
            
            # Create trainer
            print("\nCreating trainer...")
            trainer = AdaptiveParallelTrainer(self.config)
            
            # 1. Run cProfile analysis
            cprofile_stats = self.run_cprofile_analysis(trainer)
            
            # 2. Run PyTorch profiler (GPU analysis)
            pytorch_profiler = self.run_pytorch_profiler(trainer)
            
            # 3. Run detailed function profiling
            function_timings = self.profile_specific_functions(trainer)
            
            # 4. Analyze bottlenecks
            self.analyze_bottlenecks(cprofile_stats, function_timings, pytorch_profiler)
            
            print("\n" + "=" * 100)
            print("PROFILING COMPLETE")
            print("=" * 100)
            
            # Summary of output files
            print("\nGenerated files:")
            print("1. cProfile stats (.stats) - view with: python -m pstats <filename.stats>")
            print("2. Text summaries (.txt) - human-readable function timings")
            print("3. Raw stats (.json) - JSON format for analysis")
            print("4. Detailed timings (.json) - component-level timing breakdown")
            print("5. PyTorch traces - view with TensorBoard")
            print("\nTo view TensorBoard traces:")
            print(f"  tensorboard --logdir=./profiler_traces/{self.args.task_class}_doors{self.args.n_doors}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Profiling failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Detailed profiling of training')
    parser.add_argument('--network-type', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'multimemory'],
                       help='Network architecture')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--task-class', type=str, default='doors',
                       choices=['basic', 'doors', 'buttons', 'complex'],
                       help='Task class')
    parser.add_argument('--complexity-level', type=float, default=0.7,
                       help='Complexity level (0.0 to 1.0)')
    parser.add_argument('--n-doors', type=int, default=5,
                       help='Number of doors in environment')
    parser.add_argument('--n-buttons-per-door', type=int, default=4,
                       choices=[0, 1, 2, 3, 4],
                       help='Buttons per door')
    parser.add_argument('--door-periodic', action='store_true',
                       help='Doors open/close periodically')
    parser.add_argument('--button-break-probability', type=float, default=0.0,
                       help='Probability that a button breaks when pressed')
    
    args = parser.parse_args()
    
    # Print system information
    print("\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    print(f"  CPU cores: {os.cpu_count()}")
    
    # Create and run profiler
    profiler = DetailedProfiler(args)
    success = profiler.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())