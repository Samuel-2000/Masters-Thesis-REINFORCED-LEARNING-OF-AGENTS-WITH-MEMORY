#run.py
#!/usr/bin/env python3
"""
Unified Runner for Memory Maze RL Experiments with Dynamic Complexity
"""

import sys
from pathlib import Path
from parser import parse_args

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.evaluation.benchmark import Benchmark
from src.core.utils import load_config, get_model_name_from_path
from src.core.env_factory import EnvironmentFactory

from datetime import datetime


def main():
    args = parse_args()
    
    # Setup directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmarks").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/videos").mkdir(parents=True, exist_ok=True)
    
    if args.command == "train":
        from src.training.trainer import Trainer
        
        # Load config
        config = load_config(args.config)
        
        # Override config with command line args
        if args.network_type:
            config["model"]["type"] = args.network_type
        if args.auxiliary_tasks:
            config["training"]["auxiliary_tasks"] = True
            config["model"]["use_auxiliary"] = True
        if args.epochs:
            config["training"]["epochs"] = args.epochs
        if args.batch_size:
            config["training"]["batch_size"] = args.batch_size
        if args.lr:
            config["training"]["learning_rate"] = args.lr
        
        # Override environment config with complexity parameters
        if hasattr(args, 'task_class'):
            config["environment"]["task_class"] = args.task_class
        
        if hasattr(args, 'complexity_level'):
            config["environment"]["complexity_level"] = args.complexity_level
        
        if hasattr(args, 'n_doors'):
            config["environment"]["n_doors"] = args.n_doors
        
        if hasattr(args, 'n_buttons_per_door'):
            config["environment"]["n_buttons_per_door"] = args.n_buttons_per_door
        
        if hasattr(args, 'door_periodic'):
            config["environment"]["door_periodic"] = args.door_periodic
        
        if hasattr(args, 'button_break_probability'):
            config["environment"]["button_break_probability"] = args.button_break_probability
        
        # Override dynamic complexity parameters
        if hasattr(args, 'dynamic_complexity'):
            config["training"]["dynamic_complexity"] = args.dynamic_complexity
        
        if hasattr(args, 'performance_window'):
            config["training"]["performance_window"] = args.performance_window
        
        if hasattr(args, 'complexity_increase_threshold'):
            config["training"]["complexity_increase_threshold"] = args.complexity_increase_threshold
        
        if hasattr(args, 'complexity_decrease_threshold'):
            config["training"]["complexity_decrease_threshold"] = args.complexity_decrease_threshold
        
        if hasattr(args, 'complexity_step'):
            config["training"]["complexity_step"] = args.complexity_step
        
        if hasattr(args, 'min_complexity'):
            config["training"]["min_complexity"] = args.min_complexity
        
        if hasattr(args, 'max_complexity'):
            config["training"]["max_complexity"] = args.max_complexity
        
        if hasattr(args, 'adjustment_interval'):
            config["training"]["adjustment_interval"] = args.adjustment_interval
        
        if hasattr(args, 'curriculum_stages'):
            config["training"]["curriculum_stages"] = args.curriculum_stages
        
        # Print environment configuration
        print(f"\n{'='*80}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*80}")
        print(f"Model: {config['model']['type'].upper()}")
        print(f"Epochs: {config['training']['epochs']:,}")
        print(f"Batch size: {config['training']['batch_size']}")
        print(f"Learning rate: {config['training']['learning_rate']}")
        print(f"Auxiliary tasks: {'Yes' if config['training'].get('auxiliary_tasks', False) else 'No'}")
        
        print(f"\nEnvironment:")
        print(f"  Task class: {config['environment'].get('task_class', 'basic')}")
        print(f"  Complexity level: {config['environment'].get('complexity_level', 0.0):.2f}")
        
        if config['environment'].get('task_class', 'basic') != 'basic':
            print(f"  Number of doors: {config['environment'].get('n_doors', 0)}")
            print(f"  Buttons per door: {config['environment'].get('n_buttons_per_door', 0)}")
            if config['environment'].get('button_break_probability', 0.0) > 0:
                print(f"  Button break probability: {config['environment'].get('button_break_probability', 0.0):.2f}")
            if config['environment'].get('door_periodic', False):
                print("  Doors: Periodic (open/close automatically)")
            else:
                print("  Doors: Require button press")
        
        print(f"\nDynamic Complexity:")
        dynamic_enabled = config['training'].get('dynamic_complexity', False)
        print(f"  Enabled: {'Yes' if dynamic_enabled else 'No'}")
        
        if dynamic_enabled:
            print(f"  Performance window: {config['training'].get('performance_window', 100)}")
            print(f"  Increase threshold: {config['training'].get('complexity_increase_threshold', 0.95)}")
            print(f"  Decrease threshold: {config['training'].get('complexity_decrease_threshold', 0.7)}")
            print(f"  Complexity step: {config['training'].get('complexity_step', 0.05)}")
            print(f"  Adjustment interval: {config['training'].get('adjustment_interval', 500)}")
            print(f"  Curriculum stages: {config['training'].get('curriculum_stages', ['basic', 'doors', 'buttons', 'complex'])}")
        
        print(f"{'='*80}\n")
        
        # Create trainer and train
        trainer = Trainer(config)
        trainer.train()
        
    elif args.command == "test":
        from src.core.agent import Agent
        
        # Extract model name for video naming (without .pt)
        base_model_name = get_model_name_from_path(args.model)
        
        # Load agent
        print(f"Loading agent from {args.model}...")
        agent = Agent.load(args.model)
        
        # Determine what to test
        if args.dynamic:
            # Test all combinations: stages × complexities
            test_configs = []
            for stage in args.stages:
                for complexity in args.complexities:
                    test_configs.append({
                        'task_class': stage,
                        'complexity_level': complexity,
                        'n_doors': -1,  # Means "use task class default"
                        'n_buttons_per_door': -1,  # Means "use task class default"
                        'door_periodic': None,  # Means "use task class default"
                        'button_break_probability': -1.0  # Means "use task class default"
                    })
        elif args.test_all_stages:
            # Test all stages with single complexity
            test_configs = []
            for stage in args.stages:
                test_configs.append({
                    'task_class': stage,
                    'complexity_level': args.complexity_level,
                    'n_doors': args.n_doors,
                    'n_buttons_per_door': args.n_buttons_per_door,
                    'door_periodic': args.door_periodic,
                    'button_break_probability': args.button_break_probability
                })
        else:
            # Single test with specified/default values (default: basic, 0.0)
            test_configs = [{
                'task_class': args.task_class,
                'complexity_level': args.complexity_level,
                'n_doors': args.n_doors,
                'n_buttons_per_door': args.n_buttons_per_door,
                'door_periodic': args.door_periodic,
                'button_break_probability': args.button_break_probability
            }]
        
        # Run tests for each configuration
        all_results = []


        
        for config_idx, config in enumerate(test_configs):
            print(f"\n{'='*60}")
            print(f"Test {config_idx+1}/{len(test_configs)}")
            print(f"{'='*60}")
            #print(f"Task class: {config['task_class']}")
            #print(f"Complexity level: {config['complexity_level']:.2f}")

            print(f"DEBUG - Creating env with params:")
            print(f"  task_class: {config['task_class']}")
            print(f"  complexity_level: {config['complexity_level']}")
            print(f"  n_doors: {config['n_doors']}")
            print(f"  n_buttons_per_door: {config['n_buttons_per_door']}")
            print(f"  door_periodic: {config['door_periodic']}")

            
            # Create environment
            env = EnvironmentFactory.create_from_config({
                'grid_size': 11,
                'max_steps': 100,
                'obstacle_fraction': 0.25,
                'n_food_sources': 4,
                'food_energy': 10.0,
                'initial_energy': 30.0,
                'energy_decay': 0.98,
                'energy_per_step': 0.1,
                'render_size': 512 if args.visualize or args.save_video else 0,
                'task_class': config['task_class'],
                'complexity_level': config['complexity_level'],
                'n_doors': config['n_doors'],
                'n_buttons_per_door': config['n_buttons_per_door'],
                'door_periodic': config['door_periodic'],
                'button_break_probability': config['button_break_probability']
            }, test_mode=True)

            print(f"DEBUG - Env created with:")
            print(f"  env.task_class: {env.task_class}")
            print(f"  env.complexity_level: {env.complexity_level}")
            print(f"  env.n_doors: {env.n_doors}")
            print(f"  env.door_periodic: {env.door_periodic}")

            #if config['task_class'] != 'basic':
            #    print(f"Number of doors: {env.n_doors}")
            #    print(f"  Doors periodic: {env.door_periodic}")
            #    print(f"  Buttons per door: {env.n_buttons_per_door}")
            #    if env.button_break_probability > 0:
            #        print(f"    Button break probability: {env.button_break_probability:.2f}")
            
            # Create model name for video naming
            if len(test_configs) > 1:
                # Always include stage and complexity for multiple tests
                complexity_str = f"{config['complexity_level']:.2f}".replace('.', '_')
                model_name = f"{base_model_name}_{config['task_class']}_comp_{complexity_str}"
            else:
                # For single test, include stage and complexity only if not basic/0.0
                if config['task_class'] != 'basic' or config['complexity_level'] != 0.0:
                    complexity_str = f"{config['complexity_level']:.2f}".replace('.', '_')
                    model_name = f"{base_model_name}_{config['task_class']}_comp_{complexity_str}"
                else:
                    model_name = base_model_name
            
            # Run test
            test_results = agent.test(
                env=env,
                episodes=args.episodes,
                visualize=args.visualize,
                save_video=args.save_video,
                model_name=model_name
            )
            
            # Add configuration info to results
            test_results['config'] = config
            all_results.append(test_results)
            
            print(f"\nResults:")
            print(f"  Average Reward: {test_results['avg_reward']:.2f}")
            print(f"  Success Rate: {test_results['success_rate']:.1f}%")
            print(f"  Average Steps: {test_results['avg_steps']:.1f}")
            print(f"  Std Reward: {test_results['std_reward']:.2f}")
        
        # Print summary if multiple tests
        if len(all_results) > 1:
            print(f"\n{'='*80}")
            print("TESTING SUMMARY")
            print(f"{'='*80}")
            print(f"{'Task Class':<10} {'Complexity':<12} {'Avg Reward':<12} {'Success':<10} {'Avg Steps':<10}")
            print("-" * 70)
            
            for result in all_results:
                config = result['config']
                print(f"{config['task_class']:<10} "
                    f"{config['complexity_level']:<12.2f} "
                    f"{result['avg_reward']:<12.2f} "
                    f"{result['success_rate']:<10.1f}% "
                    f"{result['avg_steps']:<10.1f}")
            
            # Find best configuration
            best_result = max(all_results, key=lambda x: x['avg_reward'])
            best_config = best_result['config']
            print(f"\nBest configuration:")
            print(f"  Task class: {best_config['task_class']}")
            print(f"  Complexity: {best_config['complexity_level']:.2f}")
            print(f"  Average Reward: {best_result['avg_reward']:.2f}")
            print(f"  Success Rate: {best_result['success_rate']:.1f}%")
            
            # Save summary to file
            summary_path = f'results/test_summary_{base_model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            with open(summary_path, 'w') as f:
                f.write("Testing Summary\n")
                f.write("="*50 + "\n")
                f.write(f"Model: {base_model_name}\n")
                f.write(f"Total tests: {len(all_results)}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for result in all_results:
                    config = result['config']
                    f.write(f"Configuration:\n")
                    f.write(f"  Task class: {config['task_class']}\n")
                    f.write(f"  Complexity: {config['complexity_level']:.2f}\n")
                    if config['task_class'] != 'basic':
                        f.write(f"  Doors: {config['n_doors']}, Buttons per door: {config['n_buttons_per_door']}\n")
                    f.write(f"Results:\n")
                    f.write(f"  Avg Reward: {result['avg_reward']:.2f}\n")
                    f.write(f"  Success Rate: {result['success_rate']:.1f}%\n")
                    f.write(f"  Avg Steps: {result['avg_steps']:.1f}\n")
                    f.write(f"  Std Reward: {result['std_reward']:.2f}\n")
                    f.write("-"*40 + "\n")
            
            print(f"\nSummary saved to: {summary_path}")
        
    elif args.command == "benchmark":
        print(f"Running benchmark on models in {args.models_dir}...")
        
        # Create environment config using factory
        env_config = EnvironmentFactory.create_config_for_benchmark(args)
        
        print(f"\nBenchmark configuration:")
        print(f"  Task class: {env_config.get('task_class', 'basic')}")
        print(f"  Complexity level: {env_config.get('complexity_level', 0.0):.2f}")
        if env_config.get('task_class', 'basic') != 'basic':
            print(f"  Number of doors: {env_config.get('n_doors', 0)}")
            print(f"  Buttons per door: {env_config.get('n_buttons_per_door', 0)}")
        print(f"  Episodes per model: {args.benchmark_episodes}")
        
        benchmark = Benchmark(
            models_dir=args.models_dir,
            output_dir=args.output_dir
        )
        
        # Run benchmark with custom environment config
        results = benchmark.run(
            episodes_per_model=args.benchmark_episodes,
            env_config=env_config,
            verbose=True
        )
        
        if results is not None and not results.empty:
            # Show summary
            print(f"\n{'='*60}")
            print("BENCHMARK SUMMARY")
            print(f"{'='*60}")
            print(f"Task class: {env_config.get('task_class', 'basic')}")
            print(f"Complexity level: {env_config.get('complexity_level', 0.0):.2f}")
            
            # Print top models
            print("\nTop 5 models by average reward:")
            print(f"{'Model':<30} {'Reward':<10} {'Success':<10} {'Steps':<10}")
            print("-" * 70)
            
            for idx, row in results.head(5).iterrows():
                print(f"{row['model_file']:<30} "
                      f"{row['avg_reward']:>9.2f} "
                      f"{row['success_rate']:>9.1f}% "
                      f"{row['avg_steps']:>9.1f}")
        
    elif args.command == "visualize":
        from src.evaluation.visualization import Visualizer
        
        # Extract model name
        model_name = get_model_name_from_path(args.model)
        
        print(f"Loading model from {args.model}...")
        visualizer = Visualizer(args.model)
        
        # Create environment config for visualization
        env_config = EnvironmentFactory.create_config_for_benchmark(args)
        env_config['render_size'] = 512  # Enable rendering for visualization
        
        print(f"\nVisualization configuration:")
        print(f"  Task class: {args.task_class}")
        print(f"  Complexity level: {args.complexity_level:.2f}")
        if args.task_class != 'basic':
            print(f"  Number of doors: {args.n_doors}")
            if hasattr(args, 'door_periodic') and args.door_periodic:
                print("  Doors: Periodic")
        
        # Run visualization
        visualizer.run(
            episodes=args.episodes,
            env_config=env_config,
            save_video=args.save_video,
            save_gif=args.save_gif
        )
        
    elif args.command == "compare":
        from experiments.compare import run_comparison
        
        print(f"Comparing architectures: {args.architectures}")
        print(f"Task class: {args.task_class}")
        print(f"Complexity level: {args.complexity_level:.2f}")
        
        # Note: You'll need to update run_comparison to accept env parameters
        print("\nNote: The compare function needs to be updated to support")
        print("complexity parameters. Using basic environment for now.")
        
        run_comparison(
            architectures=args.architectures,
            epochs=args.epochs,
            trials=args.trials,
            output_dir=args.output_dir
        )
        
    else:
        print("Please specify a command. Use --help for usage information.")


if __name__ == "__main__":
    main()