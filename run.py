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
            print(f"  Increase threshold: {config['training'].get('complexity_increase_threshold', 0.7)}")
            print(f"  Decrease threshold: {config['training'].get('complexity_decrease_threshold', 0.3)}")
            print(f"  Complexity step: {config['training'].get('complexity_step', 0.05)}")
            print(f"  Adjustment interval: {config['training'].get('adjustment_interval', 500)}")
            print(f"  Curriculum stages: {config['training'].get('curriculum_stages', ['basic', 'doors', 'buttons', 'complex'])}")
        
        print(f"{'='*80}\n")
        
        # Create trainer and train
        trainer = Trainer(config)
        trainer.train()
        
    elif args.command == "test":
        from src.core.agent import Agent
        
        # Extract model name for video naming
        model_name = get_model_name_from_path(args.model)
        
        # Load agent
        print(f"Loading agent from {args.model}...")
        agent = Agent.load(args.model)
        
        # Create environment using factory
        print(f"\nCreating environment with task class: {args.task_class}")
        print(f"Complexity level: {args.complexity_level:.2f}")
        
        env = EnvironmentFactory.create_from_args(args, test_mode=True)
        
        # Print door/button info if applicable
        if args.task_class != 'basic':
            print(f"Number of doors: {args.n_doors}")
            print(f"Buttons per door: {args.n_buttons_per_door}")
            if args.button_break_probability > 0:
                print(f"Button break probability: {args.button_break_probability:.2f}")
            if args.door_periodic:
                print("Doors: Periodic (open/close automatically)")
            else:
                print("Doors: Require button press")
        
        # Run test
        print(f"\nTesting agent for {args.episodes} episodes...")
        test_results = agent.test(
            env=env,
            episodes=args.episodes,
            visualize=args.visualize,
            save_video=args.save_video,
            model_name=model_name
        )
        
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        print(f"Task class: {args.task_class}")
        print(f"Complexity level: {args.complexity_level:.2f}")
        print(f"Average Reward: {test_results['avg_reward']:.2f}")
        print(f"Success Rate: {test_results['success_rate']:.1f}%")
        print(f"Average Steps: {test_results['avg_steps']:.1f}")
        print(f"Std Reward: {test_results['std_reward']:.2f}")
        print(f"{'='*60}")
        
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