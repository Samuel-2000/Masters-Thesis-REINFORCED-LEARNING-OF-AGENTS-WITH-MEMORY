# run.py
#!/usr/bin/env python3
import sys
from pathlib import Path
from parser import parse_args
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.agent import Agent
from src.core.env_factory import EnvironmentFactory
from src.core.human_agent import HumanAgent
from src.core.utils import get_model_name_from_path
from src.evaluation.benchmark import Benchmark
from src.evaluation.visualization import Visualizer
from src.training.trainer import Trainer, generate_plots_from_metrics   # MODIFIED: import generate_plots_from_metrics
from src.core.constants import (
    DEFAULT_GRID_SIZE, DEFAULT_MAX_STEPS, DEFAULT_OBSTACLE_FRACTION,
    DEFAULT_FOOD_SOURCES, DEFAULT_FOOD_ENERGY, DEFAULT_INITIAL_ENERGY,
    DEFAULT_ENERGY_DECAY, DEFAULT_ENERGY_PER_STEP, DEFAULT_RENDER_SIZE,
    DEFAULT_DOOR_OPEN_DURATION, DEFAULT_DOOR_CLOSE_DURATION,
    DEFAULT_HIDDEN_SIZE, DEFAULT_GAMMA, DEFAULT_ENTROPY_COEF,
    DEFAULT_MAX_GRAD_NORM, DEFAULT_SAVE_INTERVAL, DEFAULT_TEST_INTERVAL,
)


def main():
    args = parse_args()
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmarks").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/videos").mkdir(parents=True, exist_ok=True)


    env_config = {
        "grid_size": DEFAULT_GRID_SIZE,
        "max_steps": DEFAULT_MAX_STEPS,
        "obstacle_fraction": DEFAULT_OBSTACLE_FRACTION,
        "n_food_sources": DEFAULT_FOOD_SOURCES,
        "food_energy": DEFAULT_FOOD_ENERGY,
        "initial_energy": DEFAULT_INITIAL_ENERGY,
        "energy_decay": DEFAULT_ENERGY_DECAY,
        "energy_per_step": DEFAULT_ENERGY_PER_STEP,
        "render_size": DEFAULT_RENDER_SIZE,
        "task_class": args.task_class,
        "complexity_level": args.complexity_level,
        "n_doors": args.n_doors,
        "door_open_duration": DEFAULT_DOOR_OPEN_DURATION,
        "door_close_duration": DEFAULT_DOOR_CLOSE_DURATION,
        "n_buttons_per_door": args.n_buttons_per_door,
        "button_break_probability": args.button_break_probability
    }

    if args.command == "train":
        # Build configuration dictionary directly from arguments
        config = {
            "experiment": {
                "name": args.experiment_name or f"{args.network_type}_{args.batch_size}b_{args.lr}lr",
                "save_dir": args.save_dir,
                "seed": args.seed,
                "resume": args.resume,          # ADDED
            },
            "environment": env_config,
            "model": {
                "type": args.network_type,
                "hidden_size": DEFAULT_HIDDEN_SIZE,
                "use_auxiliary": args.auxiliary_tasks,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "gamma": DEFAULT_GAMMA,
                "entropy_coef": DEFAULT_ENTROPY_COEF,
                "max_grad_norm": DEFAULT_MAX_GRAD_NORM,
                "save_interval": DEFAULT_SAVE_INTERVAL,
                "test_interval": DEFAULT_TEST_INTERVAL,
                "dynamic_complexity": args.dynamic_complexity,
                "performance_window": args.performance_window,
                "complexity_increase_threshold": args.complexity_increase_threshold,
                "complexity_decrease_threshold": args.complexity_decrease_threshold,
                "complexity_step": args.complexity_step,
                "min_complexity": args.min_complexity,
                "max_complexity": args.max_complexity,
                "adjustment_interval": args.adjustment_interval,
                "stagnation_switch_interval": args.stagnation_switch_interval,
                "stagnation_termination": args.stagnation_termination,
                "min_basic_complexity": args.min_basic_complexity,
                "curriculum_stages": args.curriculum_stages,
                "auxiliary_tasks": args.auxiliary_tasks
            },
        }
        trainer = Trainer(config)
        trainer.train()

    elif args.command == "test":
        if args.play:
            # Human play mode
            agent = HumanAgent()
        else:
            agent = Agent.load(args.model)
            base_name = get_model_name_from_path(args.model)


        # Determine test configurations
        if args.dynamic_complexity:
            test_configs = []
            for stage in args.stages:
                for comp in args.complexities:
                    test_configs.append(env_config.copy())
                    test_configs[-1]["task_class"] = stage
                    test_configs[-1]["complexity_level"] = comp
        else:
            test_configs = [env_config]


        for idx, cfg in enumerate(test_configs):
            print(f"\nTest {idx+1}/{len(test_configs)}: {cfg['task_class']} comp={cfg['complexity_level']:.2f}")
            env = EnvironmentFactory.create_from_config(cfg, test_mode=True)
            if args.play:
                results = agent.test(env, args.episodes)
            else:
                model_name = f"{base_name}_{cfg['task_class']}_comp_{cfg['complexity_level']:.2f}".replace('.','_')
                results = agent.test(env, args.episodes, args.visualize, args.save_video, model_name)
            print(f"  Reward: {results['avg_reward']:.2f}, Success: {results['success_rate']:.1f}%")

    elif args.command == "plot":          # ADDED BLOCK
        import numpy as np
        metrics_path = Path("logs/metrics") / f"{args.experiment_name}_metrics.npz"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        data = np.load(metrics_path, allow_pickle=True)
        metrics = {key: data[key] for key in data.files}
        # Convert numeric task class back to strings if stored as numbers
        if 'task_class_history' in metrics and metrics['task_class_history'].dtype.kind in 'iuf':
            stage_map = {0.0: 'basic', 0.33: 'doors', 0.66: 'buttons', 1.0: 'complex'}
            metrics['task_class_history'] = [stage_map.get(v, 'unknown') for v in metrics['task_class_history']]
        generate_plots_from_metrics(metrics, args.experiment_name, args.output_dir)
        print(f"Plots saved to {args.output_dir}/{args.experiment_name}/")

        """
        elif args.command == "benchmark":
            env_config = {
                "grid_size": 11,
                "max_steps": 100,
                "obstacle_fraction": 0.25,
                "n_food_sources": 4,
                "food_energy": 10.0,
                "initial_energy": 30.0,
                "energy_decay": 0.98,
                "energy_per_step": 0.1,
                "render_size": 0,
                "task_class": args.task_class,
                "complexity_level": args.complexity_level,
                "n_doors": args.n_doors,
                "door_open_duration": 10,
                "door_close_duration": 20,
                "n_buttons_per_door": args.n_buttons_per_door,
                "button_break_probability": args.button_break_probability,
            }
            benchmark = Benchmark(models_dir=args.models_dir, output_dir=args.output_dir)
            benchmark.run(episodes_per_model=args.benchmark_episodes, env_config=env_config, verbose=True)

        elif args.command == "visualize":
            env_config = {
                "grid_size": 11,
                "max_steps": 100,
                "obstacle_fraction": 0.25,
                "n_food_sources": 4,
                "food_energy": 10.0,
                "initial_energy": 30.0,
                "energy_decay": 0.98,
                "energy_per_step": 0.1,
                "render_size": 512,
                "task_class": args.task_class,
                "complexity_level": args.complexity_level,
                "n_doors": args.n_doors,
                "door_open_duration": 10,
                "door_close_duration": 20,
                "n_buttons_per_door": args.n_buttons_per_door,
                "button_break_probability": args.button_break_probability,
            }
            visualizer = Visualizer(model_path=args.model)
            visualizer.run(episodes=args.episodes, env_config=env_config,
                        save_video=args.save_video, save_gif=args.save_gif)

        elif args.command == "compare":
            from experiments.compare import run_comparison
            run_comparison(
                architectures=args.architectures,
                epochs=args.epochs,
                trials=args.trials,
                output_dir=args.output_dir,
                task_class=args.task_class,
                complexity_level=args.complexity_level
            )
        """
    else:
        print("Please specify a command. Use --help for usage.")


if __name__ == "__main__":
    main()