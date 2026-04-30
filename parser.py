# parser.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory Maze RL Experiments – Explicit parameters.",
        epilog="""
        Examples:
            # Train with dynamic complexity
            python run.py train --network-type lstm --epochs 10000 --batch-size 64 --lr 0.0005 [--auxiliary-tasks] --dynamic-complexity [--curriculum-stages basic doors buttons complex --performance-window 100 --adjustment-interval 500 --complexity-increase-threshold 0.95 --complexity-decrease-threshold 0.7 --complexity-step 0.05 --min-complexity 0.0 --max-complexity 1.0]

            # Train without dynamic complexity (static environment)
            python run.py train --network-type lstm --epochs 10000 --batch-size 64 --lr 0.0005 [--auxiliary-tasks] --task-class doors --complexity-level 0.7 [--n-doors 5]
            python run.py train --network-type lstm --epochs 10000 --batch-size 64 --lr 0.0005 [--auxiliary-tasks] --task-class buttons [--n-doors 5 --n-buttons-per-door 4 --button-break-probability 0.0]
            

            
            # Test a model statically
            python run.py test --model models/lstm_best.pt --episodes 10 [--visualize] --task-class doors --complexity-level 0.7 [--n-doors 5]

            # dynamic complexity test across stages and complexities
            python run.py test --model models/lstm_best.pt --episodes 5 --dynamic-complexity [--stages basic doors buttons --complexities 0.0 0.5 1.0]

            # Human play mode
            python run.py test --play --episodes 4 --task-class complex --complexity-level 0.5
            python run.py test --play --episodes 1 --dynamic-complexity [--stages basic doors buttons --complexities 0.0 0.5 1.0]

            # Plot saved metrics
            python run.py plot --experiment-name lstm_64b_0.0005lr_2026-04-28_11-35-00

            # Resume training
            python run.py train --network-type lstm --epochs 10000 --batch-size 64 --lr 0.0005 --resume models/lstm_16b_0.0005lr_2026-04-28_11-35-00_best_checkpoint.pt


        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # ---------- train command ----------
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--network-type", required=True, choices=["lstm", "transformer", "multimemory"])
    train_parser.add_argument("--epochs", required=True, type=int)
    train_parser.add_argument("--batch-size", required=True, type=int)
    train_parser.add_argument("--lr", required=True, type=float)
    train_parser.add_argument("--optimizer", type=str, default=None)
    train_parser.add_argument("--weight-decay", type=float, default=None)

    
    train_parser.add_argument("--auxiliary-tasks", action="store_true", default=False)

    train_parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    train_parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment name")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file (.pt) to resume training")


    # Dynamic complexity parameters (required only if --dynamic-complexity is set)
    train_parser.add_argument("--performance-window", type=int, default=None)
    train_parser.add_argument("--complexity-increase-threshold", type=float, default=None)
    train_parser.add_argument("--complexity-decrease-threshold", type=float, default=None)
    train_parser.add_argument("--complexity-step", type=float, default=None)
    train_parser.add_argument("--min-complexity", type=float, default=None)
    train_parser.add_argument("--max-complexity", type=float, default=None)
    train_parser.add_argument("--adjustment-interval", type=int, default=None)
    train_parser.add_argument("--stagnation-switch-interval", type=int, default=None)
    train_parser.add_argument("--stagnation-termination", type=int, default=None)
    train_parser.add_argument("--min-basic-complexity", type=float, default=None)
    train_parser.add_argument("--curriculum-stages", nargs="+", default=None,
                              choices=["basic","doors","buttons","complex"])
    # ---------- train command ----------

    # ---------- test command ----------
    test_parser = subparsers.add_parser("test", help="Test a trained model or play manually")

    test_parser.add_argument("--model", type=str, default=None, help="Path to trained model (required unless --play)")
    test_parser.add_argument("--play", action="store_true", help="Human play mode (no model needed)")
    test_parser.add_argument("--episodes", required=True, type=int)
    test_parser.add_argument("--visualize", action="store_true", default=False)
    test_parser.add_argument("--save-video", action="store_true", default=False)
    # ---------- test command ----------

    for general_parser in [train_parser, test_parser]:
        general_parser.add_argument("--dynamic-complexity", action="store_true", help="Test across all stages and complexities")
        
        # Optional filters for dynamic testing
        general_parser.add_argument("--complexities", nargs="+", type=float, default=[0.0,0.25,0.5,0.75,1.0])
        general_parser.add_argument("--stages", nargs="+", choices=["basic","doors","buttons","complex"],
                                default=["basic","doors","buttons","complex"])

        general_parser.add_argument("--task-class", type=str, choices=["basic","doors","buttons","complex"], default=None)
        general_parser.add_argument("--complexity-level", type=float, default=None)
        general_parser.add_argument("--n-doors", type=int, default=None)
        general_parser.add_argument("--n-buttons-per-door", type=int, choices=[0,1,2,3,4], default=None)
        general_parser.add_argument("--button-break-probability", type=float, default=None)

        general_parser.add_argument("--seed", type=int, default=42, help="Random seed")


    plot_parser = subparsers.add_parser("plot", help="Plot training metrics from saved data")
    plot_parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment (matches folder in logs/metrics)")
    plot_parser.add_argument("--output-dir", type=str, default="results/plots", help="Directory to save generated plots")


    """
    # ---------- benchmark command ----------
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark models")
    bench_parser.add_argument("--models-dir", required=True, type=str)
    bench_parser.add_argument("--benchmark-episodes", required=True, type=int)
    bench_parser.add_argument("--output-dir", required=True, type=str)
    bench_parser.add_argument("--task-class", required=True, choices=["basic","doors","buttons","complex"])
    bench_parser.add_argument("--complexity-level", required=True, type=float)
    bench_parser.add_argument("--n-doors", required=True, type=int)
    bench_parser.add_argument("--n-buttons-per-door", required=True, type=int, choices=[0,1,2,3,4])
    bench_parser.add_argument("--door-periodic", action="store_true")
    bench_parser.add_argument("--button-break-probability", required=True, type=float)

    # ---------- visualize command ----------
    viz_parser = subparsers.add_parser("visualize", help="Visualize model performance")
    viz_parser.add_argument("--model", required=True, type=str)
    viz_parser.add_argument("--episodes", required=True, type=int)
    viz_parser.add_argument("--save-video", action="store_true")
    viz_parser.add_argument("--save-gif", action="store_true")
    viz_parser.add_argument("--task-class", required=True, choices=["basic","doors","buttons","complex"])
    viz_parser.add_argument("--complexity-level", required=True, type=float)
    viz_parser.add_argument("--n-doors", required=True, type=int)
    viz_parser.add_argument("--n-buttons-per-door", required=True, type=int, choices=[0,1,2,3,4])
    viz_parser.add_argument("--door-periodic", action="store_true")
    viz_parser.add_argument("--button-break-probability", required=True, type=float)

    # ---------- compare command ----------
    compare_parser = subparsers.add_parser("compare", help="Compare architectures")
    compare_parser.add_argument("--architectures", nargs="+", required=True,
                                choices=["lstm","transformer","multimemory"])
    compare_parser.add_argument("--epochs", required=True, type=int)
    compare_parser.add_argument("--trials", required=True, type=int)
    compare_parser.add_argument("--output-dir", required=True, type=str)
    compare_parser.add_argument("--task-class", required=True, choices=["basic","doors","buttons","complex"])
    compare_parser.add_argument("--complexity-level", required=True, type=float)

    """

    args = parser.parse_args()

    # ---------- Post‑processing validation ----------

    required_env = []

    if not args.resume:

        if bool(args.dynamic_complexity) == bool(args.task_class):
            raise "either use dynamic_complexity or choose task_class"
        
        if bool(args.dynamic_complexity) == (args.complexity_level is not None):
            raise "either use dynamic_complexity or choose complexity_level"
        
        if bool(args.dynamic_complexity) and bool(any([args.n_doors, args.n_buttons_per_door, args.button_break_probability])):
            raise "either use dynamic_complexity or choose door and button parameters"


    elif args.command == "test":
        if bool(args.play) == bool(args.model):
            raise "either play as human or choose model"
        
    if args.task_class:
        required_env.extend(["complexity_level"])
       


    missing = [p for p in required_env if getattr(args, p) is None]
    if missing:
        parser.error(f"Additional parameters are required: {', '.join(missing)}")



    defaults = {}

    if args.dynamic_complexity:
        defaults.update({
            "performance_window": 10,
            "complexity_increase_threshold": 0.6,
            "complexity_decrease_threshold": 0.4,
            "complexity_step": 0.05,
            "min_complexity": 0.0,
            "max_complexity": 1.0,
            "adjustment_interval": 100,
            "stagnation_switch_interval": 500,
            "stagnation_termination": 2000,
            "min_basic_complexity": 0.2,
            "curriculum_stages": ["basic", "doors", "buttons", "complex"],
        })

    if args.command == "train":
        defaults.update({
            "optimizer": "adam",
            "weight_decay": 0.0
        })
        if args.auxiliary_tasks:
            print(f"Info: auxiliary tasks enabled")

    # Iterate over each argument and assign default if None
    for arg_name, default_value in defaults.items():
        if getattr(args, arg_name, None) is None:
            setattr(args, arg_name, default_value)
            print(f"Warning: --{arg_name.replace('_', '-')} was None, setting to default {default_value}")


    # For benchmark and visualize, we keep required=True for environment parameters as before.

    return args