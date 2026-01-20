#parser.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory Maze RL Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python run.py train --config configs/transformer.yaml
            python run.py train --network-type transformer --auxiliary-tasks
            python run.py test --model models/transformer_best.pt
            python run.py benchmark --benchmark-episodes 50
            python run.py visualize --model models/transformer_best.pt --save-video
            python run.py train --task-class complex --complexity-level 0.5
            python run.py test --model models/lstm_best.pt --task-class doors --complexity-level 0.7
            python run.py train --dynamic-complexity --performance-window 100 --adjustment-interval 500
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", 
                        type=str, 
                        default="configs/default.yaml",
                        help="Path to config file"
                    )
    train_parser.add_argument("--network-type", 
                        choices=["lstm", "transformer", "multimemory"],
                        default="lstm", 
                        help="Network architecture"
                    )
    train_parser.add_argument("--auxiliary-tasks", 
                        action="store_true",
                        help="Use auxiliary tasks"
                    )
    train_parser.add_argument("--epochs", 
                        type=int, 
                        default=10000,
                        help="Training epochs"
                    )
    train_parser.add_argument("--batch-size", 
                        type=int, 
                        default=64,
                        help="Batch size"
                    )
    train_parser.add_argument("--lr", 
                        type=float, 
                        default=0.0005,
                        help="Learning rate"
                    )
    train_parser.add_argument("--save-dir", 
                        type=str, 
                        default="models",
                        help="Directory to save models"
                    )
    train_parser.add_argument("--experiment-name", 
                        type=str, 
                        default=None,
                        help="Experiment name for logging"
                    )
    
    # Environment complexity parameters for train
    train_parser.add_argument("--task-class",
                        type=str,
                        choices=["basic", "doors", "buttons", "complex"],
                        default="basic",
                        help="Task class: basic, doors, buttons, complex"
                    )
    train_parser.add_argument("--complexity-level",
                        type=float,
                        default=0.0,
                        help="Complexity level (0.0 to 1.0)"
                    )
    train_parser.add_argument("--n-doors",
                        type=int,
                        default=0,
                        help="Number of doors in environment"
                    )
    train_parser.add_argument("--n-buttons-per-door",
                        type=int,
                        default=1,
                        choices=[0, 1, 2, 3, 4],
                        help="Buttons per door"
                    )
    train_parser.add_argument("--door-periodic",
                        action="store_true",
                        help="Doors open/close periodically"
                    )
    train_parser.add_argument("--button-break-probability",
                        type=float,
                        default=0.0,
                        help="Probability that a button breaks when pressed (0.0 to 1.0)"
                    )
    

    # NEW: Dynamic complexity parameters for train
    train_parser.add_argument("--dynamic-complexity",
                        action="store_true",
                        help="Enable dynamic complexity adjustment based on performance"
                    )
    train_parser.add_argument("--performance-window",
                        type=int,
                        default=100,
                        help="Number of episodes to consider for performance evaluation"
                    )
    train_parser.add_argument("--complexity-increase-threshold",
                        type=float,
                        default=0.95,
                        help="Performance threshold to increase complexity (0.0 to 1.0)"
                    )
    train_parser.add_argument("--complexity-decrease-threshold",
                        type=float,
                        default=0.7,
                        help="Performance threshold to decrease complexity (0.0 to 1.0)"
                    )
    train_parser.add_argument("--complexity-step",
                        type=float,
                        default=0.05,
                        help="Step size for complexity adjustment"
                    )
    train_parser.add_argument("--min-complexity",
                        type=float,
                        default=0.0,
                        help="Minimum complexity level"
                    )
    train_parser.add_argument("--max-complexity",
                        type=float,
                        default=1.0,
                        help="Maximum complexity level"
                    )
    train_parser.add_argument("--adjustment-interval",
                        type=int,
                        default=500,
                        help="Epochs between complexity adjustments"
                    )
    train_parser.add_argument("--curriculum-stages",
                        nargs="+",
                        default=["basic", "doors", "buttons", "complex"],
                        choices=["basic", "doors", "buttons", "complex"],
                        help="Curriculum stages in order"
                    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument("--model", 
                        type=str, 
                        # Make required=False and we'll check manually
                        required=False, 
                        help="Path to trained model (not needed for --play mode)"
                    )
    test_parser.add_argument("--episodes", 
                        type=int, 
                        default=10, 
                        help="Number of test episodes"
                    )
    test_parser.add_argument("--visualize", 
                        action="store_true", 
                        default=True, 
                        help="Show visualization"
                    )
    test_parser.add_argument("--save-video", 
                        action="store_true", 
                        help="Save test video",
                        default=True
                    )
    
    # Environment complexity parameters for test
    test_parser.add_argument("--task-class",
                        type=str,
                        choices=["basic", "doors", "buttons", "complex"],
                        default="basic",
                        help="Task class: basic, doors, buttons, complex"
                    )
    test_parser.add_argument("--complexity-level",
                        type=float,
                        default=0.0,
                        help="Complexity level (0.0 to 1.0)"
                    )
    test_parser.add_argument("--n-doors",
                        type=int,
                        default=0,
                        help="Number of doors in environment"
                    )
    test_parser.add_argument("--n-buttons-per-door",
                        type=int,
                        default=1,
                        choices=[0, 1, 2, 3, 4],
                        help="Buttons per door"
                    )
    test_parser.add_argument("--door-periodic",
                        action="store_true",
                        help="Doors open/close periodically"
                    )
    test_parser.add_argument("--button-break-probability",
                        type=float,
                        default=0.0,
                        help="Probability that a button breaks when pressed (0.0 to 1.0)"
                    )
    
    test_parser.add_argument('--dynamic',
                        action='store_true',
                        help='Test across all task classes and complexity levels'
                    )

    test_parser.add_argument('--test-all-stages',
                        action='store_true',
                        help='Test all task classes with current complexity level'
                    )

    test_parser.add_argument('--complexities',
                            nargs='+',
                            type=float,
                            default=[0.0, 0.25, 0.5, 0.75, 1.0],
                            help='Complexity levels to test (space-separated)'
                    )

    test_parser.add_argument('--stages',
                        nargs='+',
                        choices=["basic", "doors", "buttons", "complex"],
                        default=["basic", "doors", "buttons", "complex"],
                        help='Task classes to test (space-separated)'
                    )

    test_parser.add_argument("--play",
                        action="store_true",
                        help="Human play mode - control agent with keyboard instead of using a model"
                    )


    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark models")
    bench_parser.add_argument("--models-dir", 
                        type=str, 
                        default="models",
                        help="Directory containing models"
                    )
    bench_parser.add_argument("--benchmark-episodes", 
                        type=int, 
                        default=20,
                        help="Episodes per model"
                    )
    bench_parser.add_argument("--output-dir", 
                        type=str, 
                        default="results/benchmarks",
                        help="Output directory for results"
                    )
    
    # Environment complexity parameters for benchmark
    bench_parser.add_argument("--task-class",
                        type=str,
                        choices=["basic", "doors", "buttons", "complex"],
                        default="basic",
                        help="Task class for benchmarking"
                    )
    bench_parser.add_argument("--complexity-level",
                        type=float,
                        default=0.0,
                        help="Complexity level (0.0 to 1.0)"
                    )
    bench_parser.add_argument("--n-doors",
                        type=int,
                        default=0,
                        help="Number of doors in environment"
                    )
    bench_parser.add_argument("--n-buttons-per-door",
                        type=int,
                        default=1,
                        choices=[0, 1, 2, 3, 4],
                        help="Buttons per door"
                    )
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize model performance")
    viz_parser.add_argument("--model", 
                        type=str, 
                        required=True,
                        help="Path to model"
                    )
    viz_parser.add_argument("--episodes", 
                        type=int, 
                        default=3,
                        help="Number of episodes to visualize"
                    )
    viz_parser.add_argument("--save-video", 
                        action="store_true",
                        help="Save visualization as video"
                    )
    viz_parser.add_argument("--save-gif", 
                        action="store_true",
                        help="Save as GIF"
                    )
    
    # Environment complexity parameters for visualize
    viz_parser.add_argument("--task-class",
                        type=str,
                        choices=["basic", "doors", "buttons", "complex"],
                        default="basic",
                        help="Task class for visualization"
                    )
    viz_parser.add_argument("--complexity-level",
                        type=float,
                        default=0.0,
                        help="Complexity level (0.0 to 1.0)"
                    )
    viz_parser.add_argument("--n-doors",
                        type=int,
                        default=0,
                        help="Number of doors in environment"
                    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare architectures")
    compare_parser.add_argument("--architectures", 
                        nargs="+",
                        default=["lstm", "transformer", "multimemory"],
                        help="Architectures to compare"
                    )
    compare_parser.add_argument("--epochs", 
                        type=int, 
                        default=5000,
                        help="Training epochs per architecture"
                    )
    compare_parser.add_argument("--trials", 
                        type=int, 
                        default=3,
                        help="Number of trials per architecture"
                    )
    compare_parser.add_argument("--output-dir", 
                        type=str, 
                        default="results/comparisons",
                        help="Output directory"
                    )
    
    # Environment complexity parameters for compare
    compare_parser.add_argument("--task-class",
                        type=str,
                        choices=["basic", "doors", "buttons", "complex"],
                        default="basic",
                        help="Task class for comparison"
                    )
    compare_parser.add_argument("--complexity-level",
                        type=float,
                        default=0.0,
                        help="Complexity level (0.0 to 1.0)"
                    )
    
    return parser.parse_args()