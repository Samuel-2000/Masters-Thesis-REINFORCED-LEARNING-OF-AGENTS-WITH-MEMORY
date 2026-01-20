#!/usr/bin/env python3
"""
test_dynamic_complexity.py
Test script to verify dynamic complexity adjustment works
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import ComplexityManager


def test_complexity_manager():
    """Test the ComplexityManager class"""
    print("=" * 80)
    print("Testing ComplexityManager")
    print("=" * 80)
    
    # Create test config
    config = {
        'environment': {
            'task_class': 'basic',
            'complexity_level': 0.0
        },
        'training': {
            'dynamic_complexity': True,
            'performance_window': 10,
            'complexity_increase_threshold': 0.95,
            'complexity_decrease_threshold': 0.7,
            'complexity_step': 0.1,
            'min_complexity': 0.0,
            'max_complexity': 1.0,
            'adjustment_interval': 5,
            'curriculum_stages': ["basic", "doors", "buttons", "complex"]
        }
    }
    
    # Create manager
    manager = ComplexityManager(config)
    
    print(f"Initial state:")
    print(f"  Enabled: {manager.enabled}")
    print(f"  Current stage: {manager.get_current_task_class()}")
    print(f"  Current complexity: {manager.get_current_complexity():.2f}")
    
    # Test performance tracking
    print(f"\nSimulating performance...")
    for i in range(15):
        # Simulate improving performance
        reward = 0.1 + i * 0.1
        manager.add_performance(reward, i)
        
        # Try to adjust every 5 epochs
        adjustment = manager.adjust_complexity(i)
        if adjustment:
            print(f"Epoch {i}: {adjustment['action']}")
            print(f"  Stage: {adjustment['old_stage']} -> {adjustment['new_stage']}")
            print(f"  Complexity: {adjustment['old_complexity']:.2f} -> {adjustment['new_complexity']:.2f}")
    
    # Get final status
    status = manager.get_status()
    print(f"\nFinal status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    return True


# Update the test_simple_training function:

def test_simple_training():
    """Test a simple training run with dynamic complexity"""
    print("\n" + "=" * 80)
    print("Testing Simple Training with Dynamic Complexity")
    print("=" * 80)
    
    # Create a simple config
    config = {
        'experiment': {
            'name': 'test_dynamic',
            'seed': 42,
            'use_wandb': False
        },
        'environment': {
            'grid_size': 7,
            'max_steps': 20,
            'obstacle_fraction': 0.2,
            'n_food_sources': 2,
            'food_energy': 10.0,
            'initial_energy': 30.0,
            'energy_decay': 0.98,
            'energy_per_step': 0.1,
            'task_class': 'basic',
            'complexity_level': 0.0
        },
        'model': {
            'type': 'lstm',
            'hidden_size': 128,
            'use_auxiliary': False
        },
        'training': {
            'epochs': 50,
            'batch_size': 8,
            'learning_rate': 0.001,
            'gamma': 0.97,
            'entropy_coef': 0.01,
            'max_grad_norm': 1.0,
            'save_interval': 1000,
            'test_interval': 25,
            'use_auxiliary': False,
            'dynamic_complexity': True,
            'performance_window': 10,
            'complexity_increase_threshold': 0.95,
            'complexity_decrease_threshold': 0.7,
            'complexity_step': 0.2,
            'min_complexity': 0.0,
            'max_complexity': 1.0,
            'adjustment_interval': 10,
            'curriculum_stages': ["basic", "doors"]
        }
    }
    
    try:
        from src.training.trainer import Trainer
        print(f"\nCreating trainer with dynamic complexity...")
        
        # Create trainer (but don't actually train in this test)
        trainer = Trainer(config)
        
        print(f"Trainer created successfully!")
        print(f"Experiment name: {trainer.experiment_name}")
        print(f"Dynamic complexity enabled: {trainer.complexity_manager.enabled}")
        print(f"Initial complexity: {trainer.complexity_manager.get_current_complexity():.2f}")
        print(f"Initial task class: {trainer.complexity_manager.get_current_task_class()}")
        
        # Test a few simulated epochs
        print(f"\nSimulating a few epochs...")
        for i in range(5):
            # Simulate collecting experiences (simplified)
            print(f"Epoch {i}: Simulating training...")
            
            # Simulate performance
            simulated_reward = 0.2 + i * 0.15
            trainer.complexity_manager.add_performance(simulated_reward, i)
            
            # Check for adjustment
            adjustment = trainer.complexity_manager.adjust_complexity(i)
            if adjustment:
                print(f"  Complexity adjustment: {adjustment['action']}")
                print(f"  New complexity: {adjustment['new_complexity']:.2f}")
        
        print(f"\nFinal complexity status:")
        status = trainer.complexity_manager.get_status()
        for key, value in status.items():
            if key not in ['enabled', 'total_stages']:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that backward compatibility is maintained"""
    print("\n" + "=" * 80)
    print("Testing Backward Compatibility")
    print("=" * 80)
    
    # Config without dynamic complexity (old style)
    old_config = {
        'environment': {
            'grid_size': 11,
            'max_steps': 100,
            'obstacle_fraction': 0.25
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.0005
            # No dynamic_complexity key
        }
    }
    
    try:
        manager = ComplexityManager(old_config)
        print(f"ComplexityManager created with old config")
        print(f"Dynamic complexity enabled: {manager.enabled} (should be False)")
        print(f"Current task class: {manager.get_current_task_class()} (should be 'basic')")
        print(f"Current complexity: {manager.get_current_complexity()} (should be 0.0)")
        
        # Test that it doesn't adjust
        adjustment = manager.adjust_complexity(100)
        print(f"Adjustment at epoch 100: {adjustment} (should be None)")
        
        return manager.enabled == False and adjustment is None
        
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("Testing Dynamic Complexity Implementation")
    print("=" * 80)
    
    tests_passed = 0
    tests_total = 3
    
    # Run tests
    try:
        if test_complexity_manager():
            tests_passed += 1
            print("✓ ComplexityManager test passed")
    except Exception as e:
        print(f"✗ ComplexityManager test failed: {e}")
    
    try:
        if test_simple_training():
            tests_passed += 1
            print("✓ Simple training test passed")
    except Exception as e:
        print(f"✗ Simple training test failed: {e}")
    
    try:
        if test_backward_compatibility():
            tests_passed += 1
            print("✓ Backward compatibility test passed")
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
    
    print(f"\n{'='*80}")
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("All tests passed! Dynamic complexity implementation is working.")
        print("\nTo run a full training with dynamic complexity:")
        print("1. Create a config file with dynamic_complexity: true")
        print("2. Run: python run.py train --config your_config.yaml")
        print("\nOr use command line (basic example):")
        print("python run.py train --network-type lstm --epochs 5000 --batch-size 64 \\")
        print("  --dynamic-complexity --performance-window 100 --adjustment-interval 500")
    else:
        print("Some tests failed. Please check the implementation.")
    
    print(f"{'='*80}")