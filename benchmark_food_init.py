import numpy as np
import timeit
from src.core.environment import GridMazeWorld

# Configuration for testing (use a typical setup)
env_params = {
    'grid_size': 11,
    'max_steps': 100,
    'obstacle_fraction': 0.25,
    'n_food_sources': 4,
    'food_energy': 10.0,
    'initial_energy': 30.0,
    'energy_decay': 0.98,
    'energy_per_step': 0.1,
    'render_size': 0,
    'task_class': 'basic',      # or 'doors', 'buttons', 'complex'
    'complexity_level': 0.5,
    'n_doors': 0,
    'door_open_duration': 10,
    'door_close_duration': 20,
    'n_buttons_per_door': 0,
    'button_break_probability': 0.0
}

def benchmark_init_food(env, repeats):
    """Time the _init_food_sources method over multiple resets."""
    # Warm‑up
    for _ in range(10):
        env.reset()
    # Actual timing
    start = timeit.default_timer()
    for _ in range(repeats):
        env.reset() # _init_food_sources()   # each reset calls _init_food_sources internally
    elapsed = timeit.default_timer() - start
    return elapsed / repeats, elapsed

if __name__ == "__main__":
    env = GridMazeWorld(**env_params)
    repeats = 1000
    avg_time, total = benchmark_init_food(env, repeats)
    print(f"Average _init_food_sources time: {avg_time*repeats:.3f} ms")
    print(f"Total for {repeats} resets: {total:.3f} s")