import numpy as np
from src.core.environment import GridMazeWorld

# Test 1: Check observation range
print("="*60)
print("TEST 1: Observation Token Range")
print("="*60)

env = GridMazeWorld(task_class="basic", complexity_level=0.0)
obs, info = env.reset()

print(f"Initial observation: {obs}")
print(f"Observation shape: {obs.shape}")
print(f"Min token: {obs.min()}, Max token: {obs.max()}")
print(f"Vocabulary size expected: 20")
print(f"All tokens in range 0-19? {obs.min() >= 0 and obs.max() < 20}")

# Test 2: Check all possible tokens
print("\n" + "="*60)
print("TEST 2: Token Distribution")
print("="*60)

tokens_seen = set()
for step in range(100):
    action = np.random.randint(0, 6)
    obs, reward, terminated, truncated, info = env.step(action)
    tokens_seen.update(obs)
    
    if terminated or truncated:
        obs, info = env.reset()

print(f"Unique tokens seen: {sorted(tokens_seen)}")
print(f"Missing tokens in 0-19: {set(range(20)) - tokens_seen}")

# Test 3: Check specific positions
print("\n" + "="*60)
print("TEST 3: Position Analysis")
print("="*60)

env.reset()
for i in range(10):
    action = i % 6
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: Action={action}, Obs[8]={obs[8]}, Obs[9]={obs[9]}")
    if terminated or truncated:
        env.reset()

# Test 4: Energy token range
print("\n" + "="*60)
print("TEST 4: Energy Token Range")
print("="*60)

env.reset()
for energy in [0, 10, 30, 50, 70, 90, 100]:
    # Hack to set energy for testing
    env.energy = energy
    obs = env._get_observation()
    print(f"Energy={energy:3.0f} -> Token={obs[9]}")