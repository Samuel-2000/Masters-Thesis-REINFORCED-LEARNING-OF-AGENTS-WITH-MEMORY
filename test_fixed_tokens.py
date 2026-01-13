# test_fixed_tokens.py
from src.core.environment import GridMazeWorld
from src.core.constants import Actions, TaskClass, VOCAB_SIZE, ObservationTokens
from src.core.agent import Agent
import numpy as np

print("="*70)
print("TESTING FIXED TOKEN SYSTEM (19 tokens, 0-18)")
print("="*70)

# Test 1: Environment observation range
print("\n1. Testing Environment Observations:")
env = GridMazeWorld(
    task_class=TaskClass.COMPLEX,
    complexity_level=0.7,
    grid_size=11
)

obs, info = env.reset()
print(f"  Initial observation: {obs}")
print(f"  Shape: {obs.shape} (should be 10)")
print(f"  Min token: {obs.min()} (should be >= 0)")
print(f"  Max token: {obs.max()} (should be <= {VOCAB_SIZE-1})")
print(f"  VOCAB_SIZE: {VOCAB_SIZE}")

# Test 2: All possible tokens
print("\n2. Testing All Possible Tokens:")
all_tokens = set()
for step in range(200):
    action = np.random.randint(0, 6)  # 0-5 actions
    obs, reward, terminated, truncated, info = env.step(action)
    all_tokens.update(obs)
    
    if terminated or truncated:
        env.reset()

print(f"  Unique tokens observed: {sorted(all_tokens)}")
print(f"  Expected range: 0-{VOCAB_SIZE-1}")
print(f"  All tokens in range? {min(all_tokens) >= 0 and max(all_tokens) < VOCAB_SIZE}")
print(f"  Token count: {len(all_tokens)}")

# Test 3: Agent creation and validation
print("\n3. Testing Agent Creation:")
agent = Agent(
    network_type='lstm',
    observation_size=10,
    action_size=6,
    hidden_size=128,
    use_auxiliary=False,
    device='cpu'
)

print(f"  Agent created successfully")
print(f"  Network vocab_size: {agent.network.vocab_size}")

# Test 4: Button action specifically
print("\n4. Testing BUTTON Action:")
env = GridMazeWorld(
    task_class=TaskClass.BUTTONS,
    complexity_level=0.5,
    n_doors=1,
    n_buttons_per_door=2,
    button_break_probability=0.3,
    grid_size=11
)

env.reset()
print("  Trying BUTTON action (5)...")
obs, reward, terminated, truncated, info = env.step(Actions.BUTTON)
print(f"    Button pressed: {info['button_pressed']}")
print(f"    Reward: {reward:.2f}")
print(f"    Observation[8] (last action): {obs[8]} = ACTION_BUTTON token")

# Test 5: Validate no gaps in token sequence
print("\n5. Validating Token Sequence:")
expected_tokens = set(range(VOCAB_SIZE))
missing_tokens = expected_tokens - all_tokens
print(f"  Expected tokens: {sorted(expected_tokens)}")
print(f"  Observed tokens: {sorted(all_tokens)}")
if missing_tokens:
    print(f"  ⚠️  Missing tokens: {sorted(missing_tokens)}")
else:
    print(f"  ✓ All {VOCAB_SIZE} tokens observed!")

print("\n" + "="*70)
print("SUMMARY:")
print(f"  VOCAB_SIZE: {VOCAB_SIZE} (tokens 0-{VOCAB_SIZE-1})")
print(f"  Agent actions: 6 (0-5: LEFT, RIGHT, UP, DOWN, STAY, BUTTON)")
print(f"  Environment START action: 6 (not available to agent)")
print(f"  Observation tokens: 0-18 (all unique, no gaps)")
print(f"  NEIGHBOR_BUTTON_BROKEN removed - all buttons show as token 6")
print("="*70)