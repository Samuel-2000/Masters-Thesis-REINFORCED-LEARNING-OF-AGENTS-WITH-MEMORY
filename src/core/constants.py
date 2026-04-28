"""
Constants for Maze RL - centralized definitions for observations, actions, and tokens
NO UNUSED TOKENS - 0-18 (19 tokens total)
"""

from enum import IntEnum
from typing import Dict, Tuple

# ============================================================================
# OBSERVATION TOKENS (0-18) - ALL UNIQUE, NO GAPS!
# ============================================================================

class ObservationTokens(IntEnum):
    """Observation token values (0-18) - ALL UNIQUE, NO GAPS!"""
    
    # Neighbor observations (8 positions) - Tokens 0-6
    NEIGHBOR_EMPTY = 0
    NEIGHBOR_OBSTACLE = 1
    NEIGHBOR_FOOD_SOURCE = 2
    NEIGHBOR_FOOD = 3
    NEIGHBOR_DOOR_CLOSED = 4
    NEIGHBOR_DOOR_OPEN = 5
    NEIGHBOR_BUTTON = 6  # All buttons look the same (broken or working)
    
    # Action tokens - Tokens 7-12 (6 actions)
    ACTION_LEFT = 7
    ACTION_RIGHT = 8
    ACTION_UP = 9
    ACTION_DOWN = 10
    ACTION_STAY = 11
    ACTION_BUTTON = 12  # Explicit button push
    
    # Environment-only action token
    ACTION_START = 13  # Used by environment at reset
    
    # Energy level tokens - Tokens 14-18 (5 levels)
    ENERGY_LEVEL_0 = 14  # 0-20%
    ENERGY_LEVEL_1 = 15  # 20-40%
    ENERGY_LEVEL_2 = 16  # 40-60%
    ENERGY_LEVEL_3 = 17  # 60-80%
    ENERGY_LEVEL_4 = 18  # 80-100%

# Vocabulary size for embedding layer (0-18 = 19 tokens)
VOCAB_SIZE = 19
MAX_TOKEN = ObservationTokens.ENERGY_LEVEL_4  # Should be 18

# ============================================================================
# Observation structure
# ============================================================================

# Always 10 tokens: 8 neighbors + last_action + energy
OBSERVATION_SIZE = 10
NEIGHBOR_POSITIONS = 8  # NW, N, NE, W, E, SW, S, SE
ACTION_TOKEN_POSITION = 8
ENERGY_TOKEN_POSITION = 9

# ============================================================================
# ACTION CONSTANTS
# ============================================================================

class Actions(IntEnum):
    """Agent actions - separate from observation tokens!"""
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4
    BUTTON = 5  # Explicit button push action

    def __str__(self):
        return self.name
    
# Environment internal action (not available to agent)
ENV_ACTIONS_START = 6

NUM_ACTIONS = len(Actions)  # 6 agent actions
ACTION_SIZE = NUM_ACTIONS
TOTAL_ACTIONS = 7  # Agent actions (6) + environment START (1)

# ============================================================================
# TILE TYPE CONSTANTS (for grid representation)
# ============================================================================

class TileType(IntEnum):
    """Tile types for grid representation"""
    EMPTY = 0
    OBSTACLE = 1
    FOOD_SOURCE = 2
    FOOD = 3
    AGENT = 4
    DOOR_CLOSED = 5
    DOOR_OPEN = 6
    BUTTON = 7
    BUTTON_BROKEN = 8  # Internal only - agent sees as BUTTON

# ============================================================================
# MAPPING FUNCTIONS
# ============================================================================

def grid_tile_to_observation_token(tile_value: int) -> int:
    """Convert grid tile value to observation token for neighbors"""
    if tile_value == TileType.FOOD:
        return ObservationTokens.NEIGHBOR_FOOD
    elif tile_value == TileType.FOOD_SOURCE:
        return ObservationTokens.NEIGHBOR_FOOD_SOURCE
    elif tile_value == TileType.OBSTACLE:
        return ObservationTokens.NEIGHBOR_OBSTACLE
    elif tile_value == TileType.DOOR_CLOSED:
        return ObservationTokens.NEIGHBOR_DOOR_CLOSED
    elif tile_value == TileType.DOOR_OPEN:
        return ObservationTokens.NEIGHBOR_DOOR_OPEN
    elif tile_value == TileType.BUTTON or tile_value == TileType.BUTTON_BROKEN:
        return ObservationTokens.NEIGHBOR_BUTTON  # Both look the same to agent!
    else:  # EMPTY or AGENT (agent shouldn't be in neighbor observation)
        return ObservationTokens.NEIGHBOR_EMPTY

def action_to_token(action: int) -> int:
    """Convert action to observation token (last action)"""
    if action == Actions.LEFT:
        return ObservationTokens.ACTION_LEFT
    elif action == Actions.RIGHT:
        return ObservationTokens.ACTION_RIGHT
    elif action == Actions.UP:
        return ObservationTokens.ACTION_UP
    elif action == Actions.DOWN:
        return ObservationTokens.ACTION_DOWN
    elif action == Actions.STAY:
        return ObservationTokens.ACTION_STAY
    elif action == Actions.BUTTON:
        return ObservationTokens.ACTION_BUTTON
    elif action == ENV_ACTIONS_START:
        return ObservationTokens.ACTION_START
    else:
        raise ValueError(f"Unknown action: {action}")

def token_to_action(token: int) -> int:
    """Convert observation token back to action"""
    if token == ObservationTokens.ACTION_LEFT:
        return Actions.LEFT
    elif token == ObservationTokens.ACTION_RIGHT:
        return Actions.RIGHT
    elif token == ObservationTokens.ACTION_UP:
        return Actions.UP
    elif token == ObservationTokens.ACTION_DOWN:
        return Actions.DOWN
    elif token == ObservationTokens.ACTION_STAY:
        return Actions.STAY
    elif token == ObservationTokens.ACTION_BUTTON:
        return Actions.BUTTON
    elif token == ObservationTokens.ACTION_START:
        return ENV_ACTIONS_START
    else:
        raise ValueError(f"Token {token} is not an action token")

def energy_to_token(energy: float, max_energy: float = 100.0) -> int:
    """Convert continuous energy to discrete token (14-18)"""
    # Scale 0-100 to 0-4, then add 14
    scaled = int((energy / max_energy) * 5)  # 5 energy levels (0-4)
    scaled = max(0, min(4, scaled))  # Clamp to 0-4
    return ObservationTokens.ENERGY_LEVEL_0 + scaled

def token_to_energy(token: int, max_energy: float = 100.0) -> float:
    """Convert token back to approximate energy value"""
    if ObservationTokens.ENERGY_LEVEL_0 <= token <= ObservationTokens.ENERGY_LEVEL_4:
        level = token - ObservationTokens.ENERGY_LEVEL_0
        # Return midpoint of energy range
        energy_range = max_energy / 5
        return (level + 0.5) * energy_range
    return max_energy / 2  # Default if token invalid

# ============================================================================
# NETWORK CONSTANTS
# ============================================================================

# Network input/output sizes (always fixed regardless of task complexity)
OBSERVATION_SIZE = 10
ACTION_SIZE = NUM_ACTIONS


# ============================================================================
# ENVIRONMENT CONSTANTS
# ============================================================================

# Food
MIN_FOOD_REGEN_TIME = 10
MAX_FOOD_REGEN_TIME = 30
FOOD_INTERVAL_INDEX = 2
FOOD_EXISTS_INDEX = 3

class TaskClass:
    """Task class definitions"""
    BASIC = "basic"
    DOORS = "doors"
    BUTTONS = "buttons"
    COMPLEX = "complex"

# Default environment parameters
DEFAULT_GRID_SIZE = 11
DEFAULT_MAX_STEPS = 100
DEFAULT_OBSTACLE_FRACTION = 0.25
DEFAULT_FOOD_SOURCES = 4
DEFAULT_FOOD_ENERGY = 10.0
DEFAULT_INITIAL_ENERGY = 30.0
DEFAULT_ENERGY_DECAY = 0.98
DEFAULT_ENERGY_PER_STEP = 1
DEFAULT_RENDER_SIZE = 512
DEFAULT_DOOR_OPEN_DURATION = 10
DEFAULT_DOOR_CLOSE_DURATION = 20

# Default train parameters
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_GAMMA = 0.97
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SAVE_INTERVAL = 1000
DEFAULT_TEST_INTERVAL = 100



# ============================================================================
# COLOR CONSTANTS (for rendering)
# ============================================================================

TILE_COLORS = {
    TileType.EMPTY: (40, 40, 40),
    TileType.OBSTACLE: (100, 100, 100),
    TileType.FOOD_SOURCE: (10, 50, 10),
    TileType.FOOD: (50, 200, 50),
    TileType.AGENT: (50, 50, 200),
    TileType.DOOR_CLOSED: (200, 200, 200),    # Brown
    TileType.DOOR_OPEN: (50, 50, 50),     # Lighter brown
    TileType.BUTTON: (0, 0, 200),         # Blue
    TileType.BUTTON_BROKEN: (200, 0, 0) # Red
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_observation_tokens():
    """Validate that all tokens are unique and contiguous"""
    tokens = set()
    for member in ObservationTokens:
        if member.value in tokens:
            raise ValueError(f"Duplicate token value: {member.name} = {member.value}")
        tokens.add(member.value)
    
    # Check all tokens are in range 0-18
    expected_tokens = set(range(VOCAB_SIZE))
    missing_tokens = expected_tokens - tokens
    extra_tokens = tokens - expected_tokens
    
    if missing_tokens:
        print(f"⚠️  Missing tokens: {sorted(missing_tokens)}")
    
    if extra_tokens:
        print(f"⚠️  Extra tokens: {sorted(extra_tokens)}")
    
    print(f"✓ All {len(tokens)} tokens are unique (0-{max(tokens)})")
    print(f"✓ Vocabulary size: {VOCAB_SIZE}")
    
    # Verify the token sequence has no gaps
    sorted_tokens = sorted(tokens)
    for i in range(len(sorted_tokens) - 1):
        if sorted_tokens[i + 1] != sorted_tokens[i] + 1:
            print(f"⚠️  Gap found between {sorted_tokens[i]} and {sorted_tokens[i + 1]}")
    
    return True

# Run validation when module is imported
if __name__ != "__main__":
    validate_observation_tokens()