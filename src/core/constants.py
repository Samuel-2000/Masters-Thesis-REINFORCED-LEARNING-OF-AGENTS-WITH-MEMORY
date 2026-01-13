"""
Constants for Maze RL - centralized definitions for observations, actions, and tokens
ALL TOKENS ARE UNIQUE (0-19)
"""

from enum import IntEnum
from typing import Dict, Tuple

# ============================================================================
# OBSERVATION TOKENS (0-19) - ALL UNIQUE!
# ============================================================================

class ObservationTokens(IntEnum):
    """Observation token values (0-19) - ALL UNIQUE!"""
    
    # Neighbor observations (8 positions) - Tokens 0-7
    NEIGHBOR_EMPTY = 0
    NEIGHBOR_OBSTACLE = 1
    NEIGHBOR_FOOD_SOURCE = 2
    NEIGHBOR_FOOD = 3
    NEIGHBOR_DOOR_CLOSED = 4
    NEIGHBOR_DOOR_OPEN = 5
    NEIGHBOR_BUTTON = 6
    NEIGHBOR_BUTTON_BROKEN = 7
    
    # Action tokens - Tokens 8-13 (6 actions)
    ACTION_LEFT = 8
    ACTION_RIGHT = 9
    ACTION_UP = 10
    ACTION_DOWN = 11
    ACTION_STAY = 12
    ACTION_START = 13
    
    # Energy level tokens - Tokens 14-19 (6 levels)
    ENERGY_LEVEL_0 = 14  # 0-16.7%
    ENERGY_LEVEL_1 = 15  # 16.7-33.3%
    ENERGY_LEVEL_2 = 16  # 33.3-50%
    ENERGY_LEVEL_3 = 17  # 50-66.7%
    ENERGY_LEVEL_4 = 18  # 66.7-83.3%
    ENERGY_LEVEL_5 = 19  # 83.3-100%

# Vocabulary size for embedding layer (0-19 = 20 tokens)
VOCAB_SIZE = 20
MAX_TOKEN = ObservationTokens.ENERGY_LEVEL_5  # Should be 19

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
    START = 5

NUM_ACTIONS = len(Actions)
ACTION_SIZE = NUM_ACTIONS

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
    BUTTON_BROKEN = 8

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
    elif tile_value == TileType.BUTTON:
        return ObservationTokens.NEIGHBOR_BUTTON
    elif tile_value == TileType.BUTTON_BROKEN:
        return ObservationTokens.NEIGHBOR_BUTTON_BROKEN
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
    elif action == Actions.START:
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
    elif token == ObservationTokens.ACTION_START:
        return Actions.START
    else:
        raise ValueError(f"Token {token} is not an action token")

def energy_to_token(energy: float, max_energy: float = 100.0) -> int:
    """Convert continuous energy to discrete token (14-19)"""
    # Scale 0-100 to 0-5, then add 14
    scaled = int((energy / max_energy) * 6)  # 6 energy levels (0-5)
    scaled = max(0, min(5, scaled))  # Clamp to 0-5
    return ObservationTokens.ENERGY_LEVEL_0 + scaled

def token_to_energy(token: int, max_energy: float = 100.0) -> float:
    """Convert token back to approximate energy value"""
    if ObservationTokens.ENERGY_LEVEL_0 <= token <= ObservationTokens.ENERGY_LEVEL_5:
        level = token - ObservationTokens.ENERGY_LEVEL_0
        # Return midpoint of energy range
        energy_range = max_energy / 6
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
DEFAULT_ENERGY_PER_STEP = 0.1

# ============================================================================
# COLOR CONSTANTS (for rendering)
# ============================================================================

TILE_COLORS = {
    TileType.EMPTY: (40, 40, 40),
    TileType.OBSTACLE: (100, 100, 100),
    TileType.FOOD_SOURCE: (200, 50, 50),
    TileType.FOOD: (50, 200, 50),
    TileType.AGENT: (50, 50, 200),
    TileType.DOOR_CLOSED: (139, 69, 19),    # Brown
    TileType.DOOR_OPEN: (160, 120, 90),     # Lighter brown
    TileType.BUTTON: (255, 255, 0),         # Yellow
    TileType.BUTTON_BROKEN: (128, 128, 128) # Gray
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_observation_tokens():
    """Validate that all tokens are unique"""
    tokens = set()
    for member in ObservationTokens:
        if member.value in tokens:
            raise ValueError(f"Duplicate token value: {member.name} = {member.value}")
        tokens.add(member.value)
    
    # Check all tokens are in range 0-19
    if min(tokens) < 0 or max(tokens) >= VOCAB_SIZE:
        raise ValueError(f"Tokens out of range [0, {VOCAB_SIZE-1}]: min={min(tokens)}, max={max(tokens)}")
    
    print(f"✓ All {len(tokens)} tokens are unique (0-{max(tokens)})")
    return True

# Run validation when module is imported
if __name__ != "__main__":
    validate_observation_tokens()