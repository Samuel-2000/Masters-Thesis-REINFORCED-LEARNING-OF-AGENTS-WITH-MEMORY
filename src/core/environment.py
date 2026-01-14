"""
Grid Maze Environment using Gymnasium with consistent observation space
"""

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from numba import njit, prange
from typing import Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass

from .constants import (
    # Observation constants
    ObservationTokens, OBSERVATION_SIZE, NEIGHBOR_POSITIONS,
    ACTION_TOKEN_POSITION, ENERGY_TOKEN_POSITION, VOCAB_SIZE,
    # Action constants
    Actions, NUM_ACTIONS, ENV_ACTIONS_START,
    # Tile constants
    TileType, TILE_COLORS,
    # Task classes
    TaskClass,
    # Default parameters
    DEFAULT_GRID_SIZE, DEFAULT_MAX_STEPS, DEFAULT_OBSTACLE_FRACTION,
    DEFAULT_FOOD_SOURCES, DEFAULT_FOOD_ENERGY, DEFAULT_INITIAL_ENERGY,
    DEFAULT_ENERGY_DECAY, DEFAULT_ENERGY_PER_STEP,
    # Mapping functions
    action_to_token, grid_tile_to_observation_token
)


@dataclass
class Door:
    """Data class for door properties"""
    y: int
    x: int
    is_open: bool = False
    timer: int = 0
    open_duration: int = 10
    close_duration: int = 20
    can_be_opened: bool = True
    requires_button: bool = True
    is_internal_door: bool = False  # New: indicates if this door is the only exit from a room
    
    def update(self, agent_pos: Optional[np.ndarray] = None):
        """Update door state based on timer and agent position"""
        # If agent is on the door, don't close it
        if agent_pos is not None and len(agent_pos) >= 2:
            agent_y, agent_x = int(agent_pos[0]), int(agent_pos[1])
            if agent_y == self.y and agent_x == self.x:
                if self.is_open:
                    self.timer = 0  # Reset timer while agent is on door
                return
        
        # ALL doors: if open, check if should close (both periodic AND button doors close automatically)
        if self.is_open:
            self.timer += 1
            if self.timer >= self.open_duration:
                self.is_open = False
                self.timer = 0
        # ONLY periodic doors (requires_button=False): if closed, check if should open automatically
        elif not self.requires_button:
            # Periodic doors open automatically when closed
            self.timer += 1
            if self.timer >= self.close_duration:
                self.is_open = True
                self.timer = 0
        # Button-operated doors (requires_button=True): if closed, stay closed (don't open automatically)
    
    def open(self):
        """Open the door if possible"""
        if self.can_be_opened:
            self.is_open = True
            self.timer = 0
            return True
        return False


@dataclass
class Button:
    """Data class for button properties"""
    y: int
    x: int
    door_idx: int  # Index of door it controls
    is_broken: bool = False
    break_probability: float = 0.0
    
    def press(self):
        """Attempt to press the button"""
        if not self.is_broken:
            # Small chance to break when pressed (only if break_probability > 0)
            if self.break_probability > 0 and np.random.random() < self.break_probability:
                self.is_broken = True
                return False  # Button broke, didn't work
            return True  # Button worked
        return False  # Button was already broken


@njit(cache=True, parallel=True)
def add_obstacles_connectivity(grid: np.ndarray, n_obstacles: int) -> np.ndarray:
    """Add obstacles while maintaining connectivity - OPTIMIZED VERSION"""
    h, w = grid.shape
    total_cells = h * w
    
    # Get empty cells using vectorized operations
    empty_mask = (grid == TileType.EMPTY).flatten()
    empty_ids = np.where(empty_mask)[0]
    n_empty = len(empty_ids)
    
    if n_obstacles > n_empty - 1:
        n_obstacles = n_empty - 1
    
    # Pre-allocate BFS arrays
    visited = np.zeros(total_cells, dtype=np.uint8)
    queue = np.empty(total_cells, dtype=np.int32)
    
    added = 0
    for _ in range(n_obstacles):
        for attempt in range(n_empty):
            pick = np.random.randint(0, n_empty)
            cell = empty_ids[pick]
            r, c = cell // w, cell % w
            
            # Try placing obstacle
            grid[r, c] = TileType.OBSTACLE
            
            # Find start for BFS (first empty cell except current)
            start = -1
            for j in range(n_empty):
                if j == pick:
                    continue
                nid = empty_ids[j]
                rr, cc = nid // w, nid % w
                if grid[rr, cc] == TileType.EMPTY:
                    start = nid
                    break
            
            if start < 0:
                grid[r, c] = TileType.EMPTY
                continue
            
            # BFS with optimizations
            visited.fill(0)
            head = tail = 0
            visited[start] = 1
            queue[0] = start
            tail = 1
            reach = 1
            
            while head < tail:
                cur = queue[head]
                head += 1
                cr, cc = cur // w, cur % w
                
                # Check neighbors using precomputed indices
                if cr > 0 and grid[cr-1, cc] == TileType.EMPTY:
                    nid = (cr-1) * w + cc
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
                
                if cr < h-1 and grid[cr+1, cc] == TileType.EMPTY:
                    nid = (cr+1) * w + cc
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
                
                if cc > 0 and grid[cr, cc-1] == TileType.EMPTY:
                    nid = cr * w + (cc-1)
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
                
                if cc < w-1 and grid[cr, cc+1] == TileType.EMPTY:
                    nid = cr * w + (cc+1)
                    if visited[nid] == 0:
                        visited[nid] = 1
                        queue[tail] = nid
                        tail += 1
                        reach += 1
            
            if reach == n_empty - 1:
                # Success - update empty cells
                empty_ids[pick] = empty_ids[n_empty - 1]
                n_empty -= 1
                added += 1
                break
            else:
                grid[r, c] = TileType.EMPTY
    
    return grid


@njit(cache=True)
def food_step(agent_y: int, agent_x: int, 
                   food_sources: np.ndarray, 
                   food_energy: float) -> float:
    """Food processing with minimal branching"""
    energy_gained = 0.0
    n_food = food_sources.shape[0]
    
    for i in prange(n_food):
        y, x, time_left, has_food = food_sources[i]
        
        # Check if agent is on this food source
        if agent_y == y and agent_x == x and has_food:
            energy_gained += food_energy
            food_sources[i, 2] = np.random.randint(5, 15)  # Regeneration time
            food_sources[i, 3] = 0  # Mark as empty
        elif time_left > 0: # Update regeneration timer
            food_sources[i, 2] = time_left - 1
        elif time_left == 0:
            food_sources[i, 3] = 1  # Regenerate food
    
    return energy_gained


@njit(cache=True)
def get_observation_optimized(
    y: int, x: int, 
    grid: np.ndarray, 
    food_sources: np.ndarray,
    last_action: int, 
    energy: float,
    food_positions_cache: np.ndarray,
    door_open_array: np.ndarray
) -> np.ndarray:
    """
    JIT-compiled observation generation with tokens 0-18
    
    Returns observation of length 10 with tokens 0-18
    """
    obs = np.empty(10, dtype=np.int32)  # Always 10 observations
    grid_h, grid_w = grid.shape
    
    # Neighbor offsets in order: NW, N, NE, W, E, SW, S, SE
    offsets = np.array([[-1, -1], [-1, 0], [-1, 1],
                        [0, -1], [0, 1],
                        [1, -1], [1, 0], [1, 1]], dtype=np.int32)
    
    # Process 8 neighbor positions (tokens 0-6)
    for i in range(8):
        ny = y + offsets[i, 0]
        nx = x + offsets[i, 1]
        
        if 0 <= ny < grid_h and 0 <= nx < grid_w:
            grid_val = grid[ny, nx]
            
            # Check if position has food using cache
            if food_positions_cache[ny, nx] > 0:
                obs[i] = 3  # NEIGHBOR_FOOD
            # Check if it's a door
            elif grid_val == TileType.DOOR_CLOSED or grid_val == TileType.DOOR_OPEN:
                if door_open_array[ny, nx] == 1:
                    obs[i] = 5  # NEIGHBOR_DOOR_OPEN
                else:
                    obs[i] = 4  # NEIGHBOR_DOOR_CLOSED
            # Map other tile types - BUTTON and BUTTON_BROKEN look the same!
            elif grid_val == TileType.OBSTACLE:
                obs[i] = 1  # NEIGHBOR_OBSTACLE
            elif grid_val == TileType.BUTTON or grid_val == TileType.BUTTON_BROKEN:
                obs[i] = 6  # NEIGHBOR_BUTTON (both working and broken)
            elif grid_val == TileType.FOOD_SOURCE:
                obs[i] = 2  # NEIGHBOR_FOOD_SOURCE
            else:  # EMPTY or AGENT
                obs[i] = 0  # NEIGHBOR_EMPTY
        else:
            # Out of bounds = obstacle
            obs[i] = 1  # NEIGHBOR_OBSTACLE
    
    # Position 8: Last action token (7-13)
    # Map action index to token
    if last_action == Actions.LEFT:  # 0
        obs[8] = 7  # ACTION_LEFT
    elif last_action == Actions.RIGHT:  # 1
        obs[8] = 8  # ACTION_RIGHT
    elif last_action == Actions.UP:  # 2
        obs[8] = 9  # ACTION_UP
    elif last_action == Actions.DOWN:  # 3
        obs[8] = 10  # ACTION_DOWN
    elif last_action == Actions.STAY:  # 4
        obs[8] = 11  # ACTION_STAY
    elif last_action == Actions.BUTTON:  # 5
        obs[8] = 12  # ACTION_BUTTON
    else:  # Environment START (6) or other
        obs[8] = 13  # ACTION_START
    
    # Position 9: Energy level token (14-18)
    # Scale energy (0-100) to 0-4, then add 14
    energy_scaled = int((energy / 100.0) * 5)  # 0-4
    if energy_scaled < 0:
        energy_scaled = 0
    elif energy_scaled > 4:
        energy_scaled = 4
    obs[9] = 14 + energy_scaled  # ENERGY_LEVEL_0 = 14
    
    return obs


class GridMazeWorld(gym.Env):
    """Grid Maze Environment with consistent observation space"""
    
    def __init__(self, 
                 grid_size: int = DEFAULT_GRID_SIZE,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 obstacle_fraction: float = DEFAULT_OBSTACLE_FRACTION,
                 n_food_sources: int = DEFAULT_FOOD_SOURCES,
                 food_energy: float = DEFAULT_FOOD_ENERGY,
                 initial_energy: float = DEFAULT_INITIAL_ENERGY,
                 energy_decay: float = DEFAULT_ENERGY_DECAY,
                 energy_per_step: float = DEFAULT_ENERGY_PER_STEP,
                 render_size: int = 512,
                 # New parameters for extended features
                task_class: str = TaskClass.BASIC,
                complexity_level: float = 0.0,
                n_doors: int = -1,  # -1 means "use task class default"
                door_open_duration: int = 10,
                door_close_duration: int = 20,
                n_buttons_per_door: int = -1,  # -1 means "use task class default"
                button_break_probability: float = -1.0,  # -1.0 means "use task class default"
                door_periodic: bool = None):  # None means "use task class default"
        
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_food_sources = n_food_sources
        self.food_energy = food_energy
        self.initial_energy = initial_energy
        self.energy_decay = energy_decay
        self.energy_per_step = energy_per_step
        self.render_size = render_size
        
        # Extended features - store DIRECTLY
        self.task_class = task_class
        self.complexity_level = max(0.0, min(1.0, complexity_level))
        self.door_open_duration = door_open_duration
        self.door_close_duration = door_close_duration

        self.n_doors = n_doors
        self.n_buttons_per_door = n_buttons_per_door
        self.button_break_probability = button_break_probability
        self.door_periodic = door_periodic



        
        # Adjust parameters based on task class and complexity
        self._adjust_parameters_by_task_class()
        
        # Calculate obstacle count
        self.n_obstacles = int((grid_size - 2) ** 2 * obstacle_fraction)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)  # 6 agent actions
        self.observation_space = spaces.Box(
            low=0, 
            high=VOCAB_SIZE - 1,  # Max token value (18)
            shape=(OBSERVATION_SIZE,), 
            dtype=np.int32
        )
        
        # Initialize state
        self.grid = None
        self.food_sources = None
        self.food_positions_cache = None
        self.door_open_array = None
        self.agent_pos = None
        self.energy = None
        self.steps = None
        self.done = None
        self.last_action = None
        
        # Extended features state
        self.doors: List[Door] = []
        self.buttons: List[Button] = []
        
        # Colors for rendering
        self.colors = TILE_COLORS
        
        # Debug flag
        self.debug = False
    
    def _adjust_parameters_by_task_class(self):
        """Adjust environment parameters based on task class and complexity level"""
        if self.task_class == TaskClass.BASIC:
            # BASIC: No doors, no matter what user specified
            self.n_doors = 0
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
            self.door_periodic = False
            
        elif self.task_class == TaskClass.DOORS:
            # DOORS: Periodic doors (open/close on timer), NO buttons
            # If user specified -1, use complexity-based value
            if self.n_doors == -1:
                self.n_doors = max(1, int(self.complexity_level * 3))
            
            # Override: DOORS class always has NO buttons
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
            
            # Override: DOORS class always periodic
            self.door_periodic = True
            
        elif self.task_class == TaskClass.BUTTONS:
            # BUTTONS: Button-operated doors (not periodic)
            if self.n_doors == -1:
                self.n_doors = max(1, int(self.complexity_level * 3))
            
            if self.n_buttons_per_door == -1:
                self.n_buttons_per_door = 1
            
            if self.button_break_probability == -1.0:
                self.button_break_probability = self.complexity_level * 0.2
            
            # Override: BUTTONS class never periodic
            self.door_periodic = False
            
        elif self.task_class == TaskClass.COMPLEX:
            # COMPLEX: Always 50/50 split between periodic and button doors
            # door_periodic parameter is ignored for COMPLEX task class
            if self.n_doors == -1:
                self.n_doors = max(2, int(self.complexity_level * 4))
            
            if self.n_buttons_per_door == -1:
                self.n_buttons_per_door = 2 if self.complexity_level > 0.5 else 1
            
            if self.button_break_probability == -1.0:
                self.button_break_probability = self.complexity_level * 0.3
            
            # For COMPLEX task, we always do 50/50 split, so door_periodic is not used
            # But we still need to set it for backward compatibility
            self.door_periodic = True  # Meaning "some doors are periodic"
            
        else:
            raise ValueError(f"Unknown task class: {self.task_class}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create grid with borders
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.grid[0, :] = TileType.OBSTACLE
        self.grid[-1, :] = TileType.OBSTACLE
        self.grid[:, 0] = TileType.OBSTACLE
        self.grid[:, -1] = TileType.OBSTACLE
        
        # Add obstacles
        self.grid = add_obstacles_connectivity(self.grid, self.n_obstacles)
        
        # Initialize food sources
        self._init_food_sources()
        
        # Initialize door open array BEFORE doors and buttons
        self.door_open_array = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Initialize doors and buttons (if any)
        self._init_doors_and_buttons()
        
        # Initialize food positions cache
        self.food_positions_cache = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._update_food_cache()
        
        # Place agent
        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        self.agent_pos = empty_cells[np.random.choice(len(empty_cells))]
        
        # Reset state variables
        self.energy = self.initial_energy
        self.steps = 0
        self.done = False
        self.last_action = ENV_ACTIONS_START  # Environment START action
        
        info = {
            'energy': self.energy,
            'steps': self.steps,
            'position': self.agent_pos.copy(),
            'task_class': self.task_class,
            'complexity_level': self.complexity_level,
            'n_doors': len(self.doors),
            'n_buttons': len(self.buttons)
        }
        
        # Get observation
        obs = self._get_observation()
        
        if self.debug:
            print(f"\nReset Environment:")
            print(f"  Task class: {self.task_class}")
            print(f"  Complexity: {self.complexity_level}")
            print(f"  Doors: {len(self.doors)}, Buttons: {len(self.buttons)}")
            print(f"  Agent pos: {self.agent_pos}")
            print(f"  Initial obs: {obs}")
            print(f"  Obs range: {obs.min()} to {obs.max()}")
            print(f"  VOCAB_SIZE: {VOCAB_SIZE}")
        
        return obs, info
    
    def _init_food_sources(self):
        """Food source initialization"""
        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        indices = np.random.choice(len(empty_cells), self.n_food_sources, replace=False)
        
        self.food_sources = np.zeros((self.n_food_sources, 4), dtype=np.int32)
        for i, idx in enumerate(indices):
            y, x = empty_cells[idx]
            regen_time = np.random.randint(5, 15)
            self.food_sources[i] = [y, x, regen_time, 1]  # Start with food
            
            # Mark food source on grid
            self.grid[y, x] = TileType.FOOD_SOURCE
    
    def _analyze_room_connectivity(self, door_y: int, door_x: int):
        """
        Analyze if a door is internal (only exit from a room).
        Returns True if removing this door would create two disconnected regions.
        """
        # Temporarily mark door as obstacle to test connectivity
        original_tile = self.grid[door_y, door_x]
        self.grid[door_y, door_x] = TileType.OBSTACLE
        
        # Find empty cells on each side of the door
        side1_cells = []
        side2_cells = []
        
        # Check north/south sides
        if door_y > 0 and self.grid[door_y-1, door_x] in [TileType.EMPTY, TileType.FOOD, TileType.FOOD_SOURCE]:
            side1_cells.append((door_y-1, door_x))
        if door_y < self.grid_size-1 and self.grid[door_y+1, door_x] in [TileType.EMPTY, TileType.FOOD, TileType.FOOD_SOURCE]:
            side2_cells.append((door_y+1, door_x))
        
        # Check east/west sides
        if door_x > 0 and self.grid[door_y, door_x-1] in [TileType.EMPTY, TileType.FOOD, TileType.FOOD_SOURCE]:
            side1_cells.append((door_y, door_x-1))
        if door_x < self.grid_size-1 and self.grid[door_y, door_x+1] in [TileType.EMPTY, TileType.FOOD, TileType.FOOD_SOURCE]:
            side2_cells.append((door_y, door_x+1))
        
        # If we have cells on both sides, check if they're connected without the door
        if side1_cells and side2_cells:
            start = side1_cells[0]
            # BFS from start, not allowed through the door
            visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            queue = [start]
            visited[start[0], start[1]] = True
            
            while queue:
                y, x = queue.pop(0)
                
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    
                    if (0 <= ny < self.grid_size and 0 <= nx < self.grid_size and
                        not visited[ny, nx] and
                        self.grid[ny, nx] != TileType.OBSTACLE and
                        not (ny == door_y and nx == door_x)):
                        
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            
            # Check if any cell from side2 is reachable
            reachable = any(visited[y, x] for y, x in side2_cells)
            is_internal = not reachable
        
        else:
            is_internal = False
        
        # Restore original tile
        self.grid[door_y, door_x] = original_tile
        
        return is_internal
    
    def _is_button_position_valid(self, button_y: int, button_x: int, door: Door) -> bool:
        """
        Check if a button position is valid (accessible from its side of the door)
        """
        # Basic validation
        if not (0 <= button_y < self.grid_size and 0 <= button_x < self.grid_size):
            return False
        
        # Cell must be empty
        if self.grid[button_y, button_x] != TileType.EMPTY:
            return False
        
        # Check if there's a clear path to the door (ignoring the door itself)
        # We use BFS with the constraint that we can't go through the door
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        queue = [(button_y, button_x)]
        visited[button_y, button_x] = True
        
        while queue:
            y, x = queue.pop(0)
            
            # Check if we've reached the door's neighborhood
            if abs(y - door.y) + abs(x - door.x) == 1:
                # We're adjacent to the door - check if we can actually reach it
                # without going through obstacles
                return True
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                
                # Skip out of bounds
                if not (0 <= ny < self.grid_size and 0 <= nx < self.grid_size):
                    continue
                
                # Skip visited cells
                if visited[ny, nx]:
                    continue
                
                # Skip obstacles and other doors
                if self.grid[ny, nx] in [TileType.OBSTACLE, TileType.DOOR_CLOSED, TileType.DOOR_OPEN]:
                    continue
                
                # Skip the door cell itself (can't go through closed doors)
                if ny == door.y and nx == door.x and not door.is_open:
                    continue
                
                visited[ny, nx] = True
                queue.append((ny, nx))
        
        return False
    
    def _create_button_at(self, y: int, x: int, door_idx: int):
        """Create a button at the specified position"""
        button = Button(
            y=y, x=x,
            door_idx=door_idx,
            break_probability=self.button_break_probability,
            is_broken=False
        )
        
        self.buttons.append(button)
        self.grid[y, x] = TileType.BUTTON
    
    def _place_buttons_for_door(self, door: Door, door_idx: int):
        """Place buttons for a door (max 2 buttons)"""
        y, x = door.y, door.x
        
        # Determine number of buttons for this door
        n_buttons_to_place = self.n_buttons_per_door
        
        # If this is an internal door (only exit), we need at least 1 button, max 2
        if door.is_internal_door:
            n_buttons_to_place = min(2, max(1, n_buttons_to_place))
        
        # Find empty cells adjacent to the door (max 4 positions: N, S, E, W)
        adjacent_cells = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                if self.grid[ny, nx] == TileType.EMPTY:
                    # Check connectivity - button should be accessible from its side
                    if self._is_button_position_valid(ny, nx, door):
                        adjacent_cells.append((ny, nx, dy, dx))
        
        # If we need 2 buttons but only have adjacent cells on one side,
        # we need to ensure the button is accessible from both sides
        if n_buttons_to_place == 2 and len(adjacent_cells) < 2:
            # For internal doors, we MUST have buttons on both sides
            # If we can't place 2 buttons, mark door as problematic
            if door.is_internal_door:
                # We'll try to find alternative positions further away
                for distance in range(2, 4):  # Try up to 3 cells away
                    for dy in range(-distance, distance + 1):
                        for dx in range(-distance, distance + 1):
                            if abs(dy) + abs(dx) == distance:  # Manhattan distance
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                                    if self.grid[ny, nx] == TileType.EMPTY:
                                        if self._is_button_position_valid(ny, nx, door):
                                            adjacent_cells.append((ny, nx, dy, dx))
                                            if len(adjacent_cells) >= 2:
                                                break
                        if len(adjacent_cells) >= 2:
                            break
                    if len(adjacent_cells) >= 2:
                        break
        
        # Place buttons
        placed_buttons = 0
        
        # First, try to place one button on each side if we need 2
        if n_buttons_to_place == 2:
            # Group cells by side (based on direction from door)
            sides = {'north': [], 'south': [], 'east': [], 'west': []}
            for ny, nx, dy, dx in adjacent_cells:
                if dy == -1:
                    sides['north'].append((ny, nx))
                elif dy == 1:
                    sides['south'].append((ny, nx))
                elif dx == -1:
                    sides['west'].append((ny, nx))
                elif dx == 1:
                    sides['east'].append((ny, nx))
            
            # Try to pick one button from opposite sides if possible
            opposite_pairs = [('north', 'south'), ('east', 'west')]
            for side1, side2 in opposite_pairs:
                if sides[side1] and sides[side2]:
                    # Place one button from each side
                    for side in [side1, side2]:
                        ny, nx = sides[side][0]
                        self._create_button_at(ny, nx, door_idx)
                        placed_buttons += 1
                        # Remove this position from further consideration
                        sides[side] = sides[side][1:]
                    break
            
            # If we couldn't place on opposite sides, just place available buttons
            if placed_buttons < 2:
                for side in ['north', 'south', 'east', 'west']:
                    for ny, nx in sides[side]:
                        if placed_buttons >= 2:
                            break
                        self._create_button_at(ny, nx, door_idx)
                        placed_buttons += 1
        
        else:  # Place 0 or 1 button
            if n_buttons_to_place == 1 and adjacent_cells:
                # Place one button in any available adjacent cell
                ny, nx, _, _ = adjacent_cells[0]
                self._create_button_at(ny, nx, door_idx)
                placed_buttons += 1
        
        return placed_buttons
    
    def _init_doors_and_buttons(self):
        """Initialize doors and buttons based on configuration"""
        self.doors = []
        self.buttons = []
        
        if self.n_doors == 0:
            return
        
        # Find positions for doors
        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        
        # Try to place doors at strategic positions
        candidate_cells = []
        for y, x in empty_cells:
            # Count empty neighbors
            empty_neighbors = 0
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    if self.grid[ny, nx] == TileType.EMPTY:
                        empty_neighbors += 1
            
            if 2 <= empty_neighbors <= 3:
                candidate_cells.append((y, x, empty_neighbors))
        
        # Sort by strategic value
        candidate_cells.sort(key=lambda x: x[2], reverse=True)
        
        # Place doors
        placed_doors = 0
        for y, x, _ in candidate_cells:
            if placed_doors >= self.n_doors:
                break
            
            # Check if this would be an internal door
            is_internal = self._analyze_room_connectivity(y, x)
            
            # Determine if this specific door requires buttons
            # Default: doors require buttons if n_buttons_per_door > 0
            requires_button = self.n_buttons_per_door > 0
            
            # Handle different task classes
            if self.task_class == TaskClass.COMPLEX:
                # COMPLEX task: Always 50/50 split between periodic and button doors
                # Ignore door_periodic parameter, do 50% periodic, 50% button
                if np.random.random() < 0.5:
                    requires_button = False
            elif self.task_class == TaskClass.DOORS:
                # DOORS task: All doors are periodic (no buttons)
                requires_button = False
            elif self.task_class == TaskClass.BUTTONS:
                # BUTTONS task: All doors are button-operated
                requires_button = True
            # BASIC task doesn't have doors (n_doors=0)
            
            # Create door
            door = Door(
                y=y, x=x,
                open_duration=self.door_open_duration,
                close_duration=self.door_close_duration,
                requires_button=requires_button,
                can_be_opened=True,
                is_internal_door=is_internal
            )
            
            # Set initial state for periodic doors (doors that don't require buttons)
            if not requires_button:
                # Periodic doors start randomly open or closed
                door.is_open = np.random.random() < 0.5
            
            self.doors.append(door)
            self.grid[y, x] = TileType.DOOR_CLOSED
            
            # Update door open array
            self.door_open_array[y, x] = 1 if door.is_open else 0
            
            # Place buttons for this door if needed (only for button-operated doors)
            if requires_button:
                n_placed = self._place_buttons_for_door(door, placed_doors)
                
                # If this is an internal door and we couldn't place at least 1 button,
                # remove the door and try another position
                if is_internal and n_placed < 1:
                    # Remove any buttons we placed
                    buttons_to_remove = [b for b in self.buttons if b.door_idx == placed_doors]
                    for button in buttons_to_remove:
                        self.buttons.remove(button)
                        self.grid[button.y, button.x] = TileType.EMPTY
                    
                    self.doors.pop()
                    self.grid[y, x] = TileType.EMPTY
                    continue
            
            placed_doors += 1
    
    def _update_food_cache(self):
        """Update the food positions cache"""
        self.food_positions_cache.fill(0)
        for i in range(self.food_sources.shape[0]):
            y, x, _, has_food = self.food_sources[i]
            if has_food:
                self.food_positions_cache[y, x] = 1
    
    def _update_door_states(self):
        """Update all door states with agent position check"""
        for door in self.doors:
            door.update(self.agent_pos)
            
            # Update grid representation
            if door.is_open:
                self.grid[door.y, door.x] = TileType.DOOR_OPEN
                self.door_open_array[door.y, door.x] = 1
            else:
                self.grid[door.y, door.x] = TileType.DOOR_CLOSED
                self.door_open_array[door.y, door.x] = 0
    
    def _check_button_press(self, button_y: int, button_x: int) -> bool:
        """Check if button at position (button_y, button_x) can be pressed"""
        for button in self.buttons:
            if button.y == button_y and button.x == button_x:
                # Don't allow pressing broken buttons
                if button.is_broken:
                    return False
                
                # Check if button is working
                success = button.press()
                
                # Update visual state
                if button.is_broken:
                    self.grid[button_y, button_x] = TileType.BUTTON_BROKEN
                    # If all buttons for this door are broken, mark door as unopenable
                    door = self.doors[button.door_idx]
                    other_buttons_working = any(
                        b for b in self.buttons 
                        if b.door_idx == button.door_idx and not b.is_broken
                    )
                    if not other_buttons_working:
                        door.can_be_opened = False
                    return False
                
                # Button worked, try to open the associated door
                if success and 0 <= button.door_idx < len(self.doors):
                    door = self.doors[button.door_idx]
                    if door.open():
                        self.door_open_array[door.y, door.x] = 1
                        return True
                break
        return False
    
    def _can_move_to(self, y: int, x: int) -> bool:
        """Check if agent can move to position (y, x)"""
        if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
            return False
        
        tile_type = self.grid[y, x]
        
        # Can move through empty tiles, open doors, food, and buttons
        return tile_type in [
            TileType.EMPTY,
            TileType.DOOR_OPEN,
            TileType.FOOD,
            TileType.FOOD_SOURCE,
            TileType.BUTTON,
            TileType.BUTTON_BROKEN
        ]
    
    def _get_adjacent_button_positions(self, y: int, x: int) -> List[Tuple[int, int]]:
        """Get positions of buttons adjacent to (y, x)"""
        adjacent = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:  # Include current cell
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                if self.grid[ny, nx] in [TileType.BUTTON, TileType.BUTTON_BROKEN]:
                    adjacent.append((ny, nx))
        return adjacent
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step function with consistent observation space"""
        if self.done:
            obs = self._get_observation()
            return obs, 0.0, True, True, {}
        
        # Validate action
        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"Invalid action: {action}. Must be 0-{NUM_ACTIONS-1}")
        
        # Update door states first
        self._update_door_states()
        
        # Handle different action types
        button_pressed = False
        moved = False
        y, x = self.agent_pos
        
        if action == Actions.BUTTON:
            # BUTTON action: try to press adjacent buttons
            adjacent_buttons = self._get_adjacent_button_positions(y, x)
            for by, bx in adjacent_buttons:
                if self._check_button_press(by, bx):
                    button_pressed = True
                    break
            # Agent doesn't move when pressing button
        else:
            # Movement actions
            moved = True
            if action == Actions.LEFT:
                if x > 0 and self._can_move_to(y, x-1):
                    x -= 1
            elif action == Actions.RIGHT:
                if x < self.grid_size-1 and self._can_move_to(y, x+1):
                    x += 1
            elif action == Actions.UP:
                if y > 0 and self._can_move_to(y-1, x):
                    y -= 1
            elif action == Actions.DOWN:
                if y < self.grid_size-1 and self._can_move_to(y+1, x):
                    y += 1
            # STAY action: agent doesn't move
        
        # Update agent position if moved
        if moved:
            self.agent_pos = np.array([y, x])
        
        # Process food if agent moved onto food
        energy_gained = 0.0
        if moved:
            energy_gained = food_step(y, x, self.food_sources, self.food_energy)
            
            # Update food cache if food was collected
            if energy_gained > 0:
                self.food_positions_cache[y, x] = 0
        
        # Update energy
        self.energy = (self.energy * self.energy_decay + 
                      energy_gained - self.energy_per_step)
        
        # Clip energy
        self.energy = max(0.0, min(self.energy, 100.0))
        
        # Update state
        self.steps += 1
        self.last_action = action  # Store the action taken
        
        # Check termination
        terminated = (self.steps >= self.max_steps or self.energy <= 0)
        truncated = False
        self.done = terminated or truncated
        
        # Calculate reward
        #reward = 0.01  # Survival reward per step
        #if energy_gained > 0:
        #    reward += 1.0  # Food collection reward
        #if action == Actions.BUTTON:
        #    if button_pressed:
        #        reward += 0.5  # Successful button press reward
        #    else:
        #        # Small penalty for pressing a non-working button
        #        # Helps agent learn button press isn't always effective
        #        reward -= 0.1
        
        #if self.energy < 10:
        #    reward -= 0.1  # Low energy penalty
        
        # Update food cache if food regenerated
        if self.steps % 2 == 0:
            self._update_food_cache()
        
        # Get observation
        obs = self._get_observation()
        
        if self.debug and self.steps % 10 == 0:
            print(f"Step {self.steps}: action={Actions(action).name}, energy={self.energy:.1f}")
            print(f"  Observation: {obs}")
        
        info = {
            'energy': self.energy,
            'steps': self.steps,
            'position': self.agent_pos.copy(),
            'food_collected': energy_gained > 0,
            'button_pressed': button_pressed,
            'action_taken': action,
            'task_class': self.task_class,
            'complexity_level': self.complexity_level,
            'n_doors_active': sum(1 for d in self.doors if d.can_be_opened),
            'n_buttons_working': sum(1 for b in self.buttons if not b.is_broken)
        }
        
        return obs, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation using JIT-compiled function"""
        return get_observation_optimized(
            self.agent_pos[0], self.agent_pos[1],
            self.grid, self.food_sources,
            self.last_action, self.energy,
            self.food_positions_cache,
            self.door_open_array
        )
    
    def render(self) -> Optional[np.ndarray]:
        """Render current state"""
        if not hasattr(self, '_render_buffer') or self._render_buffer is None:
            cell_size = self.render_size // self.grid_size
            self._render_buffer = np.zeros(
                (self.grid_size * cell_size, self.grid_size * cell_size, 3), 
                dtype=np.uint8
            )
            self._cell_size = cell_size
        
        # Clear buffer
        self._render_buffer.fill(0)
        
        # Draw grid cells
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = self.colors[self.grid[y, x]]
                y_start = y * self._cell_size
                x_start = x * self._cell_size
                self._render_buffer[y_start:y_start+self._cell_size, 
                                   x_start:x_start+self._cell_size] = color
        
        # Draw food
        for i in range(self.food_sources.shape[0]):
            y, x, _, has_food = self.food_sources[i]
            if has_food:
                center_y = int((y + 0.5) * self._cell_size)
                center_x = int((x + 0.5) * self._cell_size)
                radius = self._cell_size // 3
                cv2.circle(self._render_buffer, (center_x, center_y), 
                          radius, (0, 255, 0), -1)
        
        # Draw agent
        ay, ax = self.agent_pos
        center_y = int((ay + 0.5) * self._cell_size)
        center_x = int((ax + 0.5) * self._cell_size)
        radius = self._cell_size // 2
        cv2.circle(self._render_buffer, (center_x, center_y), 
                  radius, (255, 255, 255), -1)
        
        # Add info overlay
        info = f"Energy: {self.energy:.1f} | Step: {self.steps}/{self.max_steps}"
        info += f" | Task: {self.task_class} (Lvl: {self.complexity_level:.1f})"
        cv2.putText(self._render_buffer, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        return self._render_buffer
    
    def close(self):
        """Close environment"""
        if hasattr(self, '_render_buffer'):
            self._render_buffer = None
        cv2.destroyAllWindows()


# vectorized environment alternative
class VectorGridMazeWorld(GridMazeWorld):
    """version optimized for vectorized use (no rendering overhead)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Disable render components for pure training
        self._render_buffer = None
    
    def render(self):
        """Override to return None when not needed"""
        return None
    
    def close(self):
        """Minimal close"""
        pass