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
    is_choke_point: bool = False  # Indicates if this door is a meaningful blocker
    
    def update(self, agent_pos: Optional[np.ndarray] = None):
        """Update door state based on timer and agent position"""
        # If agent is on the door, don't close it
        if agent_pos is not None and len(agent_pos) >= 2:
            agent_y, agent_x = int(agent_pos[0]), int(agent_pos[1])
            if agent_y == self.y and agent_x == self.x:
                if self.is_open:
                    self.timer = 0
                return
        
        # ALL doors: if open, check if should close
        if self.is_open:
            self.timer += 1
            if self.timer >= self.open_duration:
                self.is_open = False
                self.timer = 0
        # ONLY periodic doors (requires_button=False): open automatically
        elif not self.requires_button:
            self.timer += 1
            if self.timer >= self.close_duration:
                self.is_open = True
                self.timer = 0
    
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
    door_idx: int
    is_broken: bool = False
    break_probability: float = 0.0
    
    def press(self):
        """Attempt to press the button"""
        if not self.is_broken:
            if self.break_probability > 0 and np.random.random() < self.break_probability:
                self.is_broken = True
                return False
            return True
        return False


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
                
                # Check neighbors
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
        
        if agent_y == y and agent_x == x and has_food:
            energy_gained += food_energy
            food_sources[i, 2] = np.random.randint(5, 15)
            food_sources[i, 3] = 0
        elif time_left > 0:
            food_sources[i, 2] = time_left - 1
        elif time_left == 0:
            food_sources[i, 3] = 1
    
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
    """JIT-compiled observation generation with tokens 0-18"""
    obs = np.empty(10, dtype=np.int32)
    grid_h, grid_w = grid.shape
    
    # Neighbor offsets
    offsets = np.array([[-1, -1], [-1, 0], [-1, 1],
                        [0, -1], [0, 1],
                        [1, -1], [1, 0], [1, 1]], dtype=np.int32)
    
    # Process 8 neighbor positions
    for i in range(8):
        ny = y + offsets[i, 0]
        nx = x + offsets[i, 1]
        
        if 0 <= ny < grid_h and 0 <= nx < grid_w:
            grid_val = grid[ny, nx]
            
            if food_positions_cache[ny, nx] > 0:
                obs[i] = 3  # NEIGHBOR_FOOD
            elif grid_val == TileType.DOOR_CLOSED or grid_val == TileType.DOOR_OPEN:
                if door_open_array[ny, nx] == 1:
                    obs[i] = 5  # NEIGHBOR_DOOR_OPEN
                else:
                    obs[i] = 4  # NEIGHBOR_DOOR_CLOSED
            elif grid_val == TileType.OBSTACLE:
                obs[i] = 1  # NEIGHBOR_OBSTACLE
            elif grid_val == TileType.BUTTON or grid_val == TileType.BUTTON_BROKEN:
                obs[i] = 6  # NEIGHBOR_BUTTON
            elif grid_val == TileType.FOOD_SOURCE:
                obs[i] = 2  # NEIGHBOR_FOOD_SOURCE
            else:
                obs[i] = 0  # NEIGHBOR_EMPTY
        else:
            obs[i] = 1  # NEIGHBOR_OBSTACLE
    
    # Last action token
    if last_action == Actions.LEFT:
        obs[8] = 7
    elif last_action == Actions.RIGHT:
        obs[8] = 8
    elif last_action == Actions.UP:
        obs[8] = 9
    elif last_action == Actions.DOWN:
        obs[8] = 10
    elif last_action == Actions.STAY:
        obs[8] = 11
    elif last_action == Actions.BUTTON:
        obs[8] = 12
    else:
        obs[8] = 13
    
    # Energy level token
    energy_scaled = int((energy / 100.0) * 5)
    if energy_scaled < 0:
        energy_scaled = 0
    elif energy_scaled > 4:
        energy_scaled = 4
    obs[9] = 14 + energy_scaled
    
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
                 task_class: str = TaskClass.BASIC,
                 complexity_level: float = 0.0,
                 n_doors: int = -1,
                 door_open_duration: int = 10,
                 door_close_duration: int = 20,
                 n_buttons_per_door: int = -1,
                 button_break_probability: float = -1.0,
                 door_periodic: bool = None):
        
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_food_sources = n_food_sources
        self.food_energy = food_energy
        self.initial_energy = initial_energy
        self.energy_decay = energy_decay
        self.energy_per_step = energy_per_step
        self.render_size = render_size
        
        # Extended features
        self.task_class = task_class
        self.complexity_level = max(0.0, min(1.0, complexity_level))
        self.door_open_duration = door_open_duration
        self.door_close_duration = door_close_duration
        self.n_doors = n_doors
        self.n_buttons_per_door = n_buttons_per_door
        self.button_break_probability = button_break_probability
        self.door_periodic = door_periodic
        
        # Adjust parameters based on task class
        self._adjust_parameters_by_task_class()
        
        # Calculate obstacle count
        self.n_obstacles = int((grid_size - 2) ** 2 * obstacle_fraction)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, 
            high=VOCAB_SIZE - 1,
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
            self.n_doors = 0
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
            self.door_periodic = False
            
        elif self.task_class == TaskClass.DOORS:
            if self.n_doors == -1:
                self.n_doors = max(1, int(self.complexity_level * 3))
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
            self.door_periodic = True
            
        elif self.task_class == TaskClass.BUTTONS:
            if self.n_doors == -1:
                self.n_doors = max(1, int(self.complexity_level * 3))
            if self.n_buttons_per_door == -1:
                self.n_buttons_per_door = 1
            if self.button_break_probability == -1.0:
                self.button_break_probability = self.complexity_level * 0.2
            self.door_periodic = False
            
        elif self.task_class == TaskClass.COMPLEX:
            if self.n_doors == -1:
                self.n_doors = max(2, int(self.complexity_level * 4))
            if self.n_buttons_per_door == -1:
                self.n_buttons_per_door = 2 if self.complexity_level > 0.5 else 1
            if self.button_break_probability == -1.0:
                self.button_break_probability = self.complexity_level * 0.3
            self.door_periodic = True
    
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
        
        # Initialize door open array
        self.door_open_array = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Initialize doors and buttons using the new meaningful door placement
        self._init_doors_and_buttons_meaningful()
        
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
        self.last_action = ENV_ACTIONS_START
        
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
        
        return obs, info
    
    def _init_food_sources(self):
        """Food source initialization"""
        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        indices = np.random.choice(len(empty_cells), self.n_food_sources, replace=False)
        
        self.food_sources = np.zeros((self.n_food_sources, 4), dtype=np.int32)
        for i, idx in enumerate(indices):
            y, x = empty_cells[idx]
            regen_time = np.random.randint(5, 15)
            self.food_sources[i] = [y, x, regen_time, 1]
            self.grid[y, x] = TileType.FOOD_SOURCE
    
    def _is_meaningful_door_position(self, y: int, x: int) -> bool:
        """Check if a position is a meaningful door location"""
        # Must be empty
        if self.grid[y, x] != TileType.EMPTY:
            return False
        
        # Check 8-neighborhood for obstacles - need at least 2 obstacles
        obstacle_count = 0
        door_in_neighborhood = False
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    if self.grid[ny, nx] == TileType.OBSTACLE:
                        obstacle_count += 1
                    elif (self.grid[ny, nx] == TileType.DOOR_CLOSED or 
                          self.grid[ny, nx] == TileType.DOOR_OPEN):
                        door_in_neighborhood = True
        
        # Need at least 2 obstacles in 8-neighborhood and no other doors nearby
        if obstacle_count < 2 or door_in_neighborhood:
            return False
        
        # Check that this is a choke point by analyzing connectivity
        # Temporarily mark this cell as obstacle to test if it splits the area
        original_value = self.grid[y, x]
        self.grid[y, x] = TileType.OBSTACLE
        
        # Find connected passable spaces on each side
        connected_regions = []
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Look for passable cells in 4-neighborhood
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                if self._is_passable(self.grid[ny, nx]) and not visited[ny, nx]:
                    # BFS from this neighbor to find connected region
                    region_size = 0
                    stack = [(ny, nx)]
                    visited[ny, nx] = True
                    
                    while stack:
                        cy, cx = stack.pop()
                        region_size += 1
                        
                        # Check 4-directional neighbors
                        for ddy, ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nny, nnx = cy + ddy, cx + ddx
                            if (0 <= nny < self.grid_size and 0 <= nnx < self.grid_size and
                                not visited[nny, nnx] and self._is_passable(self.grid[nny, nnx]) and
                                not (nny == y and nnx == x)):  # Skip the door position
                                visited[nny, nnx] = True
                                stack.append((nny, nnx))
                    
                    if region_size >= 3:  # At least 3 connected passable spaces
                        connected_regions.append(region_size)
        
        # Restore original value
        self.grid[y, x] = original_value
        
        # We need at least 2 distinct regions with at least 3 connected passable spaces each
        # This ensures the door is a meaningful blocker between two substantial areas
        if len(connected_regions) >= 2:
            # Sort regions by size
            connected_regions.sort(reverse=True)
            # Check if top 2 regions are substantial
            if connected_regions[0] >= 3 and connected_regions[1] >= 3:
                return True
        
        return False
    
    def _is_passable(self, tile_type: int) -> bool:
        """Check if a tile type is passable (not obstacle, door, or wall)"""
        return tile_type in [
            TileType.EMPTY,
            TileType.FOOD,
            TileType.FOOD_SOURCE,
            TileType.BUTTON,
            TileType.BUTTON_BROKEN
        ]
    
    def _init_doors_and_buttons_meaningful(self):
        """Initialize doors and buttons - only place doors in meaningful positions"""
        self.doors = []
        self.buttons = []
        
        if self.n_doors == 0:
            return
        
        # Find meaningful door positions
        candidate_positions = []
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if self._is_meaningful_door_position(y, x):
                    candidate_positions.append((y, x))
        
        if self.debug:
            print(f"Found {len(candidate_positions)} meaningful door positions")
        
        # Shuffle candidates
        np.random.shuffle(candidate_positions)
        
        placed_doors = 0
        for y, x in candidate_positions:
            if placed_doors >= self.n_doors:
                break
            
            # Determine door type based on task class
            requires_button = True
            if self.task_class == TaskClass.DOORS:
                requires_button = False
            elif self.task_class == TaskClass.COMPLEX:
                requires_button = np.random.random() < 0.5
            
            # Create door
            door = Door(
                y=y, x=x,
                open_duration=self.door_open_duration,
                close_duration=self.door_close_duration,
                requires_button=requires_button,
                can_be_opened=True,
                is_choke_point=True  # All doors placed this way are choke points
            )
            
            # Set initial state for periodic doors
            if not requires_button:
                door.is_open = np.random.random() < 0.5
            
            # Place door
            self.doors.append(door)
            self.grid[y, x] = TileType.DOOR_CLOSED
            self.door_open_array[y, x] = 1 if door.is_open else 0
            
            # Place buttons if needed
            if requires_button and self.n_buttons_per_door > 0:
                self._place_buttons_meaningful(door, placed_doors)
            
            placed_doors += 1
        
        if self.debug:
            print(f"Placed {placed_doors} doors in meaningful positions")
    
    def _place_buttons_meaningful(self, door: Door, door_idx: int):
        """Place buttons for a door in accessible positions"""
        y, x = door.y, door.x
        
        # Find accessible positions for buttons
        accessible_positions = []
        
        # Check all empty cells in a 3x3 area around the door
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    if self.grid[ny, nx] == TileType.EMPTY:
                        # Check if this position is accessible (has path to door when door is open)
                        if self._is_button_accessible(ny, nx, door):
                            accessible_positions.append((ny, nx))
        
        # Try to place buttons in positions that are on different sides of the door
        buttons_to_place = min(self.n_buttons_per_door, len(accessible_positions))
        if buttons_to_place == 0:
            return
        
        # Prioritize positions that are on opposite sides of the door
        placed_positions = []
        
        # Group positions by their relative direction to door
        directions = {'north': [], 'south': [], 'east': [], 'west': []}
        for by, bx in accessible_positions:
            if by < y:
                directions['north'].append((by, bx))
            elif by > y:
                directions['south'].append((by, bx))
            elif bx < x:
                directions['west'].append((by, bx))
            elif bx > x:
                directions['east'].append((by, bx))
        
        # Try to place buttons on opposite sides
        opposite_pairs = [('north', 'south'), ('east', 'west')]
        for side1, side2 in opposite_pairs:
            if directions[side1] and directions[side2] and len(placed_positions) < buttons_to_place:
                # Take first position from each side
                by1, bx1 = directions[side1][0]
                by2, bx2 = directions[side2][0]
                
                button1 = Button(
                    y=by1, x=bx1,
                    door_idx=door_idx,
                    break_probability=self.button_break_probability,
                    is_broken=False
                )
                self.buttons.append(button1)
                self.grid[by1][bx1] = TileType.BUTTON
                placed_positions.append((by1, bx1))
                
                if len(placed_positions) < buttons_to_place:
                    button2 = Button(
                        y=by2, x=bx2,
                        door_idx=door_idx,
                        break_probability=self.button_break_probability,
                        is_broken=False
                    )
                    self.buttons.append(button2)
                    self.grid[by2][bx2] = TileType.BUTTON
                    placed_positions.append((by2, bx2))
                
                # Remove used positions
                directions[side1].pop(0)
                directions[side2].pop(0)
        
        # Place remaining buttons in any accessible positions
        for direction in ['north', 'south', 'east', 'west']:
            for by, bx in directions[direction]:
                if len(placed_positions) >= buttons_to_place:
                    break
                
                button = Button(
                    y=by, x=bx,
                    door_idx=door_idx,
                    break_probability=self.button_break_probability,
                    is_broken=False
                )
                self.buttons.append(button)
                self.grid[by][bx] = TileType.BUTTON
                placed_positions.append((by, bx))
    
    def _is_button_accessible(self, button_y: int, button_x: int, door: Door) -> bool:
        """Check if a button position is accessible (has path to door when door is open)"""
        # Temporarily mark door as open for path checking
        original_door_state = self.grid[door.y, door.x]
        self.grid[door.y, door.x] = TileType.DOOR_OPEN
        
        # Use BFS to check if button can reach the door
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        queue = [(button_y, button_x)]
        visited[button_y, button_x] = True
        
        while queue:
            y, x = queue.pop(0)
            
            # Check if we've reached the door
            if y == door.y and x == door.x:
                # Restore door state
                self.grid[door.y, door.x] = original_door_state
                return True
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                
                # Skip out of bounds
                if not (0 <= ny < self.grid_size and 0 <= nx < self.grid_size):
                    continue
                
                # Skip visited cells
                if visited[ny, nx]:
                    continue
                
                # Check if cell is passable
                if self._is_passable(self.grid[ny, nx]) or (ny == door.y and nx == door.x):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        
        # Restore door state
        self.grid[door.y, door.x] = original_door_state
        return False
    
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
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
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
        self.last_action = action
        
        # Check termination
        terminated = (self.steps >= self.max_steps or self.energy <= 0)
        truncated = False
        self.done = terminated or truncated
        
        # Update food cache if food regenerated
        if self.steps % 2 == 0:
            self._update_food_cache()
        
        # Get observation
        obs = self._get_observation()
        
        if self.debug and self.steps % 10 == 0:
            print(f"Step {self.steps}: action={Actions(action).name}, energy={self.energy:.1f}")
        
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