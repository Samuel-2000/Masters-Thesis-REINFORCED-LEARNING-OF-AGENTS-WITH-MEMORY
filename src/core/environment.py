"""
Grid Maze Environment using Gymnasium with consistent observation space
"""

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from numba import njit, prange
from typing import Tuple, Dict, Any, Optional, List
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
    
    def update(self):
        """Update door state based on timer"""
        if self.is_open:
            self.timer += 1
            if self.timer >= self.open_duration:
                self.is_open = False
                self.timer = 0
        else:
            self.timer += 1
            if not self.requires_button and self.timer >= self.close_duration:
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
    door_idx: int  # Index of door it controls
    is_broken: bool = False
    break_probability: float = 0.0
    
    def press(self):
        """Attempt to press the button"""
        if not self.is_broken:
            # Small chance to break when pressed
            if np.random.random() < self.break_probability:
                self.is_broken = True
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
                 n_doors: int = 0,
                 door_open_duration: int = 10,
                 door_close_duration: int = 20,
                 n_buttons_per_door: int = 1,
                 button_break_probability: float = 0.0,
                 door_periodic: bool = False):
        
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
        self.n_doors = n_doors
        self.door_open_duration = door_open_duration
        self.door_close_duration = door_close_duration
        self.n_buttons_per_door = n_buttons_per_door
        self.button_break_probability = max(0.0, min(1.0, button_break_probability))
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
            self.n_doors = 0
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
            self.door_periodic = False
            
        elif self.task_class == TaskClass.DOORS:
            if self.n_doors == 0:
                self.n_doors = max(1, int(self.complexity_level * 3))
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
            self.door_periodic = True
            
        elif self.task_class == TaskClass.BUTTONS:
            if self.n_doors == 0:
                self.n_doors = max(1, int(self.complexity_level * 3))
            if self.n_buttons_per_door == 0:
                self.n_buttons_per_door = 1
            self.button_break_probability = self.complexity_level * 0.2
            self.door_periodic = False
            
        elif self.task_class == TaskClass.COMPLEX:
            if self.n_doors == 0:
                self.n_doors = max(2, int(self.complexity_level * 4))
            if self.n_buttons_per_door == 0:
                self.n_buttons_per_door = 2 if self.complexity_level > 0.5 else 1
            self.button_break_probability = self.complexity_level * 0.3
            self.door_periodic = self.complexity_level > 0.7
            
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
        
        # Initialize doors and buttons (if any)
        self._init_doors_and_buttons()
        
        # Initialize door open array
        self.door_open_array = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
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
        for i in range(min(self.n_doors, len(candidate_cells))):
            y, x, _ = candidate_cells[i]
            
            # Create door
            door = Door(
                y=y, x=x,
                open_duration=self.door_open_duration,
                close_duration=self.door_close_duration,
                requires_button=self.n_buttons_per_door > 0,
                can_be_opened=True
            )
            
            if self.door_periodic:
                door.requires_button = False
                door.is_open = np.random.random() < 0.5
            
            self.doors.append(door)
            self.grid[y, x] = TileType.DOOR_CLOSED
            
            # Update door open array
            self.door_open_array[y, x] = 1 if door.is_open else 0
            
            # Place buttons for this door if needed
            if self.n_buttons_per_door > 0:
                self._place_buttons_for_door(door, i)
    
    def _place_buttons_for_door(self, door: Door, door_idx: int):
        """Place buttons near a door"""
        y, x = door.y, door.x
        
        # Find empty cells adjacent to the door
        adjacent_cells = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                if self.grid[ny, nx] == TileType.EMPTY:
                    adjacent_cells.append((ny, nx))
        
        # Place buttons
        n_buttons_to_place = min(self.n_buttons_per_door, len(adjacent_cells))
        
        if n_buttons_to_place == 0:
            return
        
        # Randomly select cells for buttons
        button_cells = np.random.choice(
            len(adjacent_cells), 
            size=n_buttons_to_place, 
            replace=False
        )
        
        for cell_idx in button_cells:
            by, bx = adjacent_cells[cell_idx]
            
            # Create button
            button = Button(
                y=by, x=bx,
                door_idx=door_idx,
                break_probability=self.button_break_probability,
                is_broken=np.random.random() < self.button_break_probability
            )
            
            self.buttons.append(button)
            
            # Mark button on grid
            if button.is_broken:
                self.grid[by, bx] = TileType.BUTTON_BROKEN
            else:
                self.grid[by, bx] = TileType.BUTTON
    
    def _update_food_cache(self):
        """Update the food positions cache"""
        self.food_positions_cache.fill(0)
        for i in range(self.food_sources.shape[0]):
            y, x, _, has_food = self.food_sources[i]
            if has_food:
                self.food_positions_cache[y, x] = 1
    
    def _update_door_states(self):
        """Update all door states and door_open_array"""
        for door in self.doors:
            door.update()
            
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
                # Agent can try to press ANY button (working or broken)
                # But only working buttons actually do something
                if button.press():  # This updates broken state if needed
                    # If button is broken, pressing does nothing
                    if button.is_broken:
                        # Visual feedback: broken buttons stay the same
                        self.grid[button_y, button_x] = TileType.BUTTON_BROKEN
                        return False  # Button press failed
                    
                    # Working button: try to open the associated door
                    if 0 <= button.door_idx < len(self.doors):
                        door = self.doors[button.door_idx]
                        if door.open():
                            self.door_open_array[door.y, door.x] = 1
                            return True
                else:
                    # Button was already broken
                    return False
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
        reward = 0.01  # Survival reward per step
        if energy_gained > 0:
            reward += 1.0  # Food collection reward
        if action == Actions.BUTTON:
            if button_pressed:
                reward += 0.5  # Successful button press reward
            else:
                # Small penalty for pressing a non-working button
                # Helps agent learn button press isn't always effective
                reward -= 0.1
        
        if self.energy < 10:
            reward -= 0.1  # Low energy penalty
        
        # Update food cache if food regenerated
        if self.steps % 2 == 0:
            self._update_food_cache()
        
        # Get observation
        obs = self._get_observation()
        
        if self.debug and self.steps % 10 == 0:
            print(f"Step {self.steps}: action={Actions(action).name}, reward={reward:.3f}, energy={self.energy:.1f}")
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
        
        return obs, reward, terminated, truncated, info
    
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