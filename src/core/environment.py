"""
FULL PATCHED FILE — GridMazeWorld with template-based door placement
This file is a complete, drop-in replacement for your GridMazeWorld implementation.
It implements local 3x3 template matching for door candidate validation
(plus the previous connectivity/button placement logic).
"""

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from numba import njit, prange
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque

from .constants import (
    ObservationTokens, OBSERVATION_SIZE, NEIGHBOR_POSITIONS,
    ACTION_TOKEN_POSITION, ENERGY_TOKEN_POSITION, VOCAB_SIZE,
    Actions, NUM_ACTIONS, ENV_ACTIONS_START,
    TileType, TILE_COLORS,
    TaskClass,
    DEFAULT_GRID_SIZE, DEFAULT_MAX_STEPS, DEFAULT_OBSTACLE_FRACTION,
    DEFAULT_FOOD_SOURCES, DEFAULT_FOOD_ENERGY, DEFAULT_INITIAL_ENERGY,
    DEFAULT_ENERGY_DECAY, DEFAULT_ENERGY_PER_STEP,
    action_to_token, grid_tile_to_observation_token
)


# --------------------- Data classes ----------------------------------
@dataclass
class Door:
    y: int
    x: int
    is_open: bool = False
    timer: int = 0
    open_duration: int = 10
    close_duration: int = 20
    can_be_opened: bool = True
    requires_button: bool = True
    is_choke_point: bool = False

    def update(self, agent_pos: Optional[np.ndarray] = None):
        if agent_pos is not None and len(agent_pos) >= 2:
            agent_y, agent_x = int(agent_pos[0]), int(agent_pos[1])
            if agent_y == self.y and agent_x == self.x:
                if self.is_open:
                    self.timer = 0
                return
        if self.is_open:
            self.timer += 1
            if self.timer >= self.open_duration:
                self.is_open = False
                self.timer = 0
        elif not self.requires_button:
            self.timer += 1
            if self.timer >= self.close_duration:
                self.is_open = True
                self.timer = 0

    def open(self):
        if self.can_be_opened:
            self.is_open = True
            self.timer = 0
            return True
        return False


@dataclass
class Button:
    y: int
    x: int
    door_idx: int
    is_broken: bool = False
    break_probability: float = 0.0

    def press(self):
        if not self.is_broken:
            if self.break_probability > 0 and np.random.random() < self.break_probability:
                self.is_broken = True
                return False
            return True
        return False


# ------------------ NumPy / Numba accelerated helpers ------------------
@njit(cache=True, parallel=True)
def add_obstacles_connectivity(grid: np.ndarray, n_obstacles: int) -> np.ndarray:
    h, w = grid.shape
    total_cells = h * w
    empty_mask = (grid == TileType.EMPTY).flatten()
    empty_ids = np.where(empty_mask)[0]
    n_empty = len(empty_ids)
    if n_obstacles > n_empty - 1:
        n_obstacles = n_empty - 1
    visited = np.zeros(total_cells, dtype=np.uint8)
    queue = np.empty(total_cells, dtype=np.int32)
    added = 0
    for _ in range(n_obstacles):
        for attempt in range(n_empty):
            pick = np.random.randint(0, n_empty)
            cell = empty_ids[pick]
            r, c = cell // w, cell % w
            grid[r, c] = TileType.OBSTACLE
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
def food_step(agent_y: int, agent_x: int, food_sources: np.ndarray, food_energy: float) -> float:
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
def get_observation_optimized(y: int, x: int, grid: np.ndarray, food_sources: np.ndarray,
                              last_action: int, energy: float, food_positions_cache: np.ndarray,
                              door_open_array: np.ndarray) -> np.ndarray:
    obs = np.empty(10, dtype=np.int32)
    grid_h, grid_w = grid.shape
    offsets = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int32)
    for i in range(8):
        ny = y + offsets[i, 0]
        nx = x + offsets[i, 1]
        if 0 <= ny < grid_h and 0 <= nx < grid_w:
            grid_val = grid[ny, nx]
            if food_positions_cache[ny, nx] > 0:
                obs[i] = 3
            elif grid_val == TileType.DOOR_CLOSED or grid_val == TileType.DOOR_OPEN:
                if door_open_array[ny, nx] == 1:
                    obs[i] = 5
                else:
                    obs[i] = 4
            elif grid_val == TileType.OBSTACLE:
                obs[i] = 1
            elif grid_val == TileType.BUTTON or grid_val == TileType.BUTTON_BROKEN:
                obs[i] = 6
            elif grid_val == TileType.FOOD_SOURCE:
                obs[i] = 2
            else:
                obs[i] = 0
        else:
            obs[i] = 1
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
    energy_scaled = int((energy / 100.0) * 5)
    if energy_scaled < 0:
        energy_scaled = 0
    elif energy_scaled > 4:
        energy_scaled = 4
    obs[9] = 14 + energy_scaled
    return obs


# ------------------------- GridMazeWorld -------------------------------
class GridMazeWorld(gym.Env):
    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE, max_steps: int = DEFAULT_MAX_STEPS,
                 obstacle_fraction: float = DEFAULT_OBSTACLE_FRACTION, n_food_sources: int = DEFAULT_FOOD_SOURCES,
                 food_energy: float = DEFAULT_FOOD_ENERGY, initial_energy: float = DEFAULT_INITIAL_ENERGY,
                 energy_decay: float = DEFAULT_ENERGY_DECAY, energy_per_step: float = DEFAULT_ENERGY_PER_STEP,
                 render_size: int = 512, task_class: str = TaskClass.BASIC, complexity_level: float = 0.0,
                 n_doors: int = -1, door_open_duration: int = 10, door_close_duration: int = 20,
                 n_buttons_per_door: int = -1, button_break_probability: float = -1.0, door_periodic: bool = None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_food_sources = n_food_sources
        self.food_energy = food_energy
        self.initial_energy = initial_energy
        self.energy_decay = energy_decay
        self.energy_per_step = energy_per_step
        self.render_size = render_size
        self.task_class = task_class
        self.complexity_level = max(0.0, min(1.0, complexity_level))
        self.door_open_duration = door_open_duration
        self.door_close_duration = door_close_duration
        self.n_doors = n_doors
        self.n_buttons_per_door = n_buttons_per_door
        self.button_break_probability = button_break_probability
        self.door_periodic = door_periodic
        self._adjust_parameters_by_task_class()
        self.n_obstacles = int((grid_size - 2) ** 2 * obstacle_fraction)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=VOCAB_SIZE - 1, shape=(OBSERVATION_SIZE,), dtype=np.int32)
        self.grid = None
        self.food_sources = None
        self.food_positions_cache = None
        self.door_open_array = None
        self.agent_pos = None
        self.energy = None
        self.steps = None
        self.done = None
        self.last_action = None
        self.doors: List[Door] = []
        self.buttons: List[Button] = []
        self.colors = TILE_COLORS
        self.debug = False  # Set to False for normal use
        # Prepare template set once
        self._init_templates()

    def _adjust_parameters_by_task_class(self):
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
                self.n_buttons_per_door = 4  # Allow up to 4 buttons
            if self.button_break_probability == -1.0:
                self.button_break_probability = self.complexity_level * 0.2
            self.door_periodic = False
        elif self.task_class == TaskClass.COMPLEX:
            if self.n_doors == -1:
                self.n_doors = max(2, int(self.complexity_level * 4))
            if self.n_buttons_per_door == -1:
                self.n_buttons_per_door = 4  # Allow up to 4 buttons
            if self.button_break_probability == -1.0:
                self.button_break_probability = self.complexity_level * 0.3
            self.door_periodic = True

    # ---------------- Templates ------------------------------------------
    def _init_templates(self):
        """Define a compact set of 3x3 templates. Values: 1=must obstacle, 0=must passable, -1=don't care."""
        T_vert = np.array([
            [-1,  0, -1],
            [ 1,  0,  1],
            [-1,  0, -1]
            ], dtype=np.int8)
        T_horiz = np.array([
            [-1,  1, -1],
            [ 0,  0,  0],
            [-1,  1, -1]
            ], dtype=np.int8)

        T_diag1 = np.array([
            [-1,  0,  1],
            [ 0,  0,  0],
            [ 1,  0, -1]
            ], dtype=np.int8)
        T_diag2 = np.array([
            [ 1,  0, -1],
            [ 0,  0,  0],
            [-1,  0,  1]
            ], dtype=np.int8)


        T_a = np.array([
            [-1,  1, -1],
            [ 0,  0, -1],
            [ 1,  0, -1]
            ], dtype=np.int8)
        T_b = np.array([
            [-1, -1, -1],
            [ 0,  0,  1],
            [ 1,  0, -1]
            ], dtype=np.int8)

        T_c = np.array([
            [-1,  0,  1],
            [ 1,  0,  0],
            [-1, -1, -1]
            ], dtype=np.int8)
        T_d = np.array([
            [-1,  0,  1],
            [-1,  0,  0],
            [-1,  1, -1]
            ], dtype=np.int8)
        
        T_e = np.array([
            [-1,  1, -1],
            [-1,  0,  0],
            [-1,  0,  1]
            ], dtype=np.int8)
        T_f = np.array([
            [-1, -1, -1],
            [ 1,  0,  0],
            [-1,  0,  1]
            ], dtype=np.int8)
        

        T_g = np.array([
            [ 1,  0, -1],
            [ 0,  0, -1],
            [-1,  1, -1]
            ], dtype=np.int8)
        T_h = np.array([
            [ 1,  0, -1],
            [ 0,  0,  1],
            [-1, -1, -1]
            ], dtype=np.int8)

        self.templates = [T_vert, T_horiz, T_diag1, T_diag2, T_a, T_b, T_c, T_d, T_e, T_f, T_g, T_h]
        #self.templates = []
        #for idx, base in enumerate(base_templates):
        #    for rot in range(4):
        #        mat = np.rot90(base, k=rot)
        #        name = f"T{idx}_r{rot}"
        #        self.templates.append((mat, name))
        self.templates.sort(key=lambda tn: np.sum(tn[0] != -1), reverse=True)

    def _matches_template(self, y: int, x: int, template: np.ndarray, grid_override: Optional[np.ndarray] = None) -> bool:
        grid = grid_override if grid_override is not None else self.grid
        H, W = grid.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ty = dy + 1
                tx = dx + 1
                val = template[ty, tx]
                if val == -1:
                    continue
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    if val == 0:
                        return False
                    else:
                        continue
                tile = grid[ny, nx]
                if val == 1:
                    if tile != TileType.OBSTACLE:
                        return False
                elif val == 0:
                    if not self._is_passable(tile):
                        return False
        return True

    # ------------------ Helper utilities ---------------------------------
    def _get_temp_grid_with_periodic_doors_open(self) -> np.ndarray:
        temp = self.grid.copy()
        for d in self.doors:
            if not d.requires_button:
                temp[d.y, d.x] = TileType.DOOR_OPEN
        return temp

    def _is_passable(self, tile_type: int) -> bool:
        return tile_type in [TileType.EMPTY, TileType.FOOD, TileType.FOOD_SOURCE, TileType.BUTTON, TileType.BUTTON_BROKEN, TileType.DOOR_OPEN]

    def _manhattan_distance(self, a_y: int, a_x: int, b_y: int, b_x: int) -> int:
        return abs(a_y - b_y) + abs(a_x - b_x)

    def _find_regions_separated_by_door(self, door_y: int, door_x: int, grid_to_use: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Find all regions (connected components) that would be separated if this door were closed.
        Uses the provided grid which may already have other doors as obstacles."""
        temp_grid = grid_to_use.copy()
        temp_grid[door_y, door_x] = TileType.OBSTACLE
        grid_h, grid_w = self.grid_size, self.grid_size
        visited = np.zeros((grid_h, grid_w), dtype=bool)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        regions = []
        for y in range(grid_h):
            for x in range(grid_w):
                if visited[y, x] or temp_grid[y, x] == TileType.OBSTACLE:
                    continue
                if temp_grid[y, x] not in [TileType.EMPTY, TileType.FOOD, TileType.FOOD_SOURCE, TileType.DOOR_OPEN, TileType.BUTTON, TileType.BUTTON_BROKEN]:
                    continue
                region = []
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    region.append((cy, cx))
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if not (0 <= ny < grid_h and 0 <= nx < grid_w):
                            continue
                        if visited[ny, nx] or temp_grid[ny, nx] == TileType.OBSTACLE:
                            continue
                        if temp_grid[ny, nx] in [TileType.EMPTY, TileType.FOOD, TileType.FOOD_SOURCE, TileType.DOOR_OPEN, TileType.BUTTON, TileType.BUTTON_BROKEN]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if region:
                    regions.append(region)
        return regions

    def _bfs_distance(self, start_y: int, start_x: int, target_y: int, target_x: int, 
                    grid_override: Optional[np.ndarray] = None) -> int:
        """Calculate actual shortest path distance using BFS"""
        grid = grid_override if grid_override is not None else self.grid
        h, w = grid.shape
        
        if start_y == target_y and start_x == target_x:
            return 0
        
        visited = np.zeros((h, w), dtype=bool)
        queue = deque()
        queue.append((start_y, start_x, 0))
        visited[start_y, start_x] = True
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            y, x, dist = queue.popleft()
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                    
                if visited[ny, nx]:
                    continue
                    
                # Check if cell is passable
                tile = grid[ny, nx]
                if not self._is_passable(tile):
                    continue
                    
                if ny == target_y and nx == target_x:
                    return dist + 1
                    
                visited[ny, nx] = True
                queue.append((ny, nx, dist + 1))
        
        return float('inf')  # Unreachable

    # ---------------- Door candidate scanning using templates ------------------
    def _find_door_candidates_with_templates(self, grid_to_use: np.ndarray) -> List[Tuple[int, int]]:
        """Find potential door positions using template matching on the provided grid."""
        candidates = []
        # For template matching, we treat existing doors as obstacles
        temp_grid = grid_to_use.copy()
        H, W = self.grid_size, self.grid_size
        
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if grid_to_use[y, x] != TileType.EMPTY:
                    continue
                
                # Skip positions too close to existing doors
                near_existing_door = False
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if grid_to_use[ny, nx] in (TileType.DOOR_OPEN, TileType.DOOR_CLOSED):
                            near_existing_door = True
                            break
                    if near_existing_door:
                        break
                if near_existing_door:
                    continue
                
                # Check if this position matches any template
                matches_template = False
                for tpl in self.templates:
                    if self._matches_template(y, x, tpl, grid_override=temp_grid):
                        matches_template = True
                        break
                
                if matches_template:
                    candidates.append((y, x))
        
        return candidates

    # ---------------- Door and button placement logic --------------------------
    def _can_place_door_with_buttons(self, y: int, x: int, grid_to_use: np.ndarray) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check if a door can be placed at (y, x) with proper button placement.
        Uses the provided grid for evaluation (grid may have other doors as obstacles).
        Returns (success, button_positions)"""
        
        # Check that position is empty in the provided grid
        if grid_to_use[y, x] != TileType.EMPTY:
            if self.debug:
                print(f"  Door candidate ({y},{x}): grid cell is not empty")
            return False, []
        
        # Find regions separated by this door on the current grid (with existing doors as obstacles)
        regions = self._find_regions_separated_by_door(y, x, grid_to_use)
        
        if self.debug:
            print(f"  Door candidate ({y},{x}): found {len(regions)} regions")
            for i, region in enumerate(regions):
                print(f"    Region {i}: {len(region)} cells, sample: {region[:3]}")
        
        # We need exactly one button per region (1-4 buttons)
        required_buttons = len(regions)
        
        # Check if we have enough button allowance
        if self.n_buttons_per_door > 0 and required_buttons > self.n_buttons_per_door:
            if self.debug:
                print(f"  Door candidate ({y},{x}): requires {required_buttons} buttons but n_buttons_per_door={self.n_buttons_per_door}")
            return False, []
        
        # Check if we can place a button in EACH region
        button_positions = []
        
        for i, region in enumerate(regions):
            # Find ALL suitable positions in this region (must be EMPTY in the current grid)
            candidate_positions = []
            for ry, rx in region:
                if grid_to_use[ry, rx] == TileType.EMPTY:
                    # Calculate distance on the current grid (with other doors as obstacles)
                    distance = self._bfs_distance(y, x, ry, rx, grid_to_use)
                    if distance <= self.door_open_duration -2:
                        candidate_positions.append((ry, rx, distance))
            
            if not candidate_positions:
                if self.debug:
                    print(f"  Door candidate ({y},{x}): region {i} (size {len(region)}) has no empty cell within {self.door_open_duration} steps")
                return False, []
            
            # Choose RANDOM position from candidates (not the closest)
            if len(candidate_positions) > 0:
                # Random selection
                idx = np.random.randint(0, len(candidate_positions))
                by, bx, dist = candidate_positions[idx]
                if self.debug:
                    print(f"  Door candidate ({y},{x}): region {i} -> RANDOM button at ({by},{bx}), distance {dist}")
                button_positions.append((by, bx))
            else:
                return False, []
        
        if self.debug:
            print(f"  Door candidate ({y},{x}): SUCCESS, button positions = {button_positions}")
        return True, button_positions

    # ---------------- Main initialization of doors & buttons ----------------
    def _init_doors_and_buttons(self):
        """Initialize doors and buttons with meaningful placement.
        Each door that requires a button gets exactly one button per connected component.
        Places doors SEQUENTIALLY, treating already placed doors as obstacles."""
        self.doors = []
        self.buttons = []
        
        if self.n_doors == 0:
            if self.debug:
                print("No doors requested (n_doors=0)")
            return
        
        # Start with the grid as it is (with obstacles and food sources)
        current_grid = self.grid.copy()
        
        # We'll try to place doors one by one
        placed_doors = 0
        attempts = 0
        max_attempts = 100  # Safety limit
        
        while placed_doors < self.n_doors and attempts < max_attempts:
            attempts += 1
            
            # Find candidate door positions using templates on CURRENT grid
            door_candidates = self._find_door_candidates_with_templates(current_grid)
            
            if len(door_candidates) == 0:
                if self.debug:
                    print("No more door candidates found")
                break
            
            # Shuffle candidates
            np.random.shuffle(door_candidates)
            
            door_placed_this_round = False
            
            for y, x in door_candidates:
                if placed_doors >= self.n_doors:
                    break
                
                # Don't place doors too close together
                too_close = False
                for d in self.doors:
                    if self._manhattan_distance(y, x, d.y, d.x) < 3:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                # Determine if door requires a button
                requires_button = True
                if self.task_class == TaskClass.DOORS:
                    requires_button = False
                elif self.task_class == TaskClass.COMPLEX:
                    requires_button = np.random.random() < 0.5
                
                if not requires_button:
                    # Place periodic door (no buttons needed)
                    door = Door(y=y, x=x, 
                               open_duration=self.door_open_duration,
                               close_duration=self.door_close_duration,
                               requires_button=False,
                               can_be_opened=True,
                               is_choke_point=True)
                    door.is_open = np.random.random() < 0.5
                    
                    # Add door to lists and update grid
                    self.doors.append(door)
                    self.grid[y, x] = TileType.DOOR_CLOSED
                    current_grid[y, x] = TileType.DOOR_CLOSED  # Update current grid
                    self.door_open_array[y, x] = 1 if door.is_open else 0
                    
                    placed_doors += 1
                    door_placed_this_round = True
                    
                    if self.debug:
                        print(f"  Placed periodic door at ({y},{x})")
                    break  # Break to restart candidate search
                else:
                    # Button door - check if we can place it with buttons
                    can_place, button_positions = self._can_place_door_with_buttons(y, x, current_grid)
                    
                    if can_place:
                        # Place door
                        door = Door(y=y, x=x, 
                                   open_duration=self.door_open_duration,
                                   close_duration=self.door_close_duration,
                                   requires_button=True,
                                   can_be_opened=True,
                                   is_choke_point=True)
                        
                        door_idx = len(self.doors)
                        self.doors.append(door)
                        self.grid[y, x] = TileType.DOOR_CLOSED
                        current_grid[y, x] = TileType.DOOR_CLOSED  # Update current grid
                        self.door_open_array[y, x] = 1 if door.is_open else 0
                        
                        # Place buttons
                        for by, bx in button_positions:
                            # Button positions should be empty (checked in _can_place_door_with_buttons)
                            button = Button(y=by, x=bx, door_idx=door_idx,
                                          break_probability=self.button_break_probability,
                                          is_broken=False)
                            self.buttons.append(button)
                            self.grid[by, bx] = TileType.BUTTON
                            current_grid[by, bx] = TileType.BUTTON  # Update current grid
                        
                        placed_doors += 1
                        door_placed_this_round = True
                        
                        if self.debug:
                            print(f"  Placed button door at ({y},{x}) with {len(button_positions)} buttons")
                        break  # Break to restart candidate search
            
            if not door_placed_this_round:
                # Couldn't place any door this round
                if self.debug:
                    print(f"Could not place any door in round {attempts}")
                break
        
        if self.debug:
            print(f"\nPlaced {placed_doors} doors and {len(self.buttons)} buttons total.")
            for i, door in enumerate(self.doors):
                door_buttons = [b for b in self.buttons if b.door_idx == i]
                print(f"  Door {i}: ({door.y},{door.x}), requires_button={door.requires_button}, buttons={len(door_buttons)}")

    # ---------------- rest of env (reset/step/render/etc) -----------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.grid[0, :] = TileType.OBSTACLE
        self.grid[-1, :] = TileType.OBSTACLE
        self.grid[:, 0] = TileType.OBSTACLE
        self.grid[:, -1] = TileType.OBSTACLE
        self.grid = add_obstacles_connectivity(self.grid, self.n_obstacles)
        self._init_food_sources()
        self.door_open_array = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._init_doors_and_buttons()
        self.food_positions_cache = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._update_food_cache()
        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        if len(empty_cells) == 0:
            empty_cells = np.argwhere(self.grid != TileType.OBSTACLE)
        self.agent_pos = empty_cells[np.random.choice(len(empty_cells))]
        self.energy = self.initial_energy
        self.steps = 0
        self.done = False
        self.last_action = ENV_ACTIONS_START
        info = {'energy': self.energy, 'steps': self.steps, 'position': self.agent_pos.copy(), 'task_class': self.task_class, 'complexity_level': self.complexity_level, 'n_doors': len(self.doors), 'n_buttons': len(self.buttons)}
        obs = self._get_observation()
        if self.debug:
            print(f"\nReset Environment:\n  Doors: {len(self.doors)}, Buttons: {len(self.buttons)}\n  Agent pos: {self.agent_pos}")
        return obs, info

    def _init_food_sources(self):
        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        if len(empty_cells) == 0 or self.n_food_sources <= 0:
            self.food_sources = np.zeros((0, 4), dtype=np.int32)
            return
        indices = np.random.choice(len(empty_cells), min(self.n_food_sources, len(empty_cells)), replace=False)
        self.food_sources = np.zeros((len(indices), 4), dtype=np.int32)
        for i, idx in enumerate(indices):
            y, x = empty_cells[idx]
            regen_time = np.random.randint(5, 15)
            self.food_sources[i] = [y, x, regen_time, 1]
            self.grid[y, x] = TileType.FOOD_SOURCE

    def _update_food_cache(self):
        if self.food_sources is None or self.food_sources.shape[0] == 0:
            return
        self.food_positions_cache.fill(0)
        for i in range(self.food_sources.shape[0]):
            y, x, _, has_food = self.food_sources[i]
            if has_food:
                self.food_positions_cache[y, x] = 1

    def _update_door_states(self):
        for door in self.doors:
            door.update(self.agent_pos)
            if door.is_open:
                self.grid[door.y, door.x] = TileType.DOOR_OPEN
                self.door_open_array[door.y, door.x] = 1
            else:
                self.grid[door.y, door.x] = TileType.DOOR_CLOSED
                self.door_open_array[door.y, door.x] = 0

    def _check_button_press(self, button_y: int, button_x: int) -> bool:
        for button in self.buttons:
            if button.y == button_y and button.x == button_x:
                if button.is_broken:
                    return False
                success = button.press()
                if button.is_broken:
                    self.grid[button_y, button_x] = TileType.BUTTON_BROKEN
                    door = self.doors[button.door_idx]
                    other_buttons_working = any(b for b in self.buttons if b.door_idx == button.door_idx and not b.is_broken)
                    if not other_buttons_working:
                        door.can_be_opened = False
                    return False
                if success and 0 <= button.door_idx < len(self.doors):
                    door = self.doors[button.door_idx]
                    if door.open():
                        self.door_open_array[door.y, door.x] = 1
                        return True
                break
        return False

    def _can_move_to(self, y: int, x: int) -> bool:
        if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
            return False
        tile_type = self.grid[y, x]
        return tile_type in [TileType.EMPTY, TileType.DOOR_OPEN, TileType.FOOD, TileType.FOOD_SOURCE, TileType.BUTTON, TileType.BUTTON_BROKEN]

    def _get_adjacent_button_positions(self, y: int, x: int) -> List[Tuple[int, int]]:
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
        return get_observation_optimized(int(self.agent_pos[0]), int(self.agent_pos[1]), self.grid, self.food_sources, self.last_action, self.energy, self.food_positions_cache, self.door_open_array)

    def render(self) -> Optional[np.ndarray]:
        if not hasattr(self, '_render_buffer') or self._render_buffer is None:
            cell_size = max(1, self.render_size // self.grid_size)
            self._render_buffer = np.zeros((self.grid_size * cell_size, self.grid_size * cell_size, 3), dtype=np.uint8)
            self._cell_size = cell_size
        self._render_buffer.fill(0)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = self.colors[self.grid[y, x]]
                y_start = y * self._cell_size
                x_start = x * self._cell_size
                self._render_buffer[y_start:y_start + self._cell_size, x_start:x_start + self._cell_size] = color
        if self.food_sources is not None:
            for i in range(self.food_sources.shape[0]):
                y, x, _, has_food = self.food_sources[i]
                if has_food:
                    center_y = int((y + 0.5) * self._cell_size)
                    center_x = int((x + 0.5) * self._cell_size)
                    radius = max(1, self._cell_size // 3)
                    cv2.circle(self._render_buffer, (center_x, center_y), radius, (0, 255, 0), -1)
        ay, ax = int(self.agent_pos[0]), int(self.agent_pos[1])
        center_y = int((ay + 0.5) * self._cell_size)
        center_x = int((ax + 0.5) * self._cell_size)
        radius = max(1, self._cell_size // 2)
        cv2.circle(self._render_buffer, (center_x, center_y), radius, (255, 255, 255), -1)
        info = f"Energy: {self.energy:.1f} | Step: {self.steps}/{self.max_steps}"
        info += f" | Task: {self.task_class} (Lvl: {self.complexity_level:.1f})"
        cv2.putText(self._render_buffer, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return self._render_buffer

    def close(self):
        if hasattr(self, '_render_buffer'):
            self._render_buffer = None
        cv2.destroyAllWindows()


class VectorGridMazeWorld(GridMazeWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._render_buffer = None
    def render(self):
        return None
    def close(self):
        pass