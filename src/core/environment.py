# environment.py — GridMazeWorld with optimized template matching (vectorized masks + tree)
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from numba import njit, prange
from typing import Tuple, Dict, Any, Optional, List, Set
from dataclasses import dataclass
from collections import deque
import math

# Import your project's constants; adjust path as needed
from .constants import (
    OBSERVATION_SIZE, VOCAB_SIZE,
    Actions, NUM_ACTIONS, ENV_ACTIONS_START,
    TileType, TILE_COLORS,
    TaskClass,
    FOOD_COUNT_MAX, FOOD_COUNT_MIN, MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME, FOOD_REGEN_GROWTH_FACTOR, 
    FOOD_INTERVAL_INDEX, FOOD_EXISTS_INDEX, FOOD_COLLECTION_COUNT_INDEX
)

@njit(cache=True)
def _label_components_numba_inplace(pass_mask: np.ndarray, labels: np.ndarray):
    """
    In-place 4-connected component labeling.
    pass_mask: uint8 array with 0/1
    labels: int32 array (output), will be overwritten
    Returns number of components.
    """
    h, w = pass_mask.shape
    labels[:] = 0
    nlabels = 0

    maxsize = h * w
    stack_y = np.empty(maxsize, dtype=np.int32)
    stack_x = np.empty(maxsize, dtype=np.int32)

    for i in range(h):
        for j in range(w):
            if pass_mask[i, j] == 1 and labels[i, j] == 0:
                nlabels += 1
                lab = nlabels

                top = 0
                stack_y[top] = i
                stack_x[top] = j
                top += 1
                labels[i, j] = lab

                while top > 0:
                    top -= 1
                    cy = stack_y[top]
                    cx = stack_x[top]

                    # up
                    ny = cy - 1
                    if ny >= 0 and pass_mask[ny, cx] == 1 and labels[ny, cx] == 0:
                        labels[ny, cx] = lab
                        stack_y[top] = ny
                        stack_x[top] = cx
                        top += 1
                    # down
                    ny = cy + 1
                    if ny < h and pass_mask[ny, cx] == 1 and labels[ny, cx] == 0:
                        labels[ny, cx] = lab
                        stack_y[top] = ny
                        stack_x[top] = cx
                        top += 1
                    # left
                    nx = cx - 1
                    if nx >= 0 and pass_mask[cy, nx] == 1 and labels[cy, nx] == 0:
                        labels[cy, nx] = lab
                        stack_y[top] = cy
                        stack_x[top] = nx
                        top += 1
                    # right
                    nx = cx + 1
                    if nx < w and pass_mask[cy, nx] == 1 and labels[cy, nx] == 0:
                        labels[cy, nx] = lab
                        stack_y[top] = cy
                        stack_x[top] = nx
                        top += 1

    return nlabels

# --------------------- Data classes ----------------------------------
@dataclass
class Door:
    y: int
    x: int
    open_duration: int
    close_duration: int
    can_be_opened: bool
    requires_button: bool
    is_choke_point: bool
    door_number: int
    is_open: bool = False
    timer: int = 0

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
    break_probability: float
    button_number: int
    is_broken: bool = False

    def press(self):
        if self.is_broken:
            return False
        
        if self.break_probability > 0.0 and np.random.random() < self.break_probability:
            self.is_broken = True
            return False
        
        return True
        
# ------------------ Template tree node & matcher -----------------------
class TemplateNode:
    __slots__ = ('split_pos', 'pass_child', 'obs_child', 'templates', 'is_leaf')

    def __init__(self, is_leaf: bool, split_pos: int = -1):
        self.is_leaf = is_leaf
        self.split_pos = split_pos
        self.pass_child = None
        self.obs_child = None
        self.templates: List[int] = []


class FastTemplateMatcher:
    """
    Template matcher that:
      - Holds obstacle/passable bitmasks for templates
      - Contains a compact decision tree (TemplateNode)
      - Provides a vectorized compute_all_neighborhood_masks(grid) function
      - matches(grid, y, x, neighborhood_mask=None) performs very fast checks
    """

    # Precomputed offsets in flattened 3x3 order (row-major)
    _OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 0), (0, 1),
                (1, -1), (1, 0), (1, 1)]
    _CENTER_IDX = 4

    def __init__(self, templates_flat: List[np.ndarray], max_depth: int):
        """
        templates_flat: list of 9-element arrays with values in {-1,0,1}
        """
        self.templates_flat = [np.array(t, dtype=np.int8).reshape(9,) for t in templates_flat]
        self.n_templates = len(self.templates_flat)
        self.template_array = np.stack(self.templates_flat, axis=0) if self.n_templates > 0 else np.zeros((0, 9), dtype=np.int8)

        # Precompute bitmasks
        self.obstacle_masks = np.zeros(self.n_templates, dtype=np.uint16)
        self.passable_masks = np.zeros(self.n_templates, dtype=np.uint16)
        for i in range(self.n_templates):
            obs = 0
            pas = 0
            flat = self.template_array[i]
            for j in range(9):
                v = int(flat[j])
                if v == 1:
                    obs |= (1 << j)
                elif v == 0:
                    pas |= (1 << j)
            self.obstacle_masks[i] = obs
            self.passable_masks[i] = pas

        # Build entropy-based decision tree
        self.root = self._build_tree(list(range(self.n_templates)), depth=0, max_depth=max_depth)

    # ---------------- Vectorized neighborhood masks ------------------
    def compute_all_neighborhood_masks(self, grid: np.ndarray) -> np.ndarray:
        """
        Compute 9-bit mask per cell where bit==1 indicates obstacle-like (Obstacle or Door).
        Out-of-bounds are treated as obstacle (bit set).
        Returns array shape (H, W) dtype uint16.
        """
        H, W = grid.shape
        obstacle_bool = ((grid == TileType.OBSTACLE) |
                         (grid == TileType.DOOR_CLOSED) |
                         (grid == TileType.DOOR_OPEN))
        # pad with True so out-of-bounds are treated as obstacles
        p = np.pad(obstacle_bool.astype(np.uint8), pad_width=1, constant_values=1)
        masks = np.zeros((H, W), dtype=np.uint16)

        for bit, (dy, dx) in enumerate(self._OFFSETS):
            sub = p[1 + dy: 1 + dy + H, 1 + dx: 1 + dx + W].astype(np.uint16)
            masks |= (sub << bit)
        return masks

    # ---------------- Fallback single mask ----------------------------
    def _neighborhood_mask(self, grid: np.ndarray, y: int, x: int) -> int:
        H, W = grid.shape
        mask = 0
        for bit, (dy, dx) in enumerate(self._OFFSETS):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                mask |= (1 << bit)
            else:
                t = grid[ny, nx]
                if t == TileType.OBSTACLE or t == TileType.DOOR_CLOSED or t == TileType.DOOR_OPEN:
                    mask |= (1 << bit)
        return mask

    # ---------------- Tree builder (entropy-based) --------------------
    def _entropy_score(self, indices: List[int], pos: int) -> float:
        counts = np.zeros(3, dtype=np.int32)  # -1,0,1 -> idx 0,1,2
        for idx in indices:
            val = int(self.template_array[idx, pos]) + 1
            counts[val] += 1
        total = len(indices)
        if total <= 1:
            return 0.0
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        return entropy

    def _build_tree(self, indices: List[int], depth: int, max_depth: int) -> TemplateNode:
        # Base cases
        if len(indices) <= 2 or depth >= max_depth:
            leaf = TemplateNode(is_leaf=True)
            leaf.templates = indices.copy()
            return leaf

        # choose best split pos (exclude center)
        best_pos = -1
        best_entropy = -1.0
        for pos in range(9):
            if pos == self._CENTER_IDX:
                continue
            entropy = self._entropy_score(indices, pos)
            if entropy > best_entropy:
                best_entropy = entropy
                best_pos = pos

        if best_pos == -1 or best_entropy < 0.08:
            leaf = TemplateNode(is_leaf=True)
            leaf.templates = indices.copy()
            return leaf

        pass_indices: List[int] = []
        obs_indices: List[int] = []
        for idx in indices:
            v = int(self.template_array[idx, best_pos])
            if v == 0 or v == -1:
                pass_indices.append(idx)
            if v == 1 or v == -1:
                obs_indices.append(idx)

        # if split doesn't reduce, make leaf
        if len(pass_indices) == len(indices) and len(obs_indices) == len(indices):
            leaf = TemplateNode(is_leaf=True)
            leaf.templates = indices.copy()
            return leaf

        node = TemplateNode(is_leaf=False)
        node.split_pos = best_pos
        node.pass_child = self._build_tree(pass_indices, depth + 1, max_depth) if pass_indices else None
        node.obs_child = self._build_tree(obs_indices, depth + 1, max_depth) if obs_indices else None
        return node

    # ---------------- Matching ---------------------------------------
    def matches(self, grid: np.ndarray, y: int, x: int, neighborhood_mask: Optional[int] = None) -> bool:
        """
        Check if any template matches at (y,x).
        If neighborhood_mask provided (uint9 mask), uses it to avoid recomputing neighborhood.
        """
        if grid[y, x] != TileType.EMPTY:
            return False

        if neighborhood_mask is None:
            neighborhood_mask = self._neighborhood_mask(grid, y, x)

        # center must be passable
        if ((neighborhood_mask >> self._CENTER_IDX) & 1) != 0:
            return False

        node = self.root
        # traverse using mask bits
        while node is not None and not node.is_leaf:
            bit = (neighborhood_mask >> node.split_pos) & 1
            if bit:
                node = node.obs_child
            else:
                node = node.pass_child
            if node is None:
                break

        if node is None:
            return False

        # Check candidates using bitmasks
        for ti in node.templates:
            req_obs = int(self.obstacle_masks[ti])
            req_pass = int(self.passable_masks[ti])
            if (neighborhood_mask & req_obs) != req_obs:
                continue
            if (neighborhood_mask & req_pass) != 0:
                continue
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
        y, x, time_left, has_food, collect_cnt = food_sources[i]
        if agent_y == y and agent_x == x and has_food:
            food_sources[i, FOOD_EXISTS_INDEX] = 0
            energy_gained += food_energy
            
            new_cnt = collect_cnt + 1
            food_sources[i, FOOD_COLLECTION_COUNT_INDEX] = new_cnt

            base_regen = np.random.randint(MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME)
            new_delay = base_regen * (FOOD_REGEN_GROWTH_FACTOR ** new_cnt)
            food_sources[i, FOOD_INTERVAL_INDEX] = int(new_delay)
        elif time_left > 0:
            food_sources[i, FOOD_INTERVAL_INDEX] = time_left - 1
        elif time_left == 0:
            food_sources[i, FOOD_EXISTS_INDEX] = 1
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


# ---------------------- Numba-accelerated BFS reachable mask -----------------
@njit(cache=True)
def bfs_reachable_mask(passable_mask: np.ndarray, h: int, w: int,
                       sy: int, sx: int, maxdist: int) -> np.ndarray:
    """
    Return a uint8 mask (h,w) where mask[y,x]==1 iff (y,x) is reachable
    from (sy,sx) within maxdist steps using 4-neighbour moves and only
    traversing passable cells (passable_mask==1). The start cell is included
    (distance 0).
    """
    visited = np.zeros((h, w), dtype=np.uint8)
    # simple circular queue arrays sized to h*w
    qy = np.empty(h * w, dtype=np.int32)
    qx = np.empty(h * w, dtype=np.int32)
    qd = np.empty(h * w, dtype=np.int32)
    head = 0
    tail = 0

    # if start not passable, return empty mask
    if passable_mask[sy, sx] == 0:
        return visited

    visited[sy, sx] = 1
    qy[tail] = sy
    qx[tail] = sx
    qd[tail] = 0
    tail += 1

    while head < tail:
        cy = qy[head]
        cx = qx[head]
        cd = qd[head]
        head += 1

        if cd >= maxdist:
            continue

        # 4-neighborhood
        # up
        ny = cy - 1
        nx = cx
        if ny >= 0:
            if visited[ny, nx] == 0 and passable_mask[ny, nx] == 1:
                visited[ny, nx] = 1
                qy[tail] = ny
                qx[tail] = nx
                qd[tail] = cd + 1
                tail += 1
        # down
        ny = cy + 1
        nx = cx
        if ny < h:
            if visited[ny, nx] == 0 and passable_mask[ny, nx] == 1:
                visited[ny, nx] = 1
                qy[tail] = ny
                qx[tail] = nx
                qd[tail] = cd + 1
                tail += 1
        # left
        ny = cy
        nx = cx - 1
        if nx >= 0:
            if visited[ny, nx] == 0 and passable_mask[ny, nx] == 1:
                visited[ny, nx] = 1
                qy[tail] = ny
                qx[tail] = nx
                qd[tail] = cd + 1
                tail += 1
        # right
        ny = cy
        nx = cx + 1
        if nx < w:
            if visited[ny, nx] == 0 and passable_mask[ny, nx] == 1:
                visited[ny, nx] = 1
                qy[tail] = ny
                qx[tail] = nx
                qd[tail] = cd + 1
                tail += 1

    return visited


# ------------------------- GridMazeWorld -------------------------------
class GridMazeWorld(gym.Env):
    # Cache for ring offsets across all instances (grid_size is fixed)
    _ring_offsets_cache = {}

    @classmethod
    def _get_ring_offsets(cls, grid_size: int):
        """Return precomputed Manhattan ring offsets for a given grid size."""
        if grid_size not in cls._ring_offsets_cache:
            max_dist = 2 * (grid_size - 1)
            ring_offsets = [[] for _ in range(max_dist + 1)]
            ring_offsets[0] = [(0, 0)]
            for d in range(1, max_dist + 1):
                offsets = set()
                for dy in range(-d, d + 1):
                    dx = d - abs(dy)
                    if dx == 0:
                        offsets.add((dy, 0))
                    else:
                        offsets.add((dy, dx))
                        offsets.add((dy, -dx))
                ring_offsets[d] = list(offsets)
            cls._ring_offsets_cache[grid_size] = ring_offsets
        return cls._ring_offsets_cache[grid_size]
    
    def __init__(self, grid_size: int, max_steps: int, obstacle_fraction: float, 
                 n_food_sources: int, food_energy: float, initial_energy: float,
                 energy_decay: float, energy_per_step: float,
                 render_size: int, task_class: str, complexity_level: float,
                 n_doors: int, door_open_duration: int, door_close_duration: int,
                 n_buttons_per_door: int, button_break_probability: float):
        


        super().__init__()
        self.grid_size = grid_size
        self._ring_offsets = self._get_ring_offsets(self.grid_size)
        self.max_steps = max_steps
        self.task_class = task_class
        self.complexity_level = 0.5 if complexity_level is None else max(0.0, min(1.0, complexity_level))

        # Randomize n_food_sources based on complexity
        # Binomial trial with p = (1 - complexity), but ensure a small chance of extra even at high complexity
        p = max(0.05, 1.0 - self.complexity_level)
        extra_max = FOOD_COUNT_MAX - FOOD_COUNT_MIN
        extra = np.random.binomial(extra_max, p)
        self.n_food_sources = FOOD_COUNT_MIN + extra

        self.food_energy = food_energy
        self.initial_energy = initial_energy
        self.energy_decay = energy_decay
        self.energy_per_step = energy_per_step
        self.render_size = render_size

        self.door_open_duration = door_open_duration
        self.door_close_duration = door_close_duration
        self.n_doors = n_doors
        self.n_buttons_per_door = n_buttons_per_door
        self.button_break_probability = button_break_probability
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
        self.debug = False



        # Create optimized template matcher using 12 templates
        templates_flat = self._templates_flat_list()
        self.template_matcher = FastTemplateMatcher(templates_flat, max_depth=4)

        # Precompute door proximity offsets
        self._door_check_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                                    (0, 1), (1, -1), (1, 0), (1, 1)]
        

        self._passable_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._labels = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)


    def _adjust_parameters_by_task_class(self):
        if self.task_class == TaskClass.BASIC:
            self.n_doors = 0
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
        elif self.task_class == TaskClass.DOORS:
            if self.n_doors == None:
                self.n_doors = max(1, int(self.complexity_level * 3))
            self.n_buttons_per_door = 0
            self.button_break_probability = 0.0
        elif self.task_class == TaskClass.BUTTONS:
            if self.n_doors == None:
                self.n_doors = max(1, int(self.complexity_level * 3))
            if self.n_buttons_per_door == None:
                self.n_buttons_per_door = 4
            if self.button_break_probability == None:
                self.button_break_probability = self.complexity_level * 0.2
        elif self.task_class == TaskClass.COMPLEX:
            if self.n_doors == None:
                self.n_doors = max(2, int(self.complexity_level * 4))
            if self.n_buttons_per_door == None:
                self.n_buttons_per_door = 4
            if self.button_break_probability == None:
                self.button_break_probability = self.complexity_level * 0.3

    # ---------------- Templates ------------------------------------------
    def _templates_flat_list(self) -> List[np.ndarray]:
        """Return list of 9-element flattened templates (values -1/0/1)."""
        templates_3x3 = [
            np.array([[-1,  0, -1], [ 1,  0,  1], [-1,  0, -1]], dtype=np.int8),  # T_vert
            np.array([[-1,  1, -1], [ 0,  0,  0], [-1,  1, -1]], dtype=np.int8),  # T_horiz
            np.array([[-1,  0,  1], [ 0,  0,  0], [ 1,  0, -1]], dtype=np.int8),  # T_diag1
            np.array([[ 1,  0, -1], [ 0,  0,  0], [-1,  0,  1]], dtype=np.int8),  # T_diag2
            np.array([[-1,  1, -1], [ 0,  0, -1], [ 1,  0, -1]], dtype=np.int8),  # T_a
            np.array([[-1, -1, -1], [ 0,  0,  1], [ 1,  0, -1]], dtype=np.int8),  # T_b
            np.array([[-1,  0,  1], [ 1,  0,  0], [-1, -1, -1]], dtype=np.int8),  # T_c
            np.array([[-1,  0,  1], [-1,  0,  0], [-1,  1, -1]], dtype=np.int8),  # T_d
            np.array([[-1,  1, -1], [-1,  0,  0], [-1,  0,  1]], dtype=np.int8),  # T_e
            np.array([[-1, -1, -1], [ 1,  0,  0], [-1,  0,  1]], dtype=np.int8),  # T_T_fhoriz
            np.array([[ 1,  0, -1], [ 0,  0, -1], [-1,  1, -1]], dtype=np.int8),  # T_g
            np.array([[ 1,  0, -1], [ 0,  0,  1], [-1, -1, -1]], dtype=np.int8)   # T_h
        ]
        # flatten row-major
        return [t.flatten() for t in templates_3x3]

    # ------------------ Helper utilities ---------------------------------
    def _is_passable(self, tile_type: int) -> bool:
        return tile_type in [TileType.EMPTY, TileType.FOOD, TileType.FOOD_SOURCE, TileType.BUTTON, TileType.BUTTON_BROKEN, TileType.DOOR_OPEN]

    def _manhattan_distance(self, a_y: int, a_x: int, b_y: int, b_x: int) -> int:
        return abs(a_y - b_y) + abs(a_x - b_x)




    def _find_regions_separated_by_door(
        self,
        door_y: int,
        door_x: int,
        grid_to_use: np.ndarray,
    ) -> List[List[Tuple[int, int]]]:
        """
        Optimized, semantics-identical version:
        - Door cell is treated as OBSTACLE
        - Finds ALL connected components of passable tiles
        - Uses reusable buffers and numba labeling
        """

        h, w = self.grid_size, self.grid_size
        pass_mask = self._passable_mask
        labels = self._labels

        # Build passable mask (same semantics as original)
        pass_mask[:] = 0
        pass_mask[(grid_to_use == TileType.EMPTY)] = 1
        pass_mask[(grid_to_use == TileType.FOOD)] = 1
        pass_mask[(grid_to_use == TileType.FOOD_SOURCE)] = 1
        pass_mask[(grid_to_use == TileType.DOOR_OPEN)] = 1
        pass_mask[(grid_to_use == TileType.BUTTON)] = 1
        pass_mask[(grid_to_use == TileType.BUTTON_BROKEN)] = 1

        # Door is closed
        pass_mask[door_y, door_x] = 0

        # Label components (numba, in-place)
        nlabels = _label_components_numba_inplace(pass_mask, labels)

        # --- Full region materialization (single scan) ---
        regions: List[List[Tuple[int, int]]] = [[] for _ in range(nlabels)]

        for y in range(h):
            for x in range(w):
                lab = labels[y, x]
                if lab > 0:
                    regions[lab - 1].append((y, x))

        return regions



    # ---------------- Door candidate scanning (vectorized) ------------------
    def _find_door_candidates_with_templates(self, grid_to_use: np.ndarray) -> List[Tuple[int, int]]:
        """Find potential door positions using optimized template matching (vectorized masks)."""
        H, W = self.grid_size, self.grid_size

        # 1) Compute all neighborhood masks (vectorized)
        masks = self.template_matcher.compute_all_neighborhood_masks(grid_to_use)

        # 2) Precompute near-existing-door boolean (vectorized)
        door_bool = ((grid_to_use == TileType.DOOR_OPEN) | (grid_to_use == TileType.DOOR_CLOSED)).astype(np.uint8)
        pdoor = np.pad(door_bool, pad_width=1, constant_values=0)
        near_door = np.zeros((H, W), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                near_door |= pdoor[1 + dy:1 + dy + H, 1 + dx:1 + dx + W].astype(bool)

        # 3) Candidate centers are empty cells AND not near door
        ys, xs = np.where((grid_to_use == TileType.EMPTY) & (~near_door))
        candidates: List[Tuple[int, int]] = []
        for y, x in zip(ys, xs):
            m = int(masks[y, x])  # uint16 -> int
            # double-check center bit is clear
            if ((m >> 4) & 1) != 0:
                continue
            if self.template_matcher.matches(grid_to_use, y, x, neighborhood_mask=m):
                candidates.append((int(y), int(x)))
        return candidates

    # ---------------------- Passable mask helpers -------------------------
    def _update_passable_mask(self) -> None:
        """
        Maintain a boolean (uint8) mask of cells considered passable by pathfinding
        and BFS routines. Call this after any change to self.grid that affects
        passability (doors opening/closing, placing buttons/food, placing obstacles).
        """
        g = self.grid
        mask = np.zeros_like(g, dtype=np.uint8)
        mask[np.where((g == TileType.EMPTY) |
                      (g == TileType.FOOD) |
                      (g == TileType.FOOD_SOURCE) |
                      (g == TileType.BUTTON) |
                      (g == TileType.BUTTON_BROKEN) |
                      (g == TileType.DOOR_OPEN))] = 1
        self._passable_mask = mask


    # ---------------- Door & button placement (optimized) ------------------
    def _can_place_door_with_buttons(self, y: int, x: int, grid_to_use: np.ndarray) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Optimized version of _can_place_door_with_buttons which:
          - reuses a single numba BFS per region (via bfs_reachable_mask) to get reachable empty
            cells within the time limit, instead of calling _bfs_distance for every
            candidate cell.
          - uses precomputed passable mask for fast passability checks.
        """
        if grid_to_use[y, x] != TileType.EMPTY:
            if self.debug:
                print(f"  Door candidate ({y},{x}): grid cell is not empty")
            return False, []

        regions = self._find_regions_separated_by_door(y, x, grid_to_use)

        if self.debug:
            print(f"  Door candidate ({y},{x}): found {len(regions)} regions")

        required_buttons = len(regions)
        if self.n_buttons_per_door > 0 and required_buttons > self.n_buttons_per_door:
            if self.debug:
                print(f"  Door candidate ({y},{x}): requires {required_buttons} buttons but n_buttons_per_door={self.n_buttons_per_door}")
            return False, []

        button_positions: List[Tuple[int, int]] = []
        h, w = grid_to_use.shape
        max_dist = max(0, self.door_open_duration - 2)

        # Ensure passable mask is up-to-date for this grid
        if grid_to_use is self.grid:
            if self._passable_mask is None:
                self._update_passable_mask()
            pass_mask = self._passable_mask
        else:
            pm = np.zeros_like(grid_to_use, dtype=np.uint8)
            pm[np.where((grid_to_use == TileType.EMPTY) |
                        (grid_to_use == TileType.FOOD) |
                        (grid_to_use == TileType.FOOD_SOURCE) |
                        (grid_to_use == TileType.BUTTON) |
                        (grid_to_use == TileType.BUTTON_BROKEN) |
                        (grid_to_use == TileType.DOOR_OPEN))] = 1
            pass_mask = pm

        for i, region in enumerate(regions):
            # Run one BFS from the door location to find all reachable cells within max_dist
            reachable = bfs_reachable_mask(pass_mask, h, w, y, x, max_dist)

            # Collect candidate empty cells in this region that are reachable
            candidate_positions = []
            for (ry, rx) in region:
                if grid_to_use[ry, rx] == TileType.EMPTY and reachable[ry, rx] == 1:
                    candidate_positions.append((ry, rx))

            if not candidate_positions:
                if self.debug:
                    print(f"  Door candidate ({y},{x}): region {i} (size {len(region)}) has no empty cell within {self.door_open_duration} steps")
                return False, []

            # choose a random candidate — this preserves your previous behavior (random choice per region)
            idx = np.random.randint(0, len(candidate_positions))
            by, bx = candidate_positions[idx]
            button_positions.append((by, bx))

        if self.debug:
            print(f"  Door candidate ({y},{x}): SUCCESS, button positions = {button_positions}")
        return True, button_positions

    # ---------------- Main initialization of doors & buttons ----------------
    def _init_doors_and_buttons(self):
        self.doors = []
        self.buttons = []

        if self.n_doors == 0:
            if self.debug:
                print("No doors requested")
            return

        current_grid = self.grid.copy()
        placed_doors = 0
        attempts = 0
        max_attempts = 50
        next_door_number = 1  # Start numbering from 1

        while placed_doors < self.n_doors and attempts < max_attempts:
            attempts += 1
            candidates = self._find_door_candidates_with_templates(current_grid)
            if not candidates:
                if self.debug:
                    print("No door candidates found")
                break

            np.random.shuffle(candidates)
            door_placed_this_round = False

            for y, x in candidates:
                if placed_doors >= self.n_doors:
                    break

                too_close = any(self._manhattan_distance(y, x, d.y, d.x) < 3 for d in self.doors)
                if too_close:
                    continue

                # determine door type
                requires_button = True
                if self.task_class == TaskClass.DOORS:
                    requires_button = False
                elif self.task_class == TaskClass.COMPLEX:
                    requires_button = np.random.random() < 0.5

                if not requires_button:
                    door = Door(y=y, x=x,
                                open_duration=self.door_open_duration,
                                close_duration=self.door_close_duration,
                                requires_button=False,
                                can_be_opened=True,
                                is_choke_point=True,
                                door_number=next_door_number)  # Assign door number
                    door.is_open = np.random.random() < 0.5
                    door_idx = len(self.doors)
                    self.doors.append(door)
                    self.grid[y, x] = TileType.DOOR_CLOSED
                    current_grid[y, x] = TileType.DOOR_CLOSED
                    self.door_open_array[y, x] = 1 if door.is_open else 0
                    next_door_number += 1
                    placed_doors += 1
                    door_placed_this_round = True
                    if self.debug:
                        print(f"Placed periodic door #{door.door_number} at ({y},{x})")
                    break
                else:
                    can_place, button_positions = self._can_place_door_with_buttons(y, x, current_grid)
                    if can_place:
                        door = Door(y=y, x=x,
                                    open_duration=self.door_open_duration,
                                    close_duration=self.door_close_duration,
                                    requires_button=True,
                                    can_be_opened=True,
                                    is_choke_point=True,
                                    door_number=next_door_number)  # Assign door number
                        door_idx = len(self.doors)
                        self.doors.append(door)
                        self.grid[y, x] = TileType.DOOR_CLOSED
                        current_grid[y, x] = TileType.DOOR_CLOSED
                        self.door_open_array[y, x] = 0
                        
                        # Create buttons for this door
                        for btn_idx, (by, bx) in enumerate(button_positions):
                            button = Button(y=by, x=bx, 
                                        door_idx=door_idx, 
                                        break_probability=self.button_break_probability, 
                                        is_broken=False,
                                        button_number=next_door_number)  # Same number as door
                            self.buttons.append(button)
                            self.grid[by, bx] = TileType.BUTTON
                            current_grid[by, bx] = TileType.BUTTON
                            if self.debug:
                                print(f"  Placed button #{button.button_number} for door #{door.door_number} at ({by},{bx})")
                        
                        next_door_number += 1
                        placed_doors += 1
                        door_placed_this_round = True
                        if self.debug:
                            print(f"Placed button door #{door.door_number} at ({y},{x})")
                        break

            if not door_placed_this_round:
                if self.debug:
                    print(f"Could not place any door in round {attempts}")
                break

        # After modifying grid with doors/buttons, update passable mask
        self._update_passable_mask()

        if self.debug:
            print(f"Placed {placed_doors} doors and {len(self.buttons)} buttons total.")
            for door in self.doors:
                buttons_for_door = [b for b in self.buttons if b.door_idx == self.doors.index(door)]
                print(f"Door #{door.door_number} at ({door.y},{door.x}): {len(buttons_for_door)} buttons")

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

        # Initialize caches BEFORE they are used by food/doors initialization
        self.food_positions_cache = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.door_open_array = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        self._init_food_sources()          # calls _update_food_cache internally (cache now exists)
        self._init_doors_and_buttons()     # uses door_open_array

        # Ensure passable mask is updated after all placements
        self._update_passable_mask()

        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        if len(empty_cells) == 0:
            empty_cells = np.argwhere(self.grid != TileType.OBSTACLE)
        self.agent_pos = empty_cells[np.random.choice(len(empty_cells))]
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
        obs = self._get_observation()
        if self.debug:
            print(f"\nReset Environment:\n  Doors: {len(self.doors)}, Buttons: {len(self.buttons)}\n  Agent pos: {self.agent_pos}")
        return obs, info

    
    """
    def _init_food_sources(self):
        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        if len(empty_cells) == 0 or self.n_food_sources <= 0:
            self.food_sources = np.zeros((0, 4), dtype=np.int32)
            return
        indices = np.random.choice(len(empty_cells), min(self.n_food_sources, len(empty_cells)), replace=False)
        self.food_sources = np.zeros((len(indices), 4), dtype=np.int32)
        for i, idx in enumerate(indices):
            y, x = empty_cells[idx]
            regen_time = np.random.randint(MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME)
            self.food_sources[i] = [y, x, regen_time, 1]
            self.grid[y, x] = TileType.FOOD_SOURCE

    """
    
    def _init_food_sources(self):
        rng = np.random
        n_food = self.n_food_sources

        empty_cells = np.argwhere(self.grid == TileType.EMPTY)
        N = len(empty_cells)
        if N == 0 or n_food <= 0:
            self.food_sources = np.zeros((0, 5), dtype=np.int32)
            return

        n_food = min(n_food, N)
        size = self.grid_size
        centre = (size - 1) * 0.5
        ec = empty_cells.astype(np.float32)

        # Centre-biased pool
        dist = np.abs(ec[:, 0] - centre) + np.abs(ec[:, 1] - centre)
        centre_count = min(N, max(n_food, N // 4))
        centre_pool = np.argpartition(dist, centre_count - 1)[:centre_count]
        rng.shuffle(centre_pool)

        # Spread pool: regular spatial subsample with random offset
        k = max(2, int(np.sqrt(N / max(n_food, 1))))
        oy = rng.randint(0, k)
        ox = rng.randint(0, k)
        spread_mask = ((empty_cells[:, 0] - oy) % k == 0) & ((empty_cells[:, 1] - ox) % k == 0)
        spread_pool = np.flatnonzero(spread_mask)
        rng.shuffle(spread_pool)

        # Smooth split by complexity
        c = float(self.complexity_level)
        n_centre = int((1.0 - c) * n_food)
        n_centre = max(0, min(n_centre, n_food))

        chosen = np.empty(n_food, dtype=np.int32)
        used = np.zeros(N, dtype=bool)
        pos = 0

        if n_centre > 0:
            centre_part = centre_pool[:n_centre]
            chosen[:len(centre_part)] = centre_part
            used[centre_part] = True
            pos = len(centre_part)

        if pos < n_food:
            spread_avail = spread_pool[~used[spread_pool]]
            take = min(n_food - pos, len(spread_avail))
            if take > 0:
                part = spread_avail[:take]
                chosen[pos:pos + take] = part
                used[part] = True
                pos += take

        if pos < n_food:
            remaining = np.flatnonzero(~used)
            extra = rng.choice(remaining, size=n_food - pos, replace=False)
            chosen[pos:] = extra

        self.food_sources = np.zeros((n_food, 5), dtype=np.int32)
        regen = rng.randint(MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME, size=n_food)

        for i, idx in enumerate(chosen):
            y, x = empty_cells[idx]
            self.food_sources[i] = [y, x, regen[i], 1, 0]
            self.grid[y, x] = TileType.FOOD_SOURCE

        self._update_food_cache()


    """
    def _init_food_sources(self):
        empty_cells = [tuple(cell) for cell in np.argwhere(self.grid == TileType.EMPTY)]
        if len(empty_cells) == 0 or self.n_food_sources <= 0:
            self.food_sources = np.zeros((0, 4), dtype=np.int32)
            return

        self.food_sources = np.zeros((self.n_food_sources, 4), dtype=np.int32)
        center = self.grid_size // 2

        # Gaussian distribution for row/col sampling
        std = (1.0 + self.complexity_level * (center - 1))
        indices = np.arange(self.grid_size)
        probs = np.exp(-0.5 * ((indices - center) / std) ** 2)
        probs /= probs.sum()

        empty_set = set(empty_cells)
        size = self.grid_size
        ring_offsets = self._ring_offsets

        # Repulsion strength scales with complexity (0 = none, 1 = full)
        strength = (0.1 + self.complexity_level) * size

        for i in range(self.n_food_sources):
            row = np.random.choice(size, p=probs)
            col = np.random.choice(size, p=probs)

            # Single‑step repulsion from already placed foods
            dx_total = 0.0
            dy_total = 0.0
            for j in range(i):
                other_y = self.food_sources[j, 0]
                other_x = self.food_sources[j, 1]
                dy = row - other_y
                dx = col - other_x
                dist = abs(dy) + abs(dx)
                if dist == 0:
                    # Random push if exactly overlapping
                    dx_total += np.random.uniform(-1, 1) * strength
                    dy_total += np.random.uniform(-1, 1) * strength
                else:
                    # Force = strength / (dist + ε) ; direction away
                    force = strength / (dist + 1e-6)
                    dx_total += force * (dx / dist)
                    dy_total += force * (dy / dist)

            # Apply displacement once
            row = int(np.clip(row + dy_total, 0, size - 1))
            col = int(np.clip(col + dx_total, 0, size - 1))

            #print(f"dx_total: {int(dx_total)}, dy_total: {int(dy_total)}, row: {row}, col: {col}")

            # Find nearest empty cell (spiral search)
            found = False
            for d in range(len(ring_offsets)):
                for dy, dx in ring_offsets[d]:
                    ny, nx = row + dy, col + dx
                    if 0 <= ny < size and 0 <= nx < size and (ny, nx) in empty_set:
                        row, col = ny, nx
                        found = True
                        break
                if found:
                    break
            if not found:
                row, col = next(iter(empty_set))

            regen = np.random.randint(MIN_FOOD_REGEN_TIME, MAX_FOOD_REGEN_TIME)
            self.food_sources[i] = [row, col, regen, 1]
            self.grid[row, col] = TileType.FOOD_SOURCE
            empty_set.remove((row, col))

        self._update_food_cache()
    """
       

    def _update_food_cache(self):
        if self.food_sources is None or self.food_sources.shape[0] == 0:
            return
        self.food_positions_cache.fill(0)
        for i in range(self.food_sources.shape[0]):
            y, x, _, has_food, _ = self.food_sources[i]
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
        # After doors change, update passable mask
        self._update_passable_mask()

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
                    # passable mask changed (button became broken -> tile type may be considered same passable class,
                    # but if you treat BUTTON_BROKEN differently adjust _update_passable_mask call placement)
                    self._update_passable_mask()
                    return False
                if success and 0 <= button.door_idx < len(self.doors):
                    door = self.doors[button.door_idx]
                    if door.open():
                        self.door_open_array[door.y, door.x] = 1
                        # door changed -> update grid / mask
                        self.grid[door.y, door.x] = TileType.DOOR_OPEN
                        self._update_passable_mask()
                        return True
                break
        return False

    def _can_move_to(self, y: int, x: int) -> bool:
        if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
            return False
        return self._is_passable(self.grid[y, x])

    def _get_adjacent_button_positions(self, y: int, x: int) -> List[Tuple[int, int]]:
        adjacent = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                if self.grid[ny, nx] in [TileType.BUTTON, TileType.BUTTON_BROKEN]:
                    adjacent.append((ny, nx))
        return adjacent

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            obs = self._get_observation()
            return obs, 0.0, True, True, {}

        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"Invalid action: {action}. Must be 0-{NUM_ACTIONS-1}")

        # Update door states first
        self._update_door_states()

        # Handle different action types
        button_pressed = False
        moved = False
        y, x = self.agent_pos

        if action == Actions.BUTTON:
            adjacent_buttons = self._get_adjacent_button_positions(y, x)
            for by, bx in adjacent_buttons:
                if self._check_button_press(by, bx):
                    button_pressed = True
                    break
        else:
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

        if moved:
            self.agent_pos = np.array([y, x])

        # Process food if agent moved onto food
        energy_gained = 0.0
        if moved:
            energy_gained = food_step(y, x, self.food_sources, self.food_energy)
            if energy_gained > 0:
                self.food_positions_cache[y, x] = 0

        # Update energy
        self.energy = (self.energy * self.energy_decay +
                       energy_gained - self.energy_per_step)
        self.energy = max(0.0, min(self.energy, 100.0))

        # Update state
        self.steps += 1
        self.last_action = action

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
                reward -= 0.1

        if self.energy < 10:
            reward -= 0.1

        # Update food cache if food regenerated
        if self.steps % 2 == 0:
            self._update_food_cache()

        obs = self._get_observation()

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
        return get_observation_optimized(
            int(self.agent_pos[0]), int(self.agent_pos[1]),
            self.grid, self.food_sources, self.last_action,
            self.energy, self.food_positions_cache, self.door_open_array
        )

    def render(self) -> Optional[np.ndarray]:
        if not hasattr(self, '_render_buffer') or self._render_buffer is None:
            cell_size = max(1, self.render_size // self.grid_size)
            self._render_buffer = np.zeros(
                (self.grid_size * cell_size, self.grid_size * cell_size, 3),
                dtype=np.uint8
            )
            self._cell_size = cell_size

        self._render_buffer.fill(0)

        # Draw grid tiles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = self.colors[self.grid[y, x]]
                y_start = y * self._cell_size
                x_start = x * self._cell_size
                self._render_buffer[y_start:y_start + self._cell_size,
                                x_start:x_start + self._cell_size] = color

        # Draw door numbers
        for door in self.doors:
            center_y = int((door.y + 0.5) * self._cell_size)
            center_x = int((door.x + 0.5) * self._cell_size)
            font_scale = self._cell_size / 30.0
            thickness = max(1, int(self._cell_size / 20))
            radius = max(2, self._cell_size // 4)
            circle_color = (50, 50, 50) if door.is_open else (200, 200, 200)
            cv2.circle(self._render_buffer, (center_x, center_y), radius, circle_color, -1)
            text = str(door.door_number)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(self._render_buffer, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Draw button numbers
        for button in self.buttons:
            center_y = int((button.y + 0.5) * self._cell_size)
            center_x = int((button.x + 0.5) * self._cell_size)
            font_scale = self._cell_size / 30.0
            thickness = max(1, int(self._cell_size / 20))
            radius = max(2, self._cell_size // 5)
            circle_color = (200, 0, 0) if button.is_broken else (0, 0, 200)
            cv2.circle(self._render_buffer, (center_x, center_y), radius, circle_color, -1)
            door = self.doors[button.door_idx]
            text = str(door.door_number)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(self._render_buffer, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Draw food sources
        if self.food_sources is not None:
            for i in range(self.food_sources.shape[0]):
                y, x, delay, has_food, _ = self.food_sources[i]
                center_y = int((y + 0.5) * self._cell_size)
                center_x = int((x + 0.5) * self._cell_size)
                if has_food:
                    radius = max(1, self._cell_size // 3)
                    cv2.circle(self._render_buffer, (center_x, center_y),
                            radius, (0, 255, 0), -1)
                else:
                    # Draw countdown number for regenerating food
                    #if delay > 0:
                    # Small dark circle behind number for contrast
                    small_radius = max(1, self._cell_size // 5)
                    cv2.circle(self._render_buffer, (center_x, center_y), small_radius, (0, 0, 0), -1)
                    font_scale = self._cell_size / 40.0
                    thickness = max(1, int(self._cell_size / 30))
                    text = str(delay)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = center_x - text_size[0] // 2
                    text_y = center_y + text_size[1] // 2
                    cv2.putText(self._render_buffer, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Draw agent
        ay, ax = int(self.agent_pos[0]), int(self.agent_pos[1])
        center_y = int((ay + 0.5) * self._cell_size)
        center_x = int((ax + 0.5) * self._cell_size)
        radius = max(1, self._cell_size // 2)
        cv2.circle(self._render_buffer, (center_x, center_y),
                radius, (255, 255, 255), -1)

        # Draw info text
        info = f"Energy: {self.energy:.1f} | Step: {self.steps}/{self.max_steps}"
        info += f" | Task: {self.task_class} (Lvl: {self.complexity_level:.1f})"
        info_doors = f"Doors: {len(self.doors)} | Buttons: {len(self.buttons)}"
        cv2.putText(self._render_buffer, info, (10, 15), cv2.QT_FONT_NORMAL, 0.55, (255, 255, 255), 1)
        cv2.putText(self._render_buffer, info_doors, (10, 35), cv2.QT_FONT_NORMAL, 0.55, (255, 255, 255), 1)

        return self._render_buffer

class VectorGridMazeWorld(GridMazeWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._render_buffer = None

    def render(self):
        return None

    def close(self):
        pass
