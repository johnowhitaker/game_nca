from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn.functional as F


DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class MazeExample:
    walls: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    dist: np.ndarray
    target_value: np.ndarray
    path_mask: np.ndarray
    direction: np.ndarray

    @property
    def optimal_length(self) -> int:
        return int(self.dist[self.start])


@dataclass
class MazeBatch:
    obs: torch.Tensor
    walls: torch.Tensor
    open_mask: torch.Tensor
    path_mask: torch.Tensor
    target_value: torch.Tensor
    direction: torch.Tensor
    starts: torch.Tensor
    goals: torch.Tensor
    optimal_lengths: torch.Tensor


class MazeDataset:
    def __init__(self, examples: list[MazeExample], device: torch.device):
        if not examples:
            raise ValueError("MazeDataset needs at least one example.")
        self.examples = examples
        self.device = device
        self.size = examples[0].walls.shape[0]
        self.obs = []
        self.walls = []
        self.open_mask = []
        self.path_mask = []
        self.target_value = []
        self.direction = []
        self.starts = []
        self.goals = []
        self.optimal_lengths = []
        for ex in examples:
            wall = torch.tensor(ex.walls, dtype=torch.float32)
            goal = torch.zeros_like(wall)
            start = torch.zeros_like(wall)
            goal[ex.goal] = 1.0
            start[ex.start] = 1.0
            self.obs.append(torch.stack([wall, goal, start], dim=0))
            self.walls.append(wall.bool())
            self.open_mask.append((~wall.bool()).float())
            self.path_mask.append(torch.tensor(ex.path_mask, dtype=torch.float32))
            self.target_value.append(torch.tensor(ex.target_value, dtype=torch.float32))
            self.direction.append(torch.tensor(ex.direction, dtype=torch.long))
            self.starts.append(torch.tensor(ex.start, dtype=torch.long))
            self.goals.append(torch.tensor(ex.goal, dtype=torch.long))
            self.optimal_lengths.append(torch.tensor(ex.optimal_length, dtype=torch.float32))

        self.obs = torch.stack(self.obs).to(device)
        self.walls = torch.stack(self.walls).to(device)
        self.open_mask = torch.stack(self.open_mask).to(device)
        self.path_mask = torch.stack(self.path_mask).to(device)
        self.target_value = torch.stack(self.target_value).to(device)
        self.direction = torch.stack(self.direction).to(device)
        self.starts = torch.stack(self.starts).to(device)
        self.goals = torch.stack(self.goals).to(device)
        self.optimal_lengths = torch.stack(self.optimal_lengths).to(device)

    def sample(self, batch_size: int) -> MazeBatch:
        idx = torch.randint(len(self.examples), (batch_size,), device=self.device)
        return MazeBatch(
            obs=self.obs[idx],
            walls=self.walls[idx],
            open_mask=self.open_mask[idx],
            path_mask=self.path_mask[idx],
            target_value=self.target_value[idx],
            direction=self.direction[idx],
            starts=self.starts[idx],
            goals=self.goals[idx],
            optimal_lengths=self.optimal_lengths[idx],
        )


def _neighbors(y: int, x: int, size: int) -> list[tuple[int, int, int, int]]:
    out = []
    for dy, dx in DIRS:
        ny, nx = y + 2 * dy, x + 2 * dx
        if 1 <= ny < size - 1 and 1 <= nx < size - 1:
            out.append((ny, nx, y + dy, x + dx))
    return out


def carve_maze(size: int, rng: random.Random) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Maze size must be odd so one-cell walls line up cleanly.")
    walls = np.ones((size, size), dtype=bool)
    start = (rng.randrange(1, size, 2), rng.randrange(1, size, 2))
    stack = [start]
    walls[start] = False
    seen = {start}
    while stack:
        y, x = stack[-1]
        candidates = [(ny, nx, wy, wx) for ny, nx, wy, wx in _neighbors(y, x, size) if (ny, nx) not in seen]
        if not candidates:
            stack.pop()
            continue
        ny, nx, wy, wx = rng.choice(candidates)
        walls[wy, wx] = False
        walls[ny, nx] = False
        seen.add((ny, nx))
        stack.append((ny, nx))
    return walls


def bfs_dist(walls: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    dist = np.full(walls.shape, -1, dtype=np.int32)
    q: deque[tuple[int, int]] = deque([source])
    dist[source] = 0
    while q:
        y, x = q.popleft()
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < walls.shape[0] and 0 <= nx < walls.shape[1] and not walls[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))
    return dist


def farthest_open(dist: np.ndarray, rng: random.Random, top_frac: float = 0.12) -> tuple[int, int]:
    ys, xs = np.where(dist == dist.max())
    if len(ys) > 1:
        i = rng.randrange(len(ys))
        return int(ys[i]), int(xs[i])
    finite = np.argwhere(dist >= 0)
    finite = finite[np.argsort(dist[finite[:, 0], finite[:, 1]])]
    tail = finite[max(0, int(len(finite) * (1.0 - top_frac))) :]
    y, x = tail[rng.randrange(len(tail))]
    return int(y), int(x)


def direction_labels(walls: np.ndarray, dist_to_goal: np.ndarray) -> np.ndarray:
    direction = np.full(walls.shape, -1, dtype=np.int64)
    for y, x in np.argwhere((~walls) & (dist_to_goal > 0)):
        d = dist_to_goal[y, x]
        for i, (dy, dx) in enumerate(DIRS):
            ny, nx = int(y + dy), int(x + dx)
            if 0 <= ny < walls.shape[0] and 0 <= nx < walls.shape[1] and dist_to_goal[ny, nx] == d - 1:
                direction[y, x] = i
                break
    return direction


def generate_maze_example(size: int, seed: int) -> MazeExample:
    rng = random.Random(seed)
    walls = carve_maze(size, rng)
    open_cells = np.argwhere(~walls)
    y, x = open_cells[rng.randrange(len(open_cells))]
    a = farthest_open(bfs_dist(walls, (int(y), int(x))), rng)
    b = farthest_open(bfs_dist(walls, a), rng)
    start, goal = a, b
    dist = bfs_dist(walls, goal)
    start_dist = bfs_dist(walls, start)
    max_dist = max(1, int(dist[start]))
    path_mask = ((start_dist + dist) == max_dist) & (~walls)
    target = np.zeros_like(dist, dtype=np.float32)
    target[path_mask] = 1.0 - np.minimum(dist[path_mask], max_dist) / max_dist
    direction = direction_labels(walls, dist)
    direction[~path_mask] = -1
    return MazeExample(
        walls=walls,
        start=start,
        goal=goal,
        dist=dist,
        target_value=target,
        path_mask=path_mask,
        direction=direction,
    )


def generate_maze_dataset(size: int, count: int, seed: int, device: torch.device) -> MazeDataset:
    examples = [generate_maze_example(size, seed + i * 9973) for i in range(count)]
    return MazeDataset(examples, device=device)


def neighbor_logits(field: torch.Tensor, walls: torch.Tensor) -> torch.Tensor:
    padded = F.pad(field[:, None], (1, 1, 1, 1), value=-1.0e4)[:, 0]
    wall_pad = F.pad(walls[:, None].float(), (1, 1, 1, 1), value=1.0)[:, 0].bool()
    logits = torch.stack(
        [
            padded[:, :-2, 1:-1],
            padded[:, 2:, 1:-1],
            padded[:, 1:-1, :-2],
            padded[:, 1:-1, 2:],
        ],
        dim=1,
    )
    neighbor_walls = torch.stack(
        [
            wall_pad[:, :-2, 1:-1],
            wall_pad[:, 2:, 1:-1],
            wall_pad[:, 1:-1, :-2],
            wall_pad[:, 1:-1, 2:],
        ],
        dim=1,
    )
    return logits.masked_fill(neighbor_walls, -1.0e4)


def solve_with_field(
    field: torch.Tensor,
    walls: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    max_moves: int,
) -> tuple[bool, list[tuple[int, int]]]:
    field_cpu = field.detach().cpu()
    walls_cpu = walls.detach().cpu().bool()
    y, x = int(start[0].detach().cpu()), int(start[1].detach().cpu())
    gy, gx = int(goal[0].detach().cpu()), int(goal[1].detach().cpu())
    path = [(y, x)]
    seen = {(y, x)}
    for _ in range(max_moves):
        if (y, x) == (gy, gx):
            return True, path
        candidates: list[tuple[float, int, int]] = []
        unvisited: list[tuple[float, int, int]] = []
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < walls_cpu.shape[0] and 0 <= nx < walls_cpu.shape[1] and not bool(walls_cpu[ny, nx]):
                item = (float(field_cpu[ny, nx]), ny, nx)
                candidates.append(item)
                if (ny, nx) not in seen:
                    unvisited.append(item)
        if not candidates:
            return False, path
        choices = unvisited if unvisited else candidates
        choices.sort(reverse=True)
        _, y, x = choices[0]
        path.append((y, x))
        seen.add((y, x))
    return (y, x) == (gy, gx), path
