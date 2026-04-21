from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn.functional as F

from .maze import DIRS


@dataclass(frozen=True)
class FlowExample:
    walls: np.ndarray
    goals: np.ndarray
    sources: np.ndarray
    reachable: np.ndarray
    dist: np.ndarray
    target_value: np.ndarray
    direction: np.ndarray


@dataclass
class FlowBatch:
    obs: torch.Tensor
    walls: torch.Tensor
    open_mask: torch.Tensor
    reachable: torch.Tensor
    target_value: torch.Tensor
    direction: torch.Tensor


class FlowDataset:
    def __init__(self, examples: list[FlowExample], device: torch.device):
        self.examples = examples
        self.device = device
        self.size = examples[0].walls.shape[0]
        obs, walls, open_mask, reachable, target_value, direction = [], [], [], [], [], []
        for ex in examples:
            wall = torch.tensor(ex.walls, dtype=torch.float32)
            goal = torch.tensor(ex.goals, dtype=torch.float32)
            source = torch.tensor(ex.sources, dtype=torch.float32)
            obs.append(torch.stack([wall, goal, source], dim=0))
            walls.append(wall.bool())
            open_mask.append((~wall.bool()).float())
            reachable.append(torch.tensor(ex.reachable, dtype=torch.float32))
            target_value.append(torch.tensor(ex.target_value, dtype=torch.float32))
            direction.append(torch.tensor(ex.direction, dtype=torch.long))
        self.obs = torch.stack(obs).to(device)
        self.walls = torch.stack(walls).to(device)
        self.open_mask = torch.stack(open_mask).to(device)
        self.reachable = torch.stack(reachable).to(device)
        self.target_value = torch.stack(target_value).to(device)
        self.direction = torch.stack(direction).to(device)

    def sample(self, batch_size: int) -> FlowBatch:
        idx = torch.randint(len(self.examples), (batch_size,), device=self.device)
        return FlowBatch(
            obs=self.obs[idx],
            walls=self.walls[idx],
            open_mask=self.open_mask[idx],
            reachable=self.reachable[idx],
            target_value=self.target_value[idx],
            direction=self.direction[idx],
        )


def smooth_cave(size: int, rng: random.Random, fill_prob: float = 0.42, rounds: int = 4) -> np.ndarray:
    walls = np.array([[rng.random() < fill_prob for _ in range(size)] for _ in range(size)], dtype=bool)
    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True
    kernel = np.ones((1, 1, 3, 3), dtype=np.float32)
    x = torch.tensor(walls.astype(np.float32))[None, None]
    k = torch.tensor(kernel)
    for _ in range(rounds):
        n = F.conv2d(F.pad(x, (1, 1, 1, 1), value=1.0), k)[0, 0].numpy()
        walls = n >= 5
        walls[[0, -1], :] = True
        walls[:, [0, -1]] = True
        x = torch.tensor(walls.astype(np.float32))[None, None]
    return walls


def connected_components(walls: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(walls, dtype=bool)
    comps = []
    h, w = walls.shape
    for sy, sx in np.argwhere(~walls):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        cells = []
        q = deque([(sy, sx)])
        seen[sy, sx] = True
        while q:
            y, x = q.popleft()
            cells.append((y, x))
            for dy, dx in DIRS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not walls[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        comps.append(np.array(cells, dtype=np.int64))
    comps.sort(key=len, reverse=True)
    return comps


def multi_source_bfs(walls: np.ndarray, goals: np.ndarray) -> np.ndarray:
    dist = np.full(walls.shape, -1, dtype=np.int32)
    q: deque[tuple[int, int]] = deque()
    for y, x in np.argwhere(goals):
        y, x = int(y), int(x)
        dist[y, x] = 0
        q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < walls.shape[0] and 0 <= nx < walls.shape[1] and not walls[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))
    return dist


def direction_labels(walls: np.ndarray, dist: np.ndarray) -> np.ndarray:
    labels = np.full(walls.shape, -1, dtype=np.int64)
    for y, x in np.argwhere((~walls) & (dist > 0)):
        d = dist[y, x]
        best = []
        for i, (dy, dx) in enumerate(DIRS):
            ny, nx = int(y + dy), int(x + dx)
            if 0 <= ny < walls.shape[0] and 0 <= nx < walls.shape[1] and dist[ny, nx] >= 0:
                if dist[ny, nx] == d - 1:
                    best.append(i)
        if best:
            labels[y, x] = best[0]
    return labels


def generate_flow_example(size: int, seed: int, goals_n: int = 4, sources_n: int = 6) -> FlowExample:
    rng = random.Random(seed)
    for attempt in range(80):
        walls = smooth_cave(size, rng, fill_prob=rng.uniform(0.38, 0.46), rounds=rng.randint(3, 5))
        comps = connected_components(walls)
        if not comps or len(comps[0]) < size * size * 0.35:
            continue
        cells = comps[0]
        # Keep tiny isolated pockets closed; this makes the visual flow readable.
        keep = np.zeros_like(walls, dtype=bool)
        keep[cells[:, 0], cells[:, 1]] = True
        walls = ~keep
        walls[[0, -1], :] = True
        walls[:, [0, -1]] = True
        cells = np.argwhere(~walls)
        if len(cells) < goals_n + sources_n + 20:
            continue
        center = np.array([(size - 1) / 2.0, (size - 1) / 2.0])
        dcenter = np.linalg.norm(cells - center[None], axis=1)
        goal_pool = cells[np.argsort(dcenter)[: max(goals_n * 6, goals_n)]]
        source_pool = cells[np.argsort(dcenter)[-max(sources_n * 8, sources_n) :]]
        rng.shuffle(goal_pool)
        rng.shuffle(source_pool)
        goals = np.zeros((size, size), dtype=bool)
        sources = np.zeros((size, size), dtype=bool)
        for y, x in goal_pool[:goals_n]:
            goals[int(y), int(x)] = True
        for y, x in source_pool[:sources_n]:
            sources[int(y), int(x)] = True
        dist = multi_source_bfs(walls, goals)
        reachable = dist >= 0
        if not np.all(reachable[sources]):
            continue
        max_dist = max(1, int(dist[reachable].max()))
        target = np.zeros_like(dist, dtype=np.float32)
        target[reachable] = 1.0 - dist[reachable] / max_dist
        direction = direction_labels(walls, dist)
        return FlowExample(
            walls=walls,
            goals=goals,
            sources=sources,
            reachable=reachable,
            dist=dist,
            target_value=target,
            direction=direction,
        )
    raise RuntimeError(f"Failed to generate connected flow map size={size}")


def generate_flow_dataset(size: int, count: int, seed: int, device: torch.device) -> FlowDataset:
    return FlowDataset([generate_flow_example(size, seed + i * 7919) for i in range(count)], device=device)


def neighbor_logits(field: torch.Tensor, walls: torch.Tensor) -> torch.Tensor:
    padded = F.pad(field[:, None], (1, 1, 1, 1), value=-1.0e4)[:, 0]
    wall_pad = F.pad(walls[:, None].float(), (1, 1, 1, 1), value=1.0)[:, 0].bool()
    logits = torch.stack(
        [padded[:, :-2, 1:-1], padded[:, 2:, 1:-1], padded[:, 1:-1, :-2], padded[:, 1:-1, 2:]],
        dim=1,
    )
    neighbor_walls = torch.stack(
        [wall_pad[:, :-2, 1:-1], wall_pad[:, 2:, 1:-1], wall_pad[:, 1:-1, :-2], wall_pad[:, 1:-1, 2:]],
        dim=1,
    )
    return logits.masked_fill(neighbor_walls, -1.0e4)
