from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch

from .device import pick_device, device_report
from .maze import DIRS, generate_maze_example, solve_with_field
from .nca import NCAConfig, GameNCA
from .train_maze import make_model
from .viz import _norm_signed, write_video


def example_to_obs(example, device: torch.device) -> torch.Tensor:
    wall = torch.tensor(example.walls, dtype=torch.float32, device=device)
    goal = torch.zeros_like(wall)
    start = torch.zeros_like(wall)
    goal[example.goal] = 1.0
    start[example.start] = 1.0
    return torch.stack([wall, goal, start], dim=0)[None]


def maze_rgb(example, agent: tuple[int, int], path: list[tuple[int, int]], field: torch.Tensor | None = None) -> np.ndarray:
    walls = example.walls
    h, w = walls.shape
    rgb = np.ones((h, w, 3), dtype=np.float32) * 0.93
    rgb[walls] = np.array([0.05, 0.06, 0.08])
    if field is not None:
        f = torch.sigmoid(field).detach().cpu().numpy()
        f = (f - f[~walls].min()) / max(1e-6, f[~walls].max() - f[~walls].min())
        rgb[~walls] = 0.65 * rgb[~walls] + 0.35 * np.stack([f[~walls], 0.3 + 0.5 * f[~walls], 1.0 - f[~walls]], axis=1)
    for y, x in path:
        rgb[y, x] = np.array([1.0, 0.72, 0.18])
    rgb[example.start] = np.array([0.2, 0.65, 1.0])
    rgb[example.goal] = np.array([0.1, 0.85, 0.35])
    rgb[agent] = np.array([1.0, 0.15, 0.1])
    return np.repeat(np.repeat(np.clip(rgb, 0.0, 1.0), 10, axis=0), 10, axis=1)


def maze_frame(
    example,
    state: torch.Tensor,
    field: torch.Tensor,
    target: torch.Tensor,
    agent: tuple[int, int],
    path: list[tuple[int, int]],
    title: str,
) -> np.ndarray:
    fig = plt.figure(figsize=(13.0, 7.2), dpi=110)
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 6, height_ratios=[1.25, 1.0, 1.0])

    ax = fig.add_subplot(grid[0, 0:2])
    ax.imshow(maze_rgb(example, agent, path, field), interpolation="nearest")
    ax.set_title("maze, value overlay, agent path", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 2:4])
    arr = field.detach().cpu().numpy()
    ax.imshow(arr, cmap="magma", interpolation="nearest")
    ax.scatter([example.goal[1]], [example.goal[0]], c="lime", s=16)
    ax.scatter([agent[1]], [agent[0]], c="cyan", s=16)
    ax.set_title("learned pheromone/value channel", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 4:6])
    tgt = target.detach().cpu().numpy()
    ax.imshow(tgt, cmap="viridis", interpolation="nearest")
    ax.scatter([example.goal[1]], [example.goal[0]], c="white", s=16)
    ax.set_title("BFS training target", fontsize=10)
    ax.axis("off")

    for i in range(min(12, state.shape[0])):
        ax = fig.add_subplot(grid[1 + i // 6, i % 6])
        ax.imshow(_norm_signed(state[i]), cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"state {i}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(pad=0.7)
    canvas.draw()
    width, height = canvas.get_width_height()
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    frame = rgba[:, :, :3].copy()
    plt.close(fig)
    return frame


@torch.no_grad()
def choose_next(field: torch.Tensor, walls: np.ndarray, pos: tuple[int, int], seen: set[tuple[int, int]]) -> tuple[int, int]:
    y, x = pos
    scores = []
    unvisited = []
    field_cpu = field.detach().cpu()
    for dy, dx in DIRS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < walls.shape[0] and 0 <= nx < walls.shape[1] and not walls[ny, nx]:
            item = (float(field_cpu[ny, nx]), ny, nx)
            scores.append(item)
            if (ny, nx) not in seen:
                unvisited.append(item)
    if not scores:
        return pos
    choices = unvisited if unvisited else scores
    choices.sort(reverse=True)
    return choices[0][1], choices[0][2]


@torch.no_grad()
def score_candidate(model: GameNCA, cfg: dict, device: torch.device, size: int, seed: int, nca_steps: int):
    example = generate_maze_example(size, seed)
    obs = example_to_obs(example, device)
    state = model.seed(1, size, size, device)
    state = model(state, obs, steps=nca_steps, update_rate=cfg["nca"]["update_rate"])
    field = state[0, cfg["nca"]["action_channel"]]
    solved, path = solve_with_field(
        field,
        torch.tensor(example.walls, device=device),
        torch.tensor(example.start, device=device),
        torch.tensor(example.goal, device=device),
        max_moves=example.optimal_length * 3 + 32,
    )
    return SimpleNamespace(example=example, solved=solved, path=path, seed=seed, field=field, state=state)


@torch.no_grad()
def render(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = make_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rng = random.Random(args.seed)
    seeds = [rng.randrange(1, 2_000_000_000) for _ in range(args.candidates)]
    scored = [score_candidate(model, cfg, device, args.size, seed, args.nca_steps) for seed in seeds]
    scored.sort(key=lambda x: (int(x.solved), x.example.optimal_length), reverse=True)
    best = scored[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": args.checkpoint,
        "size": args.size,
        "nca_steps": args.nca_steps,
        "candidates": [
            {
                "seed": s.seed,
                "solved": bool(s.solved),
                "path_length": len(s.path) - 1,
                "optimal_length": s.example.optimal_length,
            }
            for s in scored
        ],
        "best_seed": best.seed,
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    example = best.example
    obs = example_to_obs(example, device)
    target = torch.tensor(example.target_value, dtype=torch.float32, device=device)
    state = model.seed(1, args.size, args.size, device)
    agent = example.start
    path = [agent]
    seen = {agent}
    frames = []
    move_every = max(1, args.move_every)
    total_steps = args.nca_steps + max(args.extra_steps, example.optimal_length * move_every * 4 + 60)
    for t in range(total_steps):
        state = model(state, obs, steps=1, update_rate=cfg["nca"]["update_rate"])
        field = state[0, cfg["nca"]["action_channel"]]
        if t >= args.nca_steps and (t - args.nca_steps) % move_every == 0 and agent != example.goal:
            agent = choose_next(field, example.walls, agent, seen)
            path.append(agent)
            seen.add(agent)
        if t % args.capture_every == 0:
            solved = agent == example.goal
            title = (
                f"Maze NCA | seed {best.seed} | size {args.size} | tick {t} | "
                f"path {len(path)-1}/{example.optimal_length} | solved {solved}"
            )
            frames.append(maze_frame(example, state[0], field, target, agent, path, title))
        if agent == example.goal and t > args.nca_steps + 20:
            break
    if path[-1] == example.goal:
        title = (
            f"Maze NCA | seed {best.seed} | size {args.size} | solved | "
            f"path {len(path)-1}/{example.optimal_length}"
        )
        frames.append(maze_frame(example, state[0], field, target, agent, path, title))
    path_written = write_video(frames, out, fps=args.fps)
    print(f"rendered {path_written} on {device_report(device)}")
    print(json.dumps(metadata, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a maze-solving NCA rollout.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--out", type=str, default="runs/maze/videos/maze_rollout.mp4")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--size", type=int, default=31)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--nca-steps", type=int, default=92)
    parser.add_argument("--extra-steps", type=int, default=120)
    parser.add_argument("--capture-every", type=int, default=2)
    parser.add_argument("--move-every", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24)
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
