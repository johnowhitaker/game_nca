from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch

from .device import device_report, pick_device
from .flow import DIRS, generate_flow_example
from .train_flow import make_model
from .viz import _norm_signed, write_video


def obs_from_example(example, device: torch.device) -> torch.Tensor:
    wall = torch.tensor(example.walls, dtype=torch.float32, device=device)
    goal = torch.tensor(example.goals, dtype=torch.float32, device=device)
    source = torch.tensor(example.sources, dtype=torch.float32, device=device)
    return torch.stack([wall, goal, source], dim=0)[None]


def init_particles(example, count: int, rng: random.Random) -> list[tuple[int, int]]:
    sources = [tuple(map(int, p)) for p in np.argwhere(example.sources)]
    return [sources[rng.randrange(len(sources))] for _ in range(count)]


def step_particle(example, field: torch.Tensor, pos: tuple[int, int], rng: random.Random) -> tuple[int, int]:
    y, x = pos
    if example.goals[y, x]:
        return pos
    field_cpu = field.detach().cpu()
    candidates = []
    for dy, dx in DIRS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < example.walls.shape[0] and 0 <= nx < example.walls.shape[1] and not example.walls[ny, nx]:
            candidates.append((float(field_cpu[ny, nx]) + rng.uniform(-0.02, 0.02), ny, nx))
    if not candidates:
        return pos
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


def garden_rgb(example, field: torch.Tensor, trail: np.ndarray, particles: list[tuple[int, int]]) -> np.ndarray:
    walls = example.walls
    f = torch.sigmoid(field).detach().cpu().numpy()
    if np.any(~walls):
        f = (f - f[~walls].min()) / max(1e-6, f[~walls].max() - f[~walls].min())
    rgb = np.zeros((*walls.shape, 3), dtype=np.float32)
    rgb[~walls] = np.array([0.04, 0.05, 0.075])
    rgb[walls] = np.array([0.005, 0.008, 0.012])
    rgb[~walls, 0] += 0.10 * f[~walls]
    rgb[~walls, 1] += 0.35 * f[~walls]
    rgb[~walls, 2] += 0.85 * f[~walls]
    rgb[..., 0] += 0.95 * trail
    rgb[..., 1] += 0.45 * trail
    for y, x in np.argwhere(example.sources):
        rgb[y, x] = np.array([0.25, 0.65, 1.0])
    for y, x in np.argwhere(example.goals):
        rgb[y, x] = np.array([0.2, 1.0, 0.35])
    for y, x in particles:
        rgb[y, x] = np.array([1.0, 0.93, 0.25])
    return np.repeat(np.repeat(np.clip(rgb, 0, 1), 9, axis=0), 9, axis=1)


def frame_image(example, state: torch.Tensor, field: torch.Tensor, trail: np.ndarray, particles, title: str) -> np.ndarray:
    fig = plt.figure(figsize=(13.0, 7.2), dpi=110)
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 6, height_ratios=[1.25, 1.0, 1.0])
    ax = fig.add_subplot(grid[0, 0:2])
    ax.imshow(garden_rgb(example, field, trail, particles), interpolation="nearest")
    ax.set_title("glow garden particles", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 2:4])
    ax.imshow(field.detach().cpu().numpy(), cmap="magma", interpolation="nearest")
    ax.set_title("learned flower potential", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 4:6])
    ax.imshow(example.target_value, cmap="viridis", interpolation="nearest")
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
    out = rgba[:, :, :3].copy()
    plt.close(fig)
    return out


@torch.no_grad()
def render(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = make_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rng = random.Random(args.seed)
    candidate_seeds = [args.seed]
    if args.candidates > 1:
        candidate_seeds = [rng.randrange(1, 2_000_000_000) for _ in range(args.candidates)]

    def score_seed(seed: int) -> tuple[int, int, int]:
        local_rng = random.Random(seed + 1009)
        ex = generate_flow_example(args.size, seed)
        local_obs = obs_from_example(ex, device)
        local_state = model.seed(1, args.size, args.size, device)
        local_particles = init_particles(ex, args.particles, local_rng)
        visits = 0
        for t in range(args.steps):
            local_state = model(local_state, local_obs, steps=1, update_rate=cfg["nca"]["update_rate"])
            if t >= args.warmup:
                field = local_state[0, cfg["nca"]["action_channel"]]
                for i, pos in enumerate(local_particles):
                    if ex.goals[pos]:
                        visits += 1
                        local_particles[i] = init_particles(ex, 1, local_rng)[0]
                    else:
                        local_particles[i] = step_particle(ex, field, pos, local_rng)
        open_cells = int((~ex.walls).sum())
        return visits, open_cells, seed

    scores = [score_seed(seed) for seed in candidate_seeds]
    scores.sort(reverse=True)
    example_seed = scores[0][2]
    example = generate_flow_example(args.size, example_seed)
    obs = obs_from_example(example, device)
    state = model.seed(1, args.size, args.size, device)
    particles = init_particles(example, args.particles, rng)
    trail = np.zeros((args.size, args.size), dtype=np.float32)
    frames = []
    hits = 0
    for t in range(args.steps):
        state = model(state, obs, steps=1, update_rate=cfg["nca"]["update_rate"])
        field = state[0, cfg["nca"]["action_channel"]]
        if t >= args.warmup:
            for i, pos in enumerate(particles):
                if example.goals[pos]:
                    hits += 1
                    particles[i] = init_particles(example, 1, rng)[0]
                else:
                    particles[i] = step_particle(example, field, pos, rng)
                trail[particles[i]] = min(1.0, trail[particles[i]] + 0.18)
            trail *= 0.965
        if t % args.capture_every == 0:
            title = f"Glow Garden NCA | tick {t} | particle flower visits {hits}"
            frames.append(frame_image(example, state[0], field, trail, particles, title))
    out = Path(args.out)
    path = write_video(frames, out, fps=args.fps)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "seed": example_seed,
                "size": args.size,
                "flower_visits": hits,
                "candidate_scores": [
                    {"seed": seed, "visits": visits, "open_cells": open_cells}
                    for visits, open_cells, seed in scores
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"rendered {path} on {device_report(device)} visits={hits}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a glow garden NCA particle rollout.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--out", type=str, default="runs/flow/videos/glow_garden.mp4")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--size", type=int, default=45)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--warmup", type=int, default=125)
    parser.add_argument("--particles", type=int, default=90)
    parser.add_argument("--capture-every", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24)
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
