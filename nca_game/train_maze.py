from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import time
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from .device import device_report, pick_device
from .maze import MazeDataset, generate_maze_dataset, neighbor_logits, solve_with_field
from .nca import GameNCA, NCAConfig


DEFAULT_MAZE_CONFIG: dict[str, Any] = {
    "seed": 11,
    "train_steps": 4200,
    "batch_size": 24,
    "dataset_count": 384,
    "lr": 0.0015,
    "grad_clip": 1.0,
    "save_every": 250,
    "log_every": 25,
    "state_l2": 0.00002,
    "value_loss_weight": 1.0,
    "direction_loss_weight": 1.35,
    "wall_loss_weight": 0.05,
    "nca_steps_min": 24,
    "nca_steps_max": 54,
    "eval_mazes": 64,
    "eval_steps": 76,
    "nca": {
        "state_channels": 16,
        "obs_channels": 3,
        "hidden_channels": 80,
        "update_rate": 1.0,
        "delta_scale": 0.14,
        "action_channel": 1,
        "zero_last": True,
        "seed_noise": 0.01,
    },
    "curriculum": [
        {"until": 2000, "size": 15, "nca_steps_min": 48, "nca_steps_max": 86},
        {"until": 4200, "size": 21, "nca_steps_min": 86, "nca_steps_max": 145},
        {"until": 7000, "size": 31, "nca_steps_min": 155, "nca_steps_max": 260},
    ],
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_MAZE_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


def stage_for_step(cfg: dict[str, Any], step: int) -> dict[str, Any]:
    for stage in cfg["curriculum"]:
        if step <= stage["until"]:
            return stage
    return cfg["curriculum"][-1]


def save_checkpoint(path: Path, model: GameNCA, cfg: dict[str, Any], step: int, optimizer: torch.optim.Optimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "step": step, "optimizer": optimizer.state_dict()}, path)


def make_model(cfg: dict[str, Any]) -> GameNCA:
    nca_cfg = NCAConfig(
        state_channels=cfg["nca"]["state_channels"],
        obs_channels=cfg["nca"]["obs_channels"],
        hidden_channels=cfg["nca"]["hidden_channels"],
        update_rate=cfg["nca"]["update_rate"],
        delta_scale=cfg["nca"]["delta_scale"],
        action_channel=cfg["nca"]["action_channel"],
        zero_last=cfg["nca"]["zero_last"],
        seed_noise=cfg["nca"]["seed_noise"],
    )
    return GameNCA(nca_cfg)


def maze_loss(model: GameNCA, batch, cfg: dict[str, Any], steps: int) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    state = model.seed(batch.obs.shape[0], batch.obs.shape[2], batch.obs.shape[3], batch.obs.device)
    state = model(state, batch.obs, steps=steps, update_rate=cfg["nca"]["update_rate"])
    field = state[:, cfg["nca"]["action_channel"]]
    pred = torch.sigmoid(field)
    non_path = (batch.open_mask - batch.path_mask).clamp_min(0.0)
    open_weight = batch.path_mask * (5.0 + 3.0 * batch.target_value) + non_path * 0.28
    value_loss = ((pred - batch.target_value).square() * open_weight).sum() / open_weight.sum().clamp_min(1.0)
    wall_loss = (pred.square() * batch.walls.float()).sum() / batch.walls.float().sum().clamp_min(1.0)
    direction_loss = F.cross_entropy(neighbor_logits(field, batch.walls), batch.direction, ignore_index=-1)
    state_penalty = state.square().mean() * cfg["state_l2"]
    loss = (
        cfg["value_loss_weight"] * value_loss
        + cfg["direction_loss_weight"] * direction_loss
        + cfg["wall_loss_weight"] * wall_loss
        + state_penalty
    )
    metrics = {
        "value_loss": float(value_loss.detach().cpu()),
        "direction_loss": float(direction_loss.detach().cpu()),
        "wall_loss": float(wall_loss.detach().cpu()),
    }
    return loss, metrics, field.detach()


@torch.no_grad()
def evaluate_model(model: GameNCA, dataset: MazeDataset, cfg: dict[str, Any], count: int, steps: int) -> dict[str, float]:
    model.eval()
    successes = 0
    path_lengths = []
    optimal_lengths = []
    batch = dataset.sample(count)
    state = model.seed(count, batch.obs.shape[2], batch.obs.shape[3], batch.obs.device)
    state = model(state, batch.obs, steps=steps, update_rate=cfg["nca"]["update_rate"])
    field = state[:, cfg["nca"]["action_channel"]]
    for i in range(count):
        max_moves = int(batch.optimal_lengths[i].detach().cpu().item() * 3 + 24)
        solved, path = solve_with_field(field[i], batch.walls[i], batch.starts[i], batch.goals[i], max_moves=max_moves)
        successes += int(solved)
        path_lengths.append(len(path) - 1)
        optimal_lengths.append(float(batch.optimal_lengths[i].detach().cpu()))
    solved_lengths = [p / max(1.0, o) for p, o in zip(path_lengths, optimal_lengths)]
    return {
        "success_rate": successes / count,
        "mean_path_length": float(sum(path_lengths) / len(path_lengths)),
        "mean_optimal_length": float(sum(optimal_lengths) / len(optimal_lengths)),
        "mean_path_ratio": float(sum(solved_lengths) / len(solved_lengths)),
    }


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train_steps"] = args.steps
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = pick_device(args.device)
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    sizes = sorted({int(stage["size"]) for stage in cfg["curriculum"]})
    datasets = {
        size: generate_maze_dataset(size, cfg["dataset_count"], cfg["seed"] * 100_000 + size * 1000, device)
        for size in sizes
    }
    log_path = out_dir / "train_log.jsonl"
    start_time = time()
    print(f"training maze NCA on {device_report(device)} sizes={sizes}")
    print(f"output: {out_dir}")

    for step in range(1, cfg["train_steps"] + 1):
        model.train()
        stage = stage_for_step(cfg, step)
        dataset = datasets[int(stage["size"])]
        nca_steps = random.randint(int(stage.get("nca_steps_min", cfg["nca_steps_min"])), int(stage.get("nca_steps_max", cfg["nca_steps_max"])))
        batch = dataset.sample(cfg["batch_size"])
        loss, loss_metrics, _field = maze_loss(model, batch, cfg, nca_steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        if step % cfg["log_every"] == 0 or step == 1:
            eval_metrics = evaluate_model(model, dataset, cfg, min(cfg["eval_mazes"], len(dataset.examples)), nca_steps)
            metrics = {
                "step": step,
                "size": int(stage["size"]),
                "nca_steps": nca_steps,
                "loss": float(loss.detach().cpu()),
                **loss_metrics,
                **eval_metrics,
                "seconds": round(time() - start_time, 3),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            print(
                f"{step:05d} size={metrics['size']} loss={metrics['loss']:.3f} "
                f"dir={metrics['direction_loss']:.3f} success={metrics['success_rate']:.2f}"
            )

        if step % cfg["save_every"] == 0 or step == cfg["train_steps"]:
            save_checkpoint(out_dir / "checkpoints" / f"step_{step:05d}.pt", model, cfg, step, optimizer)
            save_checkpoint(out_dir / "checkpoints" / "latest.pt", model, cfg, step, optimizer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an NCA to solve mazes by growing a value field.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="runs/maze")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
