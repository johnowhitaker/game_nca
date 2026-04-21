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
from .flow import FlowDataset, generate_flow_dataset, neighbor_logits
from .nca import GameNCA, NCAConfig


DEFAULT_FLOW_CONFIG: dict[str, Any] = {
    "seed": 23,
    "size": 45,
    "train_steps": 1200,
    "batch_size": 24,
    "dataset_count": 320,
    "lr": 0.0014,
    "grad_clip": 1.0,
    "save_every": 250,
    "log_every": 25,
    "state_l2": 0.00002,
    "value_loss_weight": 1.0,
    "direction_loss_weight": 0.75,
    "wall_loss_weight": 0.04,
    "nca_steps_min": 95,
    "nca_steps_max": 155,
    "nca": {
        "state_channels": 18,
        "obs_channels": 3,
        "hidden_channels": 96,
        "update_rate": 1.0,
        "delta_scale": 0.12,
        "action_channel": 1,
        "zero_last": True,
        "seed_noise": 0.01,
    },
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
    cfg = dict(DEFAULT_FLOW_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


def make_model(cfg: dict[str, Any]) -> GameNCA:
    return GameNCA(
        NCAConfig(
            state_channels=cfg["nca"]["state_channels"],
            obs_channels=cfg["nca"]["obs_channels"],
            hidden_channels=cfg["nca"]["hidden_channels"],
            update_rate=cfg["nca"]["update_rate"],
            delta_scale=cfg["nca"]["delta_scale"],
            action_channel=cfg["nca"]["action_channel"],
            zero_last=cfg["nca"]["zero_last"],
            seed_noise=cfg["nca"]["seed_noise"],
        )
    )


def save_checkpoint(path: Path, model: GameNCA, cfg: dict[str, Any], step: int, optimizer: torch.optim.Optimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "step": step, "optimizer": optimizer.state_dict()}, path)


def flow_loss(model: GameNCA, batch, cfg: dict[str, Any], steps: int):
    state = model.seed(batch.obs.shape[0], batch.obs.shape[2], batch.obs.shape[3], batch.obs.device)
    state = model(state, batch.obs, steps=steps, update_rate=cfg["nca"]["update_rate"])
    field = state[:, cfg["nca"]["action_channel"]]
    pred = torch.sigmoid(field)
    value_weight = batch.reachable * (0.5 + 2.0 * batch.target_value)
    value_loss = ((pred - batch.target_value).square() * value_weight).sum() / value_weight.sum().clamp_min(1.0)
    wall_loss = (pred.square() * batch.walls.float()).sum() / batch.walls.float().sum().clamp_min(1.0)
    direction_loss = F.cross_entropy(neighbor_logits(field, batch.walls), batch.direction, ignore_index=-1)
    state_penalty = state.square().mean() * cfg["state_l2"]
    loss = (
        cfg["value_loss_weight"] * value_loss
        + cfg["direction_loss_weight"] * direction_loss
        + cfg["wall_loss_weight"] * wall_loss
        + state_penalty
    )
    with torch.no_grad():
        logits = neighbor_logits(field, batch.walls)
        mask = batch.direction >= 0
        direction_acc = (logits.argmax(dim=1)[mask] == batch.direction[mask]).float().mean()
    return loss, {
        "value_loss": float(value_loss.detach().cpu()),
        "direction_loss": float(direction_loss.detach().cpu()),
        "direction_acc": float(direction_acc.detach().cpu()),
        "wall_loss": float(wall_loss.detach().cpu()),
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
    dataset = generate_flow_dataset(cfg["size"], cfg["dataset_count"], cfg["seed"] * 100_000, device)
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    log_path = out_dir / "train_log.jsonl"
    start_time = time()
    print(f"training flow NCA on {device_report(device)} size={cfg['size']}")
    print(f"output: {out_dir}")
    for step in range(1, cfg["train_steps"] + 1):
        model.train()
        nca_steps = random.randint(cfg["nca_steps_min"], cfg["nca_steps_max"])
        batch = dataset.sample(cfg["batch_size"])
        loss, loss_metrics = flow_loss(model, batch, cfg, nca_steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        if step % cfg["log_every"] == 0 or step == 1:
            metrics = {
                "step": step,
                "nca_steps": nca_steps,
                "loss": float(loss.detach().cpu()),
                **loss_metrics,
                "seconds": round(time() - start_time, 3),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            print(
                f"{step:05d} loss={metrics['loss']:.3f} "
                f"dir={metrics['direction_loss']:.3f} acc={metrics['direction_acc']:.2f}"
            )
        if step % cfg["save_every"] == 0 or step == cfg["train_steps"]:
            save_checkpoint(out_dir / "checkpoints" / f"step_{step:05d}.pt", model, cfg, step, optimizer)
            save_checkpoint(out_dir / "checkpoints" / "latest.pt", model, cfg, step, optimizer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an NCA glow-garden flow field.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="runs/flow")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
