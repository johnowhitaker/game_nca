from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import time
from typing import Any

import torch
import yaml

from .device import device_report, pick_device
from .games import CatchGame, PongGame
from .nca import GameNCA, NCAConfig
from .readout import axis_cross_entropy, smooth_axis_readout
from .viz import debug_frame, write_video


DEFAULT_CONFIG: dict[str, Any] = {
    "game": "pong",
    "height": 40,
    "width": 64,
    "seed": 7,
    "pool_size": 192,
    "batch_size": 24,
    "train_steps": 1500,
    "unroll_min": 6,
    "unroll_max": 14,
    "nca_steps_per_tick": 3,
    "reset_prob": 0.035,
    "max_age": 240,
    "lr": 0.0015,
    "grad_clip": 1.0,
    "save_every": 250,
    "log_every": 25,
    "render_every": 0,
    "render_steps": 180,
    "render_fps": 24,
    "state_l2": 0.00002,
    "axis_loss_weight": 0.7,
    "game_loss_weight": 1.0,
    "axis_sigma": 0.08,
    "action_temperature": 0.16,
    "spatial_temperature": 0.25,
    "nca": {
        "state_channels": 16,
        "hidden_channels": 96,
        "update_rate": 0.5,
        "delta_scale": 0.1,
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
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, user_cfg)
    return cfg


def make_game(cfg: dict[str, Any]):
    name = cfg["game"].lower()
    if name == "pong":
        return PongGame(height=cfg["height"], width=cfg["width"], **cfg.get("pong", {}))
    if name == "catch":
        return CatchGame(height=cfg["height"], width=cfg["width"], **cfg.get("catch", {}))
    raise ValueError(f"Unknown game: {cfg['game']}")


def save_checkpoint(path: Path, model: GameNCA, cfg: dict[str, Any], step: int, optimizer: torch.optim.Optimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "step": step,
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def reset_nca_on_episode_reset(model: GameNCA, state: torch.Tensor, previous_age: torch.Tensor, next_age: torch.Tensor) -> torch.Tensor:
    reset_mask = next_age < previous_age
    if not reset_mask.any():
        return state
    fresh = model.seed(state.shape[0], state.shape[2], state.shape[3], state.device)
    return torch.where(reset_mask[:, None, None, None], fresh, state)


@torch.no_grad()
def render_rollout(
    model: GameNCA,
    game,
    cfg: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    tag: str,
) -> Path:
    model.eval()
    height, width = cfg["height"], cfg["width"]
    state = model.seed(1, height, width, device)
    game_state = game.new_state(1, device)
    frames = []
    for t in range(cfg["render_steps"]):
        obs = game.render(game_state)
        state = model(state, obs, steps=cfg["nca_steps_per_tick"], update_rate=cfg["nca"]["update_rate"])
        readout = smooth_axis_readout(
            state,
            channel=cfg["nca"]["action_channel"],
            axis=game.action_axis,
            region=game.readout_region,
            temperature=cfg["action_temperature"],
            spatial_temperature=cfg["spatial_temperature"],
        )
        target = game.target_coord(game_state)
        if t % 2 == 0:
            frames.append(
                debug_frame(
                    obs[0],
                    state[0],
                    readout=type(readout)(
                        coord=readout.coord[0],
                        logits=readout.logits[0],
                        probs=readout.probs[0],
                        action_map=readout.action_map[0],
                    ),
                    target_coord=target[0],
                    axis=game.action_axis,
                    title=f"{game.name} rollout step {t}",
                )
            )
        previous_age = game_state.age
        next_game_state = game.step(game_state, readout.coord)
        state = reset_nca_on_episode_reset(model, state, previous_age, next_game_state.age)
        game_state = next_game_state
    path = out_dir / f"{tag}_{game.name}_rollout.mp4"
    return write_video(frames, path, fps=max(1, cfg["render_fps"] // 2))


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train_steps"] = args.steps
    if args.game is not None:
        cfg["game"] = args.game
    if args.no_render:
        cfg["render_every"] = 0
        cfg["render_steps"] = 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = pick_device(args.device)
    game = make_game(cfg)

    nca_cfg = NCAConfig(
        obs_channels=game.obs_channels,
        state_channels=cfg["nca"]["state_channels"],
        hidden_channels=cfg["nca"]["hidden_channels"],
        update_rate=cfg["nca"]["update_rate"],
        delta_scale=cfg["nca"]["delta_scale"],
        action_channel=cfg["nca"]["action_channel"],
        zero_last=cfg["nca"]["zero_last"],
        seed_noise=cfg["nca"]["seed_noise"],
    )
    model = GameNCA(nca_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    pool_state = model.seed(cfg["pool_size"], cfg["height"], cfg["width"], device).detach()
    pool_game = game.new_state(cfg["pool_size"], device)
    log_path = out_dir / "train_log.jsonl"

    print(f"training {game.name} NCA on {device_report(device)}")
    print(f"output: {out_dir}")

    start_time = time()
    for step in range(1, cfg["train_steps"] + 1):
        model.train()
        idx = torch.randint(cfg["pool_size"], (cfg["batch_size"],), device=device)
        state = pool_state[idx].detach()
        game_state = pool_game.index(idx)

        reset_mask = (torch.rand(cfg["batch_size"], device=device) < cfg["reset_prob"]) | (game_state.age > cfg["max_age"])
        if reset_mask.any():
            fresh_state = model.seed(cfg["batch_size"], cfg["height"], cfg["width"], device)
            state = torch.where(reset_mask[:, None, None, None], fresh_state, state)
            game_state = game.reset_where(game_state, reset_mask)

        unroll = random.randint(cfg["unroll_min"], cfg["unroll_max"])
        total_loss = torch.zeros((), device=device)
        total_axis = torch.zeros((), device=device)
        total_game = torch.zeros((), device=device)
        total_error = torch.zeros((), device=device)
        hits_before = game_state.hits.clone()
        misses_before = game_state.misses.clone()

        for _ in range(unroll):
            obs = game.render(game_state)
            state = model(state, obs, steps=cfg["nca_steps_per_tick"], update_rate=cfg["nca"]["update_rate"])
            readout = smooth_axis_readout(
                state,
                channel=cfg["nca"]["action_channel"],
                axis=game.action_axis,
                region=game.readout_region,
                temperature=cfg["action_temperature"],
                spatial_temperature=cfg["spatial_temperature"],
            )
            target = game.target_coord(game_state)
            axis_loss = axis_cross_entropy(readout.logits, target, cfg["axis_sigma"])
            game_loss = game.game_loss(game_state, readout.coord)
            loss_t = cfg["axis_loss_weight"] * axis_loss + cfg["game_loss_weight"] * game_loss
            urgency = game.urgency(game_state)
            total_loss = total_loss + (loss_t * urgency).mean()
            total_axis = total_axis + axis_loss.mean().detach()
            total_game = total_game + game_loss.mean().detach()
            total_error = total_error + (readout.coord - target).abs().mean().detach()
            previous_age = game_state.age
            next_game_state = game.step(game_state, readout.coord)
            state = reset_nca_on_episode_reset(model, state, previous_age, next_game_state.age)
            game_state = next_game_state

        state_penalty = state.square().mean() * cfg["state_l2"]
        loss = total_loss / unroll + state_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        pool_state[idx] = state.detach()
        pool_game.assign(idx, game_state)

        metrics = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "axis_loss": float((total_axis / unroll).cpu()),
            "game_loss": float((total_game / unroll).cpu()),
            "coord_error": float((total_error / unroll).cpu()),
            "hits": float((game_state.hits - hits_before).sum().detach().cpu()),
            "misses": float((game_state.misses - misses_before).sum().detach().cpu()),
            "seconds": round(time() - start_time, 3),
        }

        if step % cfg["log_every"] == 0 or step == 1:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            print(
                f"{step:05d} loss={metrics['loss']:.4f} "
                f"err={metrics['coord_error']:.3f} "
                f"hit/miss={metrics['hits']:.0f}/{metrics['misses']:.0f}"
            )

        if step % cfg["save_every"] == 0 or step == cfg["train_steps"]:
            save_checkpoint(out_dir / "checkpoints" / f"step_{step:05d}.pt", model, cfg, step, optimizer)
            save_checkpoint(out_dir / "checkpoints" / "latest.pt", model, cfg, step, optimizer)

        if cfg["render_every"] and step % cfg["render_every"] == 0:
            video_path = render_rollout(model, game, cfg, device, out_dir / "videos", f"step_{step:05d}")
            print(f"rendered {video_path}")

    if cfg["render_steps"] > 0:
        video_path = render_rollout(model, game, cfg, device, out_dir / "videos", "final")
        print(f"rendered {video_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a neural cellular automaton to play a tiny differentiable game.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--out", type=str, default="runs/pong", help="Output directory.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Torch device.")
    parser.add_argument("--game", type=str, default=None, choices=["pong", "catch"], help="Override game from config.")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps.")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint.")
    parser.add_argument("--no-render", action="store_true", help="Skip rollout video rendering.")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
