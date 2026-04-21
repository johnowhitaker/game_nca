from __future__ import annotations

import argparse
import json

import torch

from .device import device_report, pick_device
from .nca import GameNCA, NCAConfig
from .readout import smooth_axis_readout
from .train import make_game, reset_nca_on_episode_reset


@torch.no_grad()
def evaluate(checkpoint: str, device_name: str, episodes: int, steps: int, burn_in: int) -> dict[str, float]:
    device = pick_device(device_name)
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
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
    model.load_state_dict(ckpt["model"])
    model.eval()

    state = model.seed(episodes, cfg["height"], cfg["width"], device)
    game_state = game.new_state(episodes, device)
    error_sum = torch.zeros((), device=device)
    samples = 0
    hits0 = game_state.hits.clone()
    misses0 = game_state.misses.clone()

    for t in range(steps):
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
        if t >= burn_in:
            error_sum = error_sum + (readout.coord - target).abs().sum()
            samples += episodes
        previous_age = game_state.age
        next_game_state = game.step(game_state, readout.coord)
        state = reset_nca_on_episode_reset(model, state, previous_age, next_game_state.age)
        game_state = next_game_state

    hits = (game_state.hits - hits0).sum()
    misses = (game_state.misses - misses0).sum()
    total_events = (hits + misses).clamp_min(1.0)
    return {
        "game": game.name,
        "device": device_report(device),
        "episodes": float(episodes),
        "steps": float(steps),
        "burn_in": float(burn_in),
        "mean_coord_error": float((error_sum / max(samples, 1)).cpu()),
        "hits": float(hits.cpu()),
        "misses": float(misses.cpu()),
        "hit_rate": float((hits / total_events).cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained NCA game checkpoint.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--burn-in", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.device, args.episodes, args.steps, args.burn_in), indent=2))


if __name__ == "__main__":
    main()
