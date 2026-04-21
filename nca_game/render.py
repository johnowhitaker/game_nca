from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .device import device_report, pick_device
from .train import make_game, render_rollout
from .nca import GameNCA, NCAConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained NCA game rollout.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    if args.steps is not None:
        cfg["render_steps"] = args.steps
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
    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent.parent / "videos"
    path = render_rollout(model, game, cfg, device, out_dir, "render")
    print(f"rendered {path} on {device_report(device)}")


if __name__ == "__main__":
    main()
