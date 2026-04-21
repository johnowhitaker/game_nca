from __future__ import annotations

import argparse
import json

import torch

from .device import device_report, pick_device
from .maze import generate_maze_dataset
from .train_maze import evaluate_model, make_model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a maze NCA checkpoint.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--size", type=int, default=31)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--nca-steps", type=int, default=92)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()
    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = make_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    dataset = generate_maze_dataset(args.size, args.count, args.seed, device)
    metrics = evaluate_model(model, dataset, cfg, args.count, args.nca_steps)
    metrics.update({"device": device_report(device), "size": args.size, "count": args.count, "nca_steps": args.nca_steps})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
