from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AxisReadout:
    coord: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor
    action_map: torch.Tensor


def axis_coords(length: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, length, device=device)


def smooth_axis_readout(
    state: torch.Tensor,
    channel: int,
    axis: str,
    region: tuple[float, float],
    temperature: float = 0.18,
    spatial_temperature: float = 0.25,
) -> AxisReadout:
    """Turn one bright NCA channel into a differentiable x or y coordinate.

    First we take a smooth max over the non-control axis inside a region, then a
    soft argmax over the control axis. For Pong this means "find the brightest y
    on the right side of the automaton."
    """
    if axis not in {"x", "y"}:
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")

    action_map = state[:, channel]
    batch, height, width = action_map.shape
    lo, hi = region

    if axis == "y":
        x0 = max(0, min(width - 1, int(round(lo * width))))
        x1 = max(x0 + 1, min(width, int(round(hi * width))))
        region_map = action_map[:, :, x0:x1]
        logits = torch.logsumexp(region_map / spatial_temperature, dim=2) * spatial_temperature
        coords = axis_coords(height, state.device)
    else:
        y0 = max(0, min(height - 1, int(round(lo * height))))
        y1 = max(y0 + 1, min(height, int(round(hi * height))))
        region_map = action_map[:, y0:y1, :]
        logits = torch.logsumexp(region_map / spatial_temperature, dim=1) * spatial_temperature
        coords = axis_coords(width, state.device)

    probs = F.softmax(logits / temperature, dim=1)
    coord = torch.sum(probs * coords[None, :], dim=1)
    return AxisReadout(coord=coord, logits=logits, probs=probs, action_map=action_map)


def gaussian_axis_targets(target: torch.Tensor, length: int, sigma: float) -> torch.Tensor:
    coords = axis_coords(length, target.device)
    dist2 = (coords[None, :] - target[:, None]).square()
    targets = torch.exp(-0.5 * dist2 / (sigma * sigma))
    return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)


def axis_cross_entropy(logits: torch.Tensor, target_coord: torch.Tensor, sigma: float) -> torch.Tensor:
    target_dist = gaussian_axis_targets(target_coord, logits.shape[1], sigma)
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_dist * log_probs).sum(dim=1)
