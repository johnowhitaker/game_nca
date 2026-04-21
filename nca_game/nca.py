from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NCAConfig:
    state_channels: int = 16
    obs_channels: int = 3
    hidden_channels: int = 96
    update_rate: float = 0.5
    delta_scale: float = 0.1
    action_channel: int = 1
    zero_last: bool = True
    seed_noise: float = 0.01


class GameNCA(nn.Module):
    """Neural cellular automaton that reads immutable game-screen channels.

    The model follows the classic NCA recipe: fixed local perception filters
    over every channel, then a shared 1x1 MLP that emits a residual update. The
    learned update only changes the internal state channels; the game pixels are
    supplied fresh at every step as extra perception channels.
    """

    def __init__(self, cfg: NCAConfig):
        super().__init__()
        self.cfg = cfg
        self.state_channels = cfg.state_channels
        self.obs_channels = cfg.obs_channels
        self.action_channel = cfg.action_channel

        filters = torch.stack(
            [
                torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                    dtype=torch.float32,
                )
                / 8.0,
                torch.tensor(
                    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                    dtype=torch.float32,
                )
                / 8.0,
                torch.tensor(
                    [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]],
                    dtype=torch.float32,
                )
                / 16.0,
            ]
        )
        self.register_buffer("filters", filters)

        perceived_channels = (cfg.state_channels + cfg.obs_channels) * len(filters)
        self.w1 = nn.Conv2d(perceived_channels, cfg.hidden_channels, 1)
        self.w2 = nn.Conv2d(cfg.hidden_channels, cfg.state_channels, 1, bias=False)
        if cfg.zero_last:
            nn.init.zeros_(self.w2.weight)

    def seed(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        x = torch.zeros(batch_size, self.state_channels, height, width, device=device)
        x[:, 0] = 1.0
        if self.cfg.seed_noise > 0:
            x[:, 1:] = self.cfg.seed_noise * torch.randn_like(x[:, 1:])
        return x

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        y = x.reshape(batch * channels, 1, height, width)
        y = F.pad(y, (1, 1, 1, 1), mode="circular")
        y = F.conv2d(y, self.filters[:, None])
        return y.reshape(batch, channels * self.filters.shape[0], height, width)

    def step(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        update_rate: float | None = None,
    ) -> torch.Tensor:
        rate = self.cfg.update_rate if update_rate is None else update_rate
        perceived = self.perceive(torch.cat([state, obs], dim=1))
        delta = self.w2(F.relu(self.w1(perceived)))
        delta = torch.tanh(delta) * self.cfg.delta_scale
        if rate < 1.0:
            mask = (torch.rand(state.shape[0], 1, state.shape[2], state.shape[3], device=state.device) < rate).float()
            delta = delta * mask
        return state + delta

    def forward(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        steps: int = 1,
        update_rate: float | None = None,
    ) -> torch.Tensor:
        for _ in range(steps):
            state = self.step(state, obs, update_rate=update_rate)
        return state
