from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def make_xy_grid(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx[None, :, :], yy[None, :, :]


def reflect_unit_interval(y: torch.Tensor) -> torch.Tensor:
    z = torch.remainder(y + 1.0, 4.0)
    return torch.where(z <= 2.0, z - 1.0, 3.0 - z)


@dataclass
class PongState:
    ball: torch.Tensor
    vel: torch.Tensor
    paddle_y: torch.Tensor
    age: torch.Tensor
    hits: torch.Tensor
    misses: torch.Tensor

    def index(self, idx: torch.Tensor) -> "PongState":
        return PongState(
            self.ball[idx].clone(),
            self.vel[idx].clone(),
            self.paddle_y[idx].clone(),
            self.age[idx].clone(),
            self.hits[idx].clone(),
            self.misses[idx].clone(),
        )

    def assign(self, idx: torch.Tensor, other: "PongState") -> None:
        self.ball[idx] = other.ball.detach()
        self.vel[idx] = other.vel.detach()
        self.paddle_y[idx] = other.paddle_y.detach()
        self.age[idx] = other.age.detach()
        self.hits[idx] = other.hits.detach()
        self.misses[idx] = other.misses.detach()


class PongGame:
    name = "pong"
    obs_channels = 3
    action_axis = "y"
    readout_region = (0.82, 1.0)

    def __init__(
        self,
        height: int,
        width: int,
        paddle_x: float = 0.88,
        paddle_half_height: float = 0.20,
        paddle_speed: float = 0.055,
        ball_radius: float = 0.045,
        ball_speed_min: float = 0.030,
        ball_speed_max: float = 0.047,
    ):
        self.height = height
        self.width = width
        self.paddle_x = paddle_x
        self.paddle_half_height = paddle_half_height
        self.paddle_speed = paddle_speed
        self.ball_radius = ball_radius
        self.ball_speed_min = ball_speed_min
        self.ball_speed_max = ball_speed_max

    def new_state(self, batch_size: int, device: torch.device) -> PongState:
        ball = torch.empty(batch_size, 2, device=device)
        ball[:, 0] = torch.rand(batch_size, device=device) * 0.8 - 0.55
        ball[:, 1] = torch.rand(batch_size, device=device) * 1.5 - 0.75
        speed = self.ball_speed_min + torch.rand(batch_size, device=device) * (self.ball_speed_max - self.ball_speed_min)
        direction = torch.where(torch.rand(batch_size, device=device) < 0.65, 1.0, -1.0)
        slope = torch.rand(batch_size, device=device) * 0.75 - 0.375
        vx = speed * direction
        vy = speed * slope
        vel = torch.stack([vx, vy], dim=1)
        paddle_y = torch.rand(batch_size, device=device) * 1.2 - 0.6
        zeros = torch.zeros(batch_size, device=device)
        return PongState(ball=ball, vel=vel, paddle_y=paddle_y, age=zeros, hits=zeros, misses=zeros)

    def reset_where(self, state: PongState, mask: torch.Tensor, keep_score: bool = False) -> PongState:
        fresh = self.new_state(state.ball.shape[0], state.ball.device)
        m1 = mask[:, None]
        return PongState(
            ball=torch.where(m1, fresh.ball, state.ball),
            vel=torch.where(m1, fresh.vel, state.vel),
            paddle_y=torch.where(mask, fresh.paddle_y, state.paddle_y),
            age=torch.where(mask, fresh.age, state.age),
            hits=state.hits if keep_score else torch.where(mask, fresh.hits, state.hits),
            misses=state.misses if keep_score else torch.where(mask, fresh.misses, state.misses),
        )

    def render(self, state: PongState) -> torch.Tensor:
        xx, yy = make_xy_grid(self.height, self.width, state.ball.device)
        bx = state.ball[:, 0, None, None]
        by = state.ball[:, 1, None, None]
        ball = torch.exp(-((xx - bx).square() + (yy - by).square()) / (2.0 * self.ball_radius**2))

        px = torch.tensor(self.paddle_x, device=state.ball.device)
        paddle_x = torch.exp(-((xx - px).square()) / (2.0 * 0.018**2))
        paddle_y = torch.sigmoid((self.paddle_half_height - (yy - state.paddle_y[:, None, None]).abs()) / 0.018)
        paddle = paddle_x * paddle_y

        top = torch.exp(-((yy + 1.0).abs()) / 0.025)
        bottom = torch.exp(-((yy - 1.0).abs()) / 0.025)
        left = torch.exp(-((xx + 0.96).abs()) / 0.025)
        center = 0.35 * torch.exp(-(xx.square()) / (2.0 * 0.012**2))
        walls = (top + bottom + left + center).expand_as(ball).clamp(0.0, 1.0)
        return torch.stack([ball.clamp(0.0, 1.0), paddle.clamp(0.0, 1.0), walls], dim=1)

    def target_coord(self, state: PongState) -> torch.Tensor:
        vx = state.vel[:, 0].clamp_min(1e-4)
        t = (self.paddle_x - state.ball[:, 0]) / vx
        intercept = reflect_unit_interval(state.ball[:, 1] + state.vel[:, 1] * t)
        coming_right = torch.sigmoid(state.vel[:, 0] * 100.0)
        return coming_right * intercept + (1.0 - coming_right) * state.ball[:, 1]

    def urgency(self, state: PongState) -> torch.Tensor:
        coming_right = torch.sigmoid(state.vel[:, 0] * 100.0)
        near_right = torch.sigmoid((state.ball[:, 0] - 0.15) * 5.0)
        return (0.3 + 1.7 * coming_right * near_right).detach()

    def game_loss(self, state: PongState, action_y: torch.Tensor) -> torch.Tensor:
        target = self.target_coord(state)
        track = F.smooth_l1_loss(action_y, target, reduction="none")
        near = self.urgency(state)
        miss_margin = (state.ball[:, 1] - state.paddle_y).abs() - self.paddle_half_height
        miss_pressure = F.relu(miss_margin).square() * near
        return track * near + 0.3 * miss_pressure

    def step(self, state: PongState, action_y: torch.Tensor) -> PongState:
        desired_move = torch.tanh((action_y - state.paddle_y) / 0.18) * self.paddle_speed
        paddle_y = (state.paddle_y + desired_move).clamp(-1.0 + self.paddle_half_height, 1.0 - self.paddle_half_height)

        prev_ball = state.ball
        ball = state.ball + state.vel
        vel = state.vel

        top = ball[:, 1] < -1.0 + self.ball_radius
        bottom = ball[:, 1] > 1.0 - self.ball_radius
        vertical_bounce = top | bottom
        ball_y = torch.where(top, -2.0 + 2.0 * self.ball_radius - ball[:, 1], ball[:, 1])
        ball_y = torch.where(bottom, 2.0 - 2.0 * self.ball_radius - ball_y, ball_y)
        vel_y = torch.where(vertical_bounce, -vel[:, 1], vel[:, 1])

        left = ball[:, 0] < -0.96 + self.ball_radius
        ball_x = torch.where(left, -1.92 + 2.0 * self.ball_radius - ball[:, 0], ball[:, 0])
        vel_x = torch.where(left, vel[:, 0].abs(), vel[:, 0])

        crossed_paddle = (prev_ball[:, 0] < self.paddle_x - self.ball_radius) & (ball_x >= self.paddle_x - self.ball_radius) & (vel[:, 0] > 0)
        on_paddle = (ball_y - paddle_y).abs() < self.paddle_half_height + self.ball_radius
        hit = crossed_paddle & on_paddle
        miss = crossed_paddle & (~on_paddle)

        spin = (ball_y - paddle_y).clamp(-0.35, 0.35) * 0.020
        vel_x = torch.where(hit, -vel_x.abs(), vel_x)
        vel_y = torch.where(hit, vel_y + spin, vel_y)
        ball_x = torch.where(hit, torch.full_like(ball_x, self.paddle_x - self.ball_radius - 0.005), ball_x)

        next_state = PongState(
            ball=torch.stack([ball_x, ball_y], dim=1),
            vel=torch.stack([vel_x, vel_y], dim=1),
            paddle_y=paddle_y,
            age=state.age + 1.0,
            hits=state.hits + hit.float(),
            misses=state.misses + miss.float(),
        )
        return self.reset_where(next_state, miss, keep_score=True)


@dataclass
class CatchState:
    target: torch.Tensor
    speed: torch.Tensor
    paddle_x: torch.Tensor
    age: torch.Tensor
    hits: torch.Tensor
    misses: torch.Tensor

    def index(self, idx: torch.Tensor) -> "CatchState":
        return CatchState(
            self.target[idx].clone(),
            self.speed[idx].clone(),
            self.paddle_x[idx].clone(),
            self.age[idx].clone(),
            self.hits[idx].clone(),
            self.misses[idx].clone(),
        )

    def assign(self, idx: torch.Tensor, other: "CatchState") -> None:
        self.target[idx] = other.target.detach()
        self.speed[idx] = other.speed.detach()
        self.paddle_x[idx] = other.paddle_x.detach()
        self.age[idx] = other.age.detach()
        self.hits[idx] = other.hits.detach()
        self.misses[idx] = other.misses.detach()


class CatchGame:
    name = "catch"
    obs_channels = 3
    action_axis = "x"
    readout_region = (0.82, 1.0)

    def __init__(
        self,
        height: int,
        width: int,
        paddle_half_width: float = 0.18,
        paddle_speed: float = 0.060,
        target_radius: float = 0.055,
        speed_min: float = 0.025,
        speed_max: float = 0.050,
    ):
        self.height = height
        self.width = width
        self.paddle_half_width = paddle_half_width
        self.paddle_speed = paddle_speed
        self.target_radius = target_radius
        self.speed_min = speed_min
        self.speed_max = speed_max

    def new_state(self, batch_size: int, device: torch.device) -> CatchState:
        target = torch.empty(batch_size, 2, device=device)
        target[:, 0] = torch.rand(batch_size, device=device) * 1.7 - 0.85
        target[:, 1] = torch.rand(batch_size, device=device) * 0.4 - 1.0
        speed = self.speed_min + torch.rand(batch_size, device=device) * (self.speed_max - self.speed_min)
        paddle_x = torch.rand(batch_size, device=device) * 1.2 - 0.6
        zeros = torch.zeros(batch_size, device=device)
        return CatchState(target=target, speed=speed, paddle_x=paddle_x, age=zeros, hits=zeros, misses=zeros)

    def reset_where(self, state: CatchState, mask: torch.Tensor, keep_score: bool = False) -> CatchState:
        fresh = self.new_state(state.target.shape[0], state.target.device)
        m1 = mask[:, None]
        return CatchState(
            target=torch.where(m1, fresh.target, state.target),
            speed=torch.where(mask, fresh.speed, state.speed),
            paddle_x=torch.where(mask, fresh.paddle_x, state.paddle_x),
            age=torch.where(mask, fresh.age, state.age),
            hits=state.hits if keep_score else torch.where(mask, fresh.hits, state.hits),
            misses=state.misses if keep_score else torch.where(mask, fresh.misses, state.misses),
        )

    def render(self, state: CatchState) -> torch.Tensor:
        xx, yy = make_xy_grid(self.height, self.width, state.target.device)
        tx = state.target[:, 0, None, None]
        ty = state.target[:, 1, None, None]
        target = torch.exp(-((xx - tx).square() + (yy - ty).square()) / (2.0 * self.target_radius**2))
        paddle_x = torch.sigmoid((self.paddle_half_width - (xx - state.paddle_x[:, None, None]).abs()) / 0.018)
        paddle_y = torch.exp(-((yy - 0.88).square()) / (2.0 * 0.018**2))
        paddle = paddle_x * paddle_y
        floor = torch.exp(-((yy - 0.98).abs()) / 0.025).expand_as(target)
        return torch.stack([target.clamp(0.0, 1.0), paddle.clamp(0.0, 1.0), floor.clamp(0.0, 1.0)], dim=1)

    def target_coord(self, state: CatchState) -> torch.Tensor:
        return state.target[:, 0]

    def urgency(self, state: CatchState) -> torch.Tensor:
        return (0.4 + 1.6 * torch.sigmoid((state.target[:, 1] - 0.1) * 5.0)).detach()

    def game_loss(self, state: CatchState, action_x: torch.Tensor) -> torch.Tensor:
        track = F.smooth_l1_loss(action_x, self.target_coord(state), reduction="none")
        return track * self.urgency(state)

    def step(self, state: CatchState, action_x: torch.Tensor) -> CatchState:
        move = torch.tanh((action_x - state.paddle_x) / 0.18) * self.paddle_speed
        paddle_x = (state.paddle_x + move).clamp(-1.0 + self.paddle_half_width, 1.0 - self.paddle_half_width)
        target = state.target.clone()
        target[:, 1] = target[:, 1] + state.speed
        landed = target[:, 1] >= 0.88
        hit = landed & ((target[:, 0] - paddle_x).abs() <= self.paddle_half_width + self.target_radius)
        miss = landed & (~hit)
        next_state = CatchState(
            target=target,
            speed=state.speed,
            paddle_x=paddle_x,
            age=state.age + 1.0,
            hits=state.hits + hit.float(),
            misses=state.misses + miss.float(),
        )
        return self.reset_where(next_state, landed, keep_score=True)
