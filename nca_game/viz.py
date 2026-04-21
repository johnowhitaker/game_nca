from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch

from .readout import AxisReadout


def obs_to_rgb(obs: torch.Tensor) -> np.ndarray:
    o = obs.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if o.shape[0] == 1:
        rgb = np.repeat(o[0:1], 3, axis=0)
    else:
        rgb = np.zeros((3, o.shape[1], o.shape[2]), dtype=np.float32)
        rgb[0] = 0.75 * o[2] + o[0]
        rgb[1] = 0.35 * o[2] + 0.85 * o[1] + 0.35 * o[0]
        rgb[2] = 0.50 * o[2] + 0.25 * o[1]
    return np.moveaxis(np.clip(rgb, 0.0, 1.0), 0, -1)


def _norm_signed(channel: torch.Tensor) -> np.ndarray:
    arr = channel.detach().cpu().float().numpy()
    scale = np.percentile(np.abs(arr), 98)
    if scale < 1e-6:
        scale = 1.0
    return np.clip(arr / scale, -1.0, 1.0)


def debug_frame(
    obs: torch.Tensor,
    state: torch.Tensor,
    readout: AxisReadout,
    target_coord: torch.Tensor,
    axis: str,
    max_channels: int = 12,
    title: str = "",
) -> np.ndarray:
    channel_count = min(max_channels, state.shape[0])
    rows = 3
    cols = max(4, int(np.ceil(channel_count / 2)))
    fig = plt.figure(figsize=(13.0, 7.2), dpi=110)
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(rows, cols, height_ratios=[1.15, 1.0, 1.0])

    ax_game = fig.add_subplot(grid[0, 0:2])
    ax_game.imshow(obs_to_rgb(obs), interpolation="nearest")
    ax_game.set_title("game pixels", fontsize=10)
    ax_game.axis("off")

    ax_action = fig.add_subplot(grid[0, 2:4])
    action_map = readout.action_map.detach().cpu().numpy()
    ax_action.imshow(action_map, cmap="magma", interpolation="nearest")
    ax_action.set_title(f"action channel: soft {axis}-argmax", fontsize=10)
    ax_action.axis("off")
    h, w = action_map.shape
    coord = float(readout.coord.detach().cpu())
    target = float(target_coord.detach().cpu())
    if axis == "y":
        y = (coord + 1.0) * 0.5 * (h - 1)
        ty = (target + 1.0) * 0.5 * (h - 1)
        ax_action.axhline(y, color="cyan", lw=1.6)
        ax_action.axhline(ty, color="white", lw=1.1, ls="--")
    else:
        x = (coord + 1.0) * 0.5 * (w - 1)
        tx = (target + 1.0) * 0.5 * (w - 1)
        ax_action.axvline(x, color="cyan", lw=1.6)
        ax_action.axvline(tx, color="white", lw=1.1, ls="--")

    ax_prob = fig.add_subplot(grid[0, 4:cols]) if cols > 4 else None
    if ax_prob is not None:
        probs = readout.probs.detach().cpu().numpy()
        ax_prob.plot(probs, color="#00bcd4", lw=1.6)
        ax_prob.set_title("action distribution", fontsize=10)
        ax_prob.set_xticks([])
        ax_prob.set_yticks([])

    for i in range(channel_count):
        row = 1 + i // cols
        col = i % cols
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(_norm_signed(state[i]), cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"state {i}", fontsize=8)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(pad=0.7)
    canvas.draw()
    width, height = canvas.get_width_height()
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    frame = rgba[:, :, :3].copy()
    plt.close(fig)
    return frame


def write_video(frames: list[np.ndarray], path: Path, fps: int = 24) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp4" and shutil.which("ffmpeg"):
        try:
            first = frames[0]
            height, width = first.shape[:2]
            if height % 2 or width % 2:
                frames = [
                    np.pad(frame, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
                    for frame in frames
                ]
                height, width = frames[0].shape[:2]
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            assert proc.stdin is not None
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8).tobytes())
            proc.stdin.close()
            if proc.wait() == 0:
                return path
        except Exception:
            if path.exists():
                path.unlink()

    try:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=8)
        return path
    except Exception:
        if path.exists():
            path.unlink()
        fallback = path.with_suffix(".gif")
        imageio.mimsave(fallback, frames, fps=fps)
        return fallback
