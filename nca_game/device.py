from __future__ import annotations

import torch


def pick_device(preference: str = "auto") -> torch.device:
    """Pick a PyTorch device, preferring Apple MPS on Macs."""
    pref = preference.lower()
    if pref not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device preference: {preference}")

    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report(device: torch.device) -> str:
    if device.type == "mps":
        return "mps (Apple Metal)"
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"
