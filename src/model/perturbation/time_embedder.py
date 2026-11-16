"""Sinusoidal flow-time embedding compatible with iPStem."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TimeEmbedder(nn.Module):
    def __init__(self, time_freqs: int = 1024, time_max_period: int = 10000):
        super().__init__()
        if time_freqs <= 0 or time_freqs % 2:
            raise ValueError("time_freqs must be a positive even integer")
        self.dim = time_freqs
        self.time_max_period = time_max_period

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        t = t.float().reshape(-1) * self.time_max_period
        freqs = torch.exp(
            -math.log(self.time_max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.cos(), args.sin()], dim=-1)
