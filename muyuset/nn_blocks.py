"""Checkpoint-compatible shared neural blocks (no Flow model dependency)."""

import math
import torch
from torch import nn


class _TimeEmbedding(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be even and >= 4")
        self.dimension = int(dimension)
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension * 2), nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, time):
        if time.ndim != 1:
            raise ValueError("time must have shape [B]")
        half = self.dimension // 2
        frequencies = torch.exp(torch.arange(half, device=time.device, dtype=time.dtype)
                                * (-math.log(10000.0) / max(1, half - 1)))
        angles = time[:, None] * frequencies[None, :] * (2.0 * math.pi)
        return self.projection(torch.cat((torch.sin(angles), torch.cos(angles)), -1))


class _ResidualBlock(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension * 2),
            nn.SiLU(), nn.Linear(dimension * 2, dimension),
        )

    def forward(self, values):
        return values + self.network(values)
