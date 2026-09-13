"""Audio-conditioned diffusion over three persistent 8-D muyu geometries."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .conditioner import AudioConditioner
from .nn_blocks import _ResidualBlock, _TimeEmbedding


INPUT_NAMES = ("features", "strike_mask")


def _diffusion_schedule(steps: int, name: str) -> torch.Tensor:
    """Return a numerically safe VP/DDPM beta schedule in float64."""

    if int(steps) < 2:
        raise ValueError("diffusion_steps must be at least two")
    if name == "linear":
        return torch.linspace(1.0e-4, 2.0e-2, int(steps), dtype=torch.float64)
    if name == "cosine":
        offset = 0.008
        grid = torch.linspace(0, int(steps), int(steps) + 1, dtype=torch.float64)
        alpha_bars = torch.cos(
            ((grid / float(steps) + offset) / (1.0 + offset)) * math.pi * 0.5
        ).square()
        alpha_bars = alpha_bars / alpha_bars[0]
        return (1.0 - alpha_bars[1:] / alpha_bars[:-1]).clamp(1.0e-8, 0.999)
    raise ValueError("beta_schedule must be 'cosine' or 'linear'")


class GeometryDiffusion(nn.Module):
    """Velocity-predicting VP/DDPM posterior over three ordered 8-D slots.

    ``v`` prediction keeps the clean-sample reconstruction well-conditioned at
    the near-zero terminal SNR of the cosine schedule.  This matters for the
    low-dimensional geometry posterior: an epsilon error at the final cosine
    step would otherwise be amplified by roughly ``1/sqrt(alpha_bar)``.
    """

    prediction_type = "v"

    def __init__(
        self,
        *,
        windows: int = 3,
        mel_bins: int = 64,
        geometry_slots: int = 3,
        slot_latent_dim: int = 8,
        latent_dim: int = 24,
        model_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        note_layers: Optional[int] = None,
        slot_layers: Optional[int] = None,
        dropout: float = 0.0,
        diffusion_hidden_dim: int = 256,
        time_dim: int = 64,
        residual_blocks: int = 4,
        diffusion_steps: int = 1000,
        beta_schedule: str = "cosine",
    ) -> None:
        super().__init__()
        if int(geometry_slots) != 3 or int(slot_latent_dim) != 8:
            raise ValueError("audible joint diffusion v1 requires three 8-D slots")
        if int(latent_dim) != int(geometry_slots) * int(slot_latent_dim):
            raise ValueError("latent_dim must equal geometry_slots * slot_latent_dim")
        if int(diffusion_hidden_dim) <= 0 or int(residual_blocks) <= 0:
            raise ValueError("diffusion dimensions must be positive")

        self.geometry_slots = int(geometry_slots)
        self.slot_latent_dim = int(slot_latent_dim)
        self.latent_dim = int(latent_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.beta_schedule = str(beta_schedule)
        self.conditioner = AudioConditioner(
            windows=windows,
            mel_bins=mel_bins,
            geometry_slots=geometry_slots,
            latent_dim=slot_latent_dim,
            model_dim=model_dim,
            heads=heads,
            layers=layers,
            note_layers=note_layers,
            slot_layers=slot_layers,
            dropout=dropout,
        )
        self.condition_dim = self.geometry_slots * self.conditioner.model_dim
        self.time_embedding = _TimeEmbedding(int(time_dim))
        self.input_projection = nn.Sequential(
            nn.Linear(
                self.latent_dim + self.condition_dim + int(time_dim),
                int(diffusion_hidden_dim),
            ),
            nn.LayerNorm(int(diffusion_hidden_dim)),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [_ResidualBlock(int(diffusion_hidden_dim)) for _ in range(int(residual_blocks))]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(int(diffusion_hidden_dim)),
            nn.Linear(int(diffusion_hidden_dim), self.latent_dim),
        )

        betas = _diffusion_schedule(self.diffusion_steps, self.beta_schedule)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())

    def encode_condition(
        self,
        features: torch.Tensor,
        strike_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode only audible features and their validity mask."""

        slots = self.conditioner.encode(features, strike_mask)
        return slots.reshape(slots.shape[0], self.condition_dim)

    def _validate_timestep(
        self,
        timestep: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        if timestep.shape != (batch,):
            raise ValueError("timestep must have shape [B]")
        if torch.is_floating_point(timestep) or timestep.dtype == torch.bool:
            raise TypeError("timestep must contain integer DDPM indices")
        timestep = timestep.to(dtype=torch.long)
        if torch.any(timestep < 0) or torch.any(timestep >= self.diffusion_steps):
            raise ValueError("timestep is outside the DDPM schedule")
        return timestep

    def noise_prediction(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latent.ndim != 2 or noisy_latent.shape[-1] != self.latent_dim:
            raise ValueError("noisy_latent must have shape [B, latent_dim]")
        timestep = self._validate_timestep(timestep, noisy_latent.shape[0])
        if condition.shape != (noisy_latent.shape[0], self.condition_dim):
            raise ValueError("condition must have shape [B, condition_dim]")
        normalized_time = timestep.to(dtype=noisy_latent.dtype) / float(
            self.diffusion_steps - 1
        )
        hidden = self.input_projection(
            torch.cat(
                (
                    noisy_latent,
                    condition,
                    self.time_embedding(normalized_time),
                ),
                dim=-1,
            )
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        features: torch.Tensor,
        strike_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.noise_prediction(
            noisy_latent,
            timestep,
            self.encode_condition(features, strike_mask),
        )


def _validate_inputs(inputs: Dict[str, torch.Tensor]) -> None:
    if set(inputs) != set(INPUT_NAMES) or len(inputs) != len(INPUT_NAMES):
        raise ValueError("diffusion inputs must be exactly features + strike_mask")


def _extract_schedule(
    schedule: torch.Tensor,
    timestep: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    return schedule[timestep].to(device=reference.device, dtype=reference.dtype)[:, None]


def diffusion_loss(
    model: GeometryDiffusion,
    target: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    *,
    noise: Optional[torch.Tensor] = None,
    timestep: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Velocity-prediction loss for the variance-preserving process."""

    _validate_inputs(inputs)
    if target.ndim != 2 or target.shape[-1] != model.latent_dim:
        raise ValueError("target must have shape [B, latent_dim]")
    noise = torch.randn_like(target) if noise is None else noise
    timestep = (
        torch.randint(
            model.diffusion_steps,
            (target.shape[0],),
            device=target.device,
        )
        if timestep is None
        else timestep
    )
    if noise.shape != target.shape:
        raise ValueError("noise must match target shape")
    timestep = model._validate_timestep(timestep, target.shape[0])
    alpha_bar = _extract_schedule(model.alpha_bars, timestep, target)
    noisy_latent = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise
    prediction = model(
        noisy_latent,
        timestep,
        features=inputs["features"],
        strike_mask=inputs["strike_mask"],
    )
    velocity_target = alpha_bar.sqrt() * noise - (1.0 - alpha_bar).sqrt() * target
    loss = F.mse_loss(prediction, velocity_target)
    return loss, {
        "prediction": prediction,
        "target_noise": noise,
        "target_velocity": velocity_target,
        "timestep": timestep,
        "noisy_latent": noisy_latent,
    }


@torch.no_grad()
def sample_ddim(
    model: GeometryDiffusion,
    inputs: Dict[str, torch.Tensor],
    *,
    samples: int = 16,
    steps: int = 48,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Deterministic-trajectory DDIM sampling with output ``[B,K,24]``.

    DDIM uses ``eta=0``.  Supplying the same explicit ``seed`` (or an
    equivalently seeded generator) therefore reproduces the complete sample
    tensor exactly for a model in evaluation mode.
    """

    _validate_inputs(inputs)
    if int(samples) <= 0 or int(steps) <= 0:
        raise ValueError("samples and steps must be positive")
    if int(steps) > model.diffusion_steps:
        raise ValueError("DDIM steps cannot exceed diffusion_steps")
    if seed is not None and generator is not None:
        raise ValueError("pass seed or generator, not both")

    condition = model.encode_condition(
        inputs["features"],
        inputs["strike_mask"],
    )
    batch = condition.shape[0]
    condition = condition[:, None, :].expand(batch, int(samples), -1).reshape(
        batch * int(samples), -1
    )
    if seed is not None:
        generator = torch.Generator(device=condition.device).manual_seed(int(seed))
    latent = torch.randn(
        batch * int(samples),
        model.latent_dim,
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )

    schedule = torch.linspace(
        model.diffusion_steps - 1,
        0,
        int(steps),
        device=condition.device,
        dtype=torch.float64,
    ).round().to(torch.long)
    for index, current in enumerate(schedule):
        timestep = torch.full(
            (latent.shape[0],),
            int(current.item()),
            device=latent.device,
            dtype=torch.long,
        )
        predicted_velocity = model.noise_prediction(latent, timestep, condition)
        alpha_bar = model.alpha_bars[timestep[0]].to(
            device=latent.device, dtype=latent.dtype
        )
        # Stable v-parameterization identities (no division by alpha_bar).
        clean = (
            torch.sqrt(alpha_bar) * latent
            - torch.sqrt(1.0 - alpha_bar) * predicted_velocity
        )
        predicted_noise = (
            torch.sqrt(1.0 - alpha_bar) * latent
            + torch.sqrt(alpha_bar) * predicted_velocity
        )
        if index + 1 == schedule.numel():
            latent = clean
        else:
            previous = schedule[index + 1]
            previous_alpha_bar = model.alpha_bars[previous].to(
                device=latent.device, dtype=latent.dtype
            )
            latent = (
                torch.sqrt(previous_alpha_bar) * clean
                + torch.sqrt(1.0 - previous_alpha_bar) * predicted_noise
            )
    return latent.reshape(batch, int(samples), model.latent_dim)


__all__ = [
    "GeometryDiffusion",
    "diffusion_loss",
    "sample_ddim",
]
