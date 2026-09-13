"""Frozen, differentiable geometry/action-to-attack surrogate."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn


FORMAT = "muyu-audible-forward-surrogate-v1"
ATTACK_FLOOR_DB = -70.0


def prepare_attack_shapes(
    signatures: np.ndarray,
    *,
    floor_db: float = ATTACK_FLOOR_DB,
) -> np.ndarray:
    """Convert ``[...,3,64]`` absolute signatures to centered attack64.

    This is intentionally identical to ``render_action_panel``: select the
    first 0--80 ms window, floor at -70 dB, then remove the per-strike mean.
    """

    values = np.asarray(signatures)
    if values.ndim < 2 or tuple(values.shape[-2:]) != (3, 64):
        raise ValueError("signatures must end in [3,64]")
    if not np.isfinite(float(floor_db)):
        raise ValueError("floor_db must be finite")
    attack = np.maximum(
        np.asarray(values[..., 0, :], dtype=np.float32),
        np.float32(floor_db),
    )
    attack -= np.mean(attack, axis=-1, keepdims=True, dtype=np.float32)
    if not bool(np.isfinite(attack).all()):
        raise FloatingPointError("prepared attack shapes contain non-finite values")
    return np.ascontiguousarray(attack, dtype=np.float32)


def validate_action_codes(action_codes: np.ndarray) -> np.ndarray:
    """Return canonical int64 ``[A,3]`` material/velocity/position codes."""

    values = np.asarray(action_codes)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] <= 0:
        raise ValueError("action_codes must have shape [A,3]")
    if not np.issubdtype(values.dtype, np.integer):
        if not bool(np.equal(values, np.round(values)).all()):
            raise ValueError("action_codes must contain integers")
    result = np.asarray(values, dtype=np.int64)
    lower_ok = bool(np.all(result >= 0))
    upper_ok = bool(
        np.all(result[:, 0] < 3)
        and np.all(result[:, 1] < 3)
        and np.all(result[:, 2] < 5)
    )
    if not lower_ok or not upper_ok:
        raise ValueError("action codes exceed the 3 material x 3 velocity x 5 position vocabulary")
    if np.unique(result, axis=0).shape[0] != result.shape[0]:
        raise ValueError("action_codes must not contain duplicate actions")
    return np.ascontiguousarray(result)


class _ResidualBlock(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class ForwardSurrogate(nn.Module):
    """Predict centered 64-bin attack shape from geometry and strike action.

    ``geometry`` is ``[B,8]`` in train-standardized latent coordinates.
    ``action_codes`` is ``[B,3]`` ordered as material, velocity, position.
    The returned tensor is ``[B,64]`` and is exactly mean-zero per row.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        mel_bins: int = 64,
        model_dim: int = 384,
        residual_hidden_dim: int = 768,
        residual_blocks: int = 4,
        action_embedding_dim: int = 24,
        fourier_frequencies: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    ) -> None:
        super().__init__()
        integer_values = (
            latent_dim,
            mel_bins,
            model_dim,
            residual_hidden_dim,
            residual_blocks,
            action_embedding_dim,
        )
        if any(int(value) <= 0 for value in integer_values):
            raise ValueError("all model dimensions must be positive")
        frequencies = torch.as_tensor(tuple(fourier_frequencies), dtype=torch.float32)
        if frequencies.ndim != 1 or frequencies.numel() <= 0:
            raise ValueError("fourier_frequencies must be a non-empty vector")
        if not bool(torch.isfinite(frequencies).all()) or bool(torch.any(frequencies <= 0.0)):
            raise ValueError("fourier_frequencies must be finite and positive")

        self.latent_dim = int(latent_dim)
        self.mel_bins = int(mel_bins)
        self.model_dim = int(model_dim)
        self.action_embedding_dim = int(action_embedding_dim)
        self.register_buffer("fourier_frequencies", frequencies)
        geometry_input_dim = self.latent_dim * (1 + 2 * int(frequencies.numel()))
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_input_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.material_embedding = nn.Embedding(3, self.action_embedding_dim)
        self.velocity_embedding = nn.Embedding(3, self.action_embedding_dim)
        self.position_embedding = nn.Embedding(5, self.action_embedding_dim)
        self.action_projection = nn.Linear(
            3 * self.action_embedding_dim,
            self.model_dim,
        )
        self.residual = nn.ModuleList(
            _ResidualBlock(self.model_dim, int(residual_hidden_dim))
            for _ in range(int(residual_blocks))
        )
        self.output = nn.Linear(self.model_dim, self.mel_bins)

    def _validate_inputs(
        self,
        geometry: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if geometry.ndim != 2 or geometry.shape[1] != self.latent_dim:
            raise ValueError("geometry must have shape [B,latent_dim]")
        if not torch.is_floating_point(geometry):
            raise TypeError("geometry must be floating point")
        if action_codes.ndim != 2 or tuple(action_codes.shape) != (
            geometry.shape[0],
            3,
        ):
            raise ValueError("action_codes must have shape [B,3]")
        actions = action_codes.to(device=geometry.device, dtype=torch.long)
        if torch.any(actions < 0):
            raise ValueError("action codes must be non-negative")
        if (
            torch.any(actions[:, 0] >= 3)
            or torch.any(actions[:, 1] >= 3)
            or torch.any(actions[:, 2] >= 5)
        ):
            raise ValueError("action code is outside the frozen vocabulary")
        return geometry, actions

    def forward(
        self,
        geometry: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> torch.Tensor:
        geometry, actions = self._validate_inputs(geometry, action_codes)
        frequencies = self.fourier_frequencies.to(
            device=geometry.device,
            dtype=geometry.dtype,
        )
        phase = geometry[:, :, None] * frequencies[None, None, :]
        encoded_geometry = torch.cat(
            (
                geometry,
                torch.sin(phase).flatten(start_dim=1),
                torch.cos(phase).flatten(start_dim=1),
            ),
            dim=1,
        )
        geometry_state = self.geometry_encoder(encoded_geometry)
        action_state = self.action_projection(
            torch.cat(
                (
                    self.material_embedding(actions[:, 0]),
                    self.velocity_embedding(actions[:, 1]),
                    self.position_embedding(actions[:, 2]),
                ),
                dim=1,
            )
        )
        values = geometry_state + action_state
        for block in self.residual:
            values = block(values)
        prediction = self.output(values)
        return prediction - torch.mean(prediction, dim=1, keepdim=True)


def predict_action_panel(
    model: ForwardSurrogate,
    geometry_slots: torch.Tensor,
    action_codes: torch.Tensor,
) -> torch.Tensor:
    """Evaluate all actions for ``[B,3,8]`` geometries.

    Returns ``[B,3,A,64]`` while retaining gradients with respect to geometry.
    """

    if geometry_slots.ndim != 3 or tuple(geometry_slots.shape[1:]) != (
        3,
        model.latent_dim,
    ):
        raise ValueError("geometry_slots must have shape [B,3,latent_dim]")
    if action_codes.ndim != 2 or action_codes.shape[1] != 3:
        raise ValueError("action_codes must have shape [A,3]")
    batch = int(geometry_slots.shape[0])
    actions = int(action_codes.shape[0])
    expanded_geometry = geometry_slots[:, :, None, :].expand(
        batch,
        3,
        actions,
        model.latent_dim,
    )
    expanded_actions = action_codes.to(device=geometry_slots.device)[None, None, :, :].expand(
        batch,
        3,
        actions,
        3,
    )
    prediction = model(
        expanded_geometry.reshape(-1, model.latent_dim),
        expanded_actions.reshape(-1, 3),
    )
    return prediction.reshape(batch, 3, actions, model.mel_bins)


def surrogate_training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scale_db: float = 20.0,
    difference_weight: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Renderer-aligned spectral and adjacent-mel finite-difference loss."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have equal [B,F] shapes")
    if prediction.shape[1] < 2:
        raise ValueError("at least two frequency bins are required")
    if float(scale_db) <= 0.0 or float(difference_weight) < 0.0:
        raise ValueError("scale_db must be positive and difference_weight non-negative")
    spectral = torch.mean(torch.square((prediction - target) / float(scale_db)))
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    difference = torch.mean(
        torch.square((prediction_delta - target_delta) / float(scale_db))
    )
    return {
        "loss": spectral + float(difference_weight) * difference,
        "spectral_mse_scaled": spectral,
        "difference_mse_scaled": difference,
    }
