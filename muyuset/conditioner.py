"""Joint three-object inverse model for unlabelled audible Muyu strikes.

The regressor consumes only audible features and a strike-validity mask.  It
does not receive the target fish identity or any playing-condition labels.
Three learned, ordered slot queries summarize the mixed strike set and predict
standardized geometry latents in low-, middle-, and high-pitch target order.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class _AudibleNoteEncoder(nn.Module):
    """Encode each audible signature independently with shared weights."""

    def __init__(self, windows: int, mel_bins: int, model_dim: int) -> None:
        super().__init__()
        self.windows = int(windows)
        self.mel_bins = int(mel_bins)
        self.convolution = nn.Sequential(
            nn.Conv1d(self.windows, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(128, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.convolution(features)
        average = F.adaptive_avg_pool1d(encoded, 1).squeeze(-1)
        maximum = F.adaptive_max_pool1d(encoded, 1).squeeze(-1)
        return self.projection(torch.cat((average, maximum), dim=-1))


class _SlotAttentionBlock(nn.Module):
    """Let ordered slots exchange information and attend to valid notes."""

    def __init__(
        self,
        model_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.slot_norm = nn.LayerNorm(model_dim)
        self.slot_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.memory_norm = nn.LayerNorm(model_dim)
        self.cross_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        slots: torch.Tensor,
        notes: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_slots = self.slot_norm(slots)
        slot_update, _ = self.slot_attention(
            normalized_slots,
            normalized_slots,
            normalized_slots,
            need_weights=False,
        )
        slots = slots + self.dropout(slot_update)

        cross_update, attention = self.cross_attention(
            self.cross_norm(slots),
            self.memory_norm(notes),
            self.memory_norm(notes),
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        slots = slots + self.dropout(cross_update)
        slots = slots + self.dropout(self.feedforward(self.feedforward_norm(slots)))
        return slots, attention


class AudioConditioner(nn.Module):
    """Predict three standardized geometry latents from unlabelled strikes.

    ``features`` has shape ``[B, S, windows, mel_bins]`` and ``strike_mask``
    has shape ``[B, S]``.  The common training case uses ``S=12``.  A variable
    padded strike dimension is accepted so long as every batch item has at
    least one valid strike.  Invalid rows are zeroed before the note encoder;
    consequently even non-finite padding cannot affect a prediction.

    Output slot indices are semantic and fixed by supervision:
    ``0=low``, ``1=middle``, and ``2=high`` first-modal-frequency target.
    Predictions are unconstrained because their targets are standardized
    eight-dimensional latents.
    """

    slot_names = ("low", "middle", "high")

    def __init__(
        self,
        *,
        windows: int = 3,
        mel_bins: int = 64,
        geometry_slots: int = 3,
        latent_dim: int = 8,
        model_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        note_layers: Optional[int] = None,
        slot_layers: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        note_layer_count = int(layers if note_layers is None else note_layers)
        slot_layer_count = int(layers if slot_layers is None else slot_layers)
        integer_dimensions = (
            windows,
            mel_bins,
            geometry_slots,
            latent_dim,
            model_dim,
            heads,
            layers,
            note_layer_count,
            slot_layer_count,
        )
        if any(int(value) <= 0 for value in integer_dimensions):
            raise ValueError("all model dimensions and layer counts must be positive")
        if int(geometry_slots) != len(self.slot_names):
            raise ValueError("audible joint v1 requires exactly three geometry slots")
        if int(model_dim) % int(heads) != 0:
            raise ValueError("model_dim must be divisible by heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.windows = int(windows)
        self.mel_bins = int(mel_bins)
        self.geometry_slots = int(geometry_slots)
        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)

        self.note_encoder = _AudibleNoteEncoder(
            self.windows,
            self.mel_bins,
            self.model_dim,
        )
        note_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(heads),
            dim_feedforward=self.model_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.note_transformer = nn.TransformerEncoder(
            note_layer,
            num_layers=note_layer_count,
            norm=nn.LayerNorm(self.model_dim),
            enable_nested_tensor=False,
        )

        self.slot_queries = nn.Parameter(
            torch.empty(self.geometry_slots, self.model_dim)
        )
        nn.init.normal_(self.slot_queries, std=0.02)
        # An ordinal query component breaks slot symmetry explicitly while the
        # learned queries retain enough capacity for each target pitch band.
        self.register_buffer(
            "slot_ranks",
            torch.linspace(-1.0, 1.0, self.geometry_slots).reshape(1, -1, 1),
            persistent=False,
        )
        self.slot_rank_projection = nn.Linear(1, self.model_dim, bias=False)
        self.slot_blocks = nn.ModuleList(
            _SlotAttentionBlock(self.model_dim, int(heads), float(dropout))
            for _ in range(slot_layer_count)
        )
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.geometry_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.latent_dim),
        )

    def _validate_inputs(
        self,
        features: torch.Tensor,
        strike_mask: torch.Tensor,
    ) -> Tuple[int, int, torch.Tensor]:
        if features.ndim != 4 or tuple(features.shape[-2:]) != (
            self.windows,
            self.mel_bins,
        ):
            raise ValueError("features must have shape [B,S,windows,mel_bins]")
        if not torch.is_floating_point(features):
            raise TypeError("features must be floating point")
        batch, strikes = features.shape[:2]
        if batch <= 0 or strikes <= 0:
            raise ValueError("features must contain a non-empty batch and strike axis")
        if tuple(strike_mask.shape) != (batch, strikes):
            raise ValueError("strike_mask must have shape [B,S]")
        mask = strike_mask.to(device=features.device, dtype=torch.bool)
        if torch.any(~mask.any(dim=1)):
            raise ValueError("every batch item must contain at least one valid strike")
        return int(batch), int(strikes), mask

    def _encode_slots(
        self,
        features: torch.Tensor,
        strike_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, strikes, mask = self._validate_inputs(features, strike_mask)
        safe_features = torch.where(
            mask[:, :, None, None],
            features,
            torch.zeros_like(features),
        )
        notes = self.note_encoder(
            safe_features.reshape(
                batch * strikes,
                self.windows,
                self.mel_bins,
            )
        ).reshape(batch, strikes, self.model_dim)
        padding_mask = ~mask
        notes = self.note_transformer(
            notes,
            src_key_padding_mask=padding_mask,
        )

        ranks = self.slot_ranks.to(device=features.device, dtype=features.dtype)
        ordered_queries = self.slot_queries.unsqueeze(0) + self.slot_rank_projection(
            ranks
        )
        slots = ordered_queries.expand(batch, -1, -1)
        attention = features.new_zeros(batch, self.geometry_slots, strikes)
        for block in self.slot_blocks:
            slots, attention = block(slots, notes, padding_mask)
        return self.output_norm(slots), attention

    def encode(
        self,
        features: torch.Tensor,
        strike_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ordered slot embeddings for a later joint conditional flow."""

        slots, _ = self._encode_slots(features, strike_mask)
        return slots

    def forward(
        self,
        features: torch.Tensor,
        strike_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        slots, attention = self._encode_slots(features, strike_mask)
        return {
            "geometry_latents": self.geometry_head(slots),
            "slot_embeddings": slots,
            "slot_attention": attention,
        }


__all__ = ["AudioConditioner"]
