"""Frozen exact renderer and feasible-first three-object selection metric.

Physics is the archived contact-wood-damping-v2 backend; it is a calibrated
analytic proxy, not FEM or a uniquely identifiable physical reconstruction.
Observed rows must already be 80-ms, floor--70, mean-centered attack64 views.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .physics.actions import build_conditions
from .physics.audio_perceptual import render_audible_signatures
from .physics.contact import ContactIntegrationError, contact_modal_palette
from .physics.geometry import geometry_metrics, parameters_from_geometry
from .physics.latents import (
    OOD_LATENT_LOWER, OOD_LATENT_UPPER, factorized_latent_to_geometry,
)

ACTION_CODES = np.asarray(
    [(material, velocity, position) for material in range(3)
     for velocity in range(3) for position in range(5)], dtype=np.int16,
)
ACTION_CODES.setflags(write=False)
OOD_LATENT_LOWER.setflags(write=False)
OOD_LATENT_UPPER.setflags(write=False)


@dataclass(frozen=True)
class Score:
    objective: float
    usage: np.ndarray
    valid: bool
    error: Optional[str]
    slots: np.ndarray
    action_indices: np.ndarray
    raw_latents: np.ndarray

    def feasible(self, min_usage: int) -> bool:
        minimum = _minimum_usage(min_usage)
        return bool(self.valid and np.isfinite(self.objective)
                    and np.all(np.asarray(self.usage) >= minimum))

    def rank(self, min_usage: int) -> tuple:
        minimum = _minimum_usage(min_usage)
        if not self.valid or not np.isfinite(self.objective):
            return (2, float("inf"), float("inf"))
        deficit = int(np.maximum(minimum - np.asarray(self.usage), 0).sum())
        if deficit == 0:
            return (0, float(self.objective), 0)
        return (1, deficit, float(self.objective))


def _minimum_usage(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("min_usage must be a nonnegative integer")
    return int(value)


def _raw_array(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (3, 8):
        raise ValueError("raw latents must have shape [3,8]")
    if not np.isfinite(values).all():
        raise ValueError("raw latents must be finite")
    return values.copy()


def decode_latent(raw: np.ndarray):
    """Decode one eight-coordinate raw latent to SI-valued geometry."""
    return factorized_latent_to_geometry(raw)


def _object(raw: np.ndarray) -> SimpleNamespace:
    geometry = decode_latent(raw)
    metrics = geometry_metrics(geometry)
    return SimpleNamespace(geometry=geometry, metrics=metrics,
                           parameters=parameters_from_geometry(metrics))


def canonicalize(raw: np.ndarray) -> np.ndarray:
    """Sort by first audible modal frequency; raw coordinates break exact ties."""
    values = _raw_array(raw)
    keys = []
    for slot, row in enumerate(values):
        frequencies, _ = contact_modal_palette(_object(row).parameters)
        if frequencies.size == 0 or not np.isfinite(frequencies).all():
            raise FloatingPointError("geometry produced invalid modal frequencies")
        keys.append((float(frequencies[0]), tuple(row.tolist()), slot))
    order = [item[2] for item in sorted(keys)]
    return values[order].copy()


def geometry_parameters(raw: np.ndarray) -> list:
    """Three canonical slot dictionaries; dimensions and offsets are in metres."""
    return [decode_latent(row).to_dict() for row in canonicalize(raw)]


def render_signatures(raw: np.ndarray) -> np.ndarray:
    """Return absolute archived log-mel signatures [slot,45,3,64]."""
    ordered = canonicalize(raw)
    conditions = build_conditions()
    result = np.asarray([
        render_audible_signatures(_object(row), conditions, sample_rate=44100,
                                 duration_s=1.10, mel_bins=64)
        for row in ordered
    ], dtype=np.float32)
    if result.shape != (3, 45, 3, 64):
        raise RuntimeError("frozen renderer returned an unexpected signature shape")
    if not np.isfinite(result).all():
        raise FloatingPointError("rendered signatures contain nonfinite values")
    return result


def prepare_attack_shapes(signatures: np.ndarray) -> np.ndarray:
    """Archived forward-surrogate prepare_attack_shapes, without torch."""
    values = np.asarray(signatures)
    if values.ndim < 2 or tuple(values.shape[-2:]) != (3, 64):
        raise ValueError("signatures must end in [3,64]")
    attack = np.maximum(np.asarray(values[..., 0, :], dtype=np.float32),
                        np.float32(-70.0))
    attack -= np.mean(attack, axis=-1, keepdims=True, dtype=np.float32)
    if not np.isfinite(attack).all():
        raise FloatingPointError("prepared attack shapes contain nonfinite values")
    return np.ascontiguousarray(attack, dtype=np.float32)


def _invalid(error: Exception, raw: np.ndarray, notes: int) -> Score:
    if raw.shape != (3, 8):
        raw = np.full((3, 8), np.nan, dtype=np.float64)
    return Score(float("inf"), np.zeros(3, dtype=np.int32), False,
                 f"{type(error).__name__}: {error}",
                 np.full(notes, -1, dtype=np.int16),
                 np.full(notes, -1, dtype=np.int16), raw.copy())


def evaluate(raw: np.ndarray, observed: np.ndarray) -> Score:
    """Exact trim-10%-plus-0.15-P90 attack RMSE; usage remains a hard rank rule.

    Malformed observations are programming/input errors and raise. Expected
    invalid geometry and numerical rendering failures become invalid scores.
    Import, type, indexing, and arbitrary RuntimeError failures propagate.
    """
    notes = np.asarray(observed, dtype=np.float32)
    if notes.ndim != 2 or notes.shape[1] != 64 or notes.shape[0] == 0:
        raise ValueError("observed must have nonempty shape [N,64]")
    if not np.isfinite(notes).all():
        raise ValueError("observed must be finite")
    values = np.asarray(raw, dtype=np.float64)
    try:
        values = canonicalize(values)
        panel = prepare_attack_shapes(render_signatures(values))
    except (ValueError, FloatingPointError, OverflowError,
            np.linalg.LinAlgError, ContactIntegrationError) as error:
        return _invalid(error, values, len(notes))
    slot_costs = np.empty((len(notes), 3), dtype=np.float32)
    actions = np.empty((len(notes), 3), dtype=np.int16)
    for slot in range(3):
        delta = notes[:, None, :] - panel[slot][None, :, :]
        costs = np.sqrt(np.mean(np.square(delta), axis=-1))
        actions[:, slot] = np.argmin(costs, axis=1).astype(np.int16)
        slot_costs[:, slot] = np.min(costs, axis=1)
    slots = np.argmin(slot_costs, axis=1).astype(np.int16)
    rows = np.arange(len(notes))
    action_indices = actions[rows, slots]
    note_costs = slot_costs[rows, slots]
    if not np.isfinite(note_costs).all():
        return _invalid(FloatingPointError("exact costs are nonfinite"), values, len(notes))
    keep = len(notes) - int(np.floor(0.10 * len(notes)))
    objective = float(np.mean(np.sort(note_costs)[:keep])) + 0.15 * float(np.percentile(note_costs, 90))
    usage = np.bincount(slots, minlength=3).astype(np.int32)
    return Score(objective, usage, True, None, slots, action_indices, values)


__all__ = ["OOD_LATENT_LOWER", "OOD_LATENT_UPPER", "ACTION_CODES", "Score",
           "canonicalize", "decode_latent", "geometry_parameters",
           "render_signatures", "prepare_attack_shapes", "evaluate"]
