"""Perceptual signatures computed from the same mono WAV renderer users hear."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

from .contact import render_contact_excited_sound


DEFAULT_WINDOWS_S: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.08),
    (0.08, 0.30),
    (0.30, 1.10),
)


def _hz_to_mel(frequency_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def _mel_to_hz(value: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def mel_filterbank(
    sample_rate: int,
    fft_size: int,
    mel_bins: int = 64,
    minimum_hz: float = 50.0,
    maximum_hz: float = 8000.0,
) -> np.ndarray:
    """Return triangular mel filters for a real FFT power spectrum."""

    if sample_rate <= 0 or fft_size <= 0 or mel_bins <= 0:
        raise ValueError("sample rate, FFT size, and mel-bin count must be positive")
    if not 0.0 <= minimum_hz < maximum_hz <= sample_rate / 2.0:
        raise ValueError("invalid mel frequency range")
    mel_edges = np.linspace(
        _hz_to_mel(np.asarray(minimum_hz)),
        _hz_to_mel(np.asarray(maximum_hz)),
        mel_bins + 2,
    )
    frequency_edges = _mel_to_hz(mel_edges)
    indices = np.floor((fft_size + 1) * frequency_edges / sample_rate).astype(int)
    result = np.zeros((mel_bins, fft_size // 2 + 1), dtype=np.float64)
    for row in range(mel_bins):
        left, centre, right = (int(value) for value in indices[row : row + 3])
        centre = max(centre, left + 1)
        right = max(right, centre + 1)
        centre = min(centre, result.shape[1] - 1)
        right = min(right, result.shape[1])
        if centre > left:
            result[row, left:centre] = (
                np.arange(left, centre) - left
            ) / (centre - left)
        if right > centre:
            result[row, centre:right] = (
                right - np.arange(centre, right)
            ) / (right - centre)
    return result


def audible_signature(
    samples: np.ndarray,
    *,
    sample_rate: int = 44100,
    windows_s: Sequence[Tuple[float, float]] = DEFAULT_WINDOWS_S,
    mel_bins: int = 64,
    minimum_hz: float = 50.0,
    maximum_hz: float = 8000.0,
    floor_power: float = 1.0e-14,
) -> np.ndarray:
    """Return windowed absolute log-mel power in dB, shape ``[W, M]``."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be one-dimensional, finite, and non-empty")
    if floor_power <= 0.0:
        raise ValueError("floor_power must be positive")
    rows = []
    for start_s, stop_s in windows_s:
        if not 0.0 <= start_s < stop_s:
            raise ValueError("analysis windows must be positive and ascending")
        start = int(round(start_s * sample_rate))
        stop = min(values.size, int(round(stop_s * sample_rate)))
        if stop - start < 8:
            raise ValueError("analysis window lies outside the waveform")
        segment = values[start:stop]
        fft_size = 1 << (segment.size - 1).bit_length()
        window = np.hanning(segment.size)
        power = np.abs(np.fft.rfft(segment * window, n=fft_size)) ** 2
        power /= max(float(np.sum(window ** 2)), np.finfo(float).eps)
        filters = mel_filterbank(
            sample_rate, fft_size, mel_bins, minimum_hz, maximum_hz
        )
        rows.append(10.0 * np.log10(np.maximum(filters @ power, floor_power)))
    result = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("audible signature contains non-finite values")
    return result


def render_audible_signatures(
    object_item: object,
    conditions: Sequence[object],
    *,
    sample_rate: int = 44100,
    duration_s: float = 1.10,
    mel_bins: int = 64,
) -> np.ndarray:
    """Render conditions with the audible WAV backend and stack signatures."""

    signatures = []
    for condition in conditions:
        waveform = render_contact_excited_sound(
            condition.contact,
            condition.profile,
            params=object_item.parameters,
            sample_rate=sample_rate,
            duration_s=duration_s,
        )
        signatures.append(
            audible_signature(
                waveform, sample_rate=sample_rate, mel_bins=mel_bins
            )
        )
    return np.asarray(signatures, dtype=np.float32)


def audible_distance_db(first: np.ndarray, second: np.ndarray) -> float:
    """Absolute log-mel RMS difference in dB."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("audible signatures must have equal non-empty shapes")
    return float(np.sqrt(np.mean((left - right) ** 2)))


__all__ = [
    "DEFAULT_WINDOWS_S",
    "audible_distance_db",
    "audible_signature",
    "mel_filterbank",
    "render_audible_signatures",
]
