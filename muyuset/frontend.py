"""Absolute three-window log-mel inputs; optional mono WAV + locked onsets."""

import json
import numpy as np
from scipy.io import wavfile

from .physics.audio_perceptual import audible_signature


def validate_signatures(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (3, 64) or len(values) < 12:
        raise ValueError("input must have shape [N,3,64], with at least 12 hits")
    if not np.isfinite(values).all():
        raise ValueError("input signatures must be finite")
    return np.ascontiguousarray(values)


def load_signatures(path):
    return validate_signatures(np.load(path, allow_pickle=False))


def normalize_for_diffusion(signatures, checkpoint):
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.maximum(np.asarray(checkpoint["feature_std"], dtype=np.float32), 1e-3)
    # Align each hit's gain to training before standardizing all three windows.
    shift = signatures[:, 0].mean(axis=1) - mean[0].mean()
    return np.ascontiguousarray((signatures - shift[:, None, None] - mean) / std)


def extract_signatures(samples, onsets_s, sample_rate=44100):
    if sample_rate != 44100:
        raise ValueError("use a 44100 Hz waveform")
    samples = np.asarray(samples, dtype=np.float64)
    onsets = np.asarray(onsets_s, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all() or not len(samples):
        raise ValueError("waveform must be finite, mono, and nonempty")
    if onsets.ndim != 1 or not np.isfinite(onsets).all():
        raise ValueError("onsets must be a finite vector")
    if np.any(onsets < 0) or np.any(onsets >= len(samples) / sample_rate):
        raise ValueError("onsets must lie inside the waveform")
    length = round(1.10 * sample_rate)
    rows = []
    for onset in onsets:
        start = round(float(onset) * sample_rate)
        segment = np.zeros(length, dtype=np.float64)
        usable = min(length, len(samples) - start)
        segment[:usable] = samples[start:start + usable]
        rows.append(np.maximum(audible_signature(segment, sample_rate=sample_rate), -70.0))
    return validate_signatures(rows)


def load_audio(wav_path, plan_path, start=0.0, end=None):
    rate, samples = wavfile.read(wav_path)
    if rate != 44100 or samples.ndim != 1:
        raise ValueError("provide a 44100 Hz mono percussive WAV")
    if np.issubdtype(samples.dtype, np.signedinteger):
        samples = samples.astype(np.float64) / float(-np.iinfo(samples.dtype).min)
    elif samples.dtype == np.uint8:
        samples = (samples.astype(np.float64) - 128.0) / 128.0
    elif not np.issubdtype(samples.dtype, np.floating):
        raise ValueError("unsupported WAV sample format")
    with open(plan_path, encoding="utf-8") as source:
        plan = json.load(source)
    notes = plan.get("notes", plan.get("performance_plan"))
    if not isinstance(notes, list) or not notes:
        raise ValueError("plan requires a nonempty notes or performance_plan list")
    onsets = []
    for note in notes:
        if not isinstance(note, dict) or not ({"onset_s", "time_s"} & note.keys()):
            raise ValueError("each note needs onset_s or time_s")
        onsets.append(float(note.get("onset_s", note.get("time_s"))))
    onsets = np.asarray(onsets, dtype=np.float64)
    if not np.isfinite(onsets).all() or np.any(np.diff(onsets) <= 0):
        raise ValueError("onsets must be finite and strictly increasing")
    if np.any(onsets < 0) or np.any(onsets >= len(samples) / rate):
        raise ValueError("all planned onsets must lie inside the complete waveform")
    stop = len(samples) / rate if end is None else float(end)
    if not np.isfinite(start) or not np.isfinite(stop) or not 0 <= start < stop <= len(samples) / rate:
        raise ValueError("invalid start/end interval")
    onsets = onsets[(onsets >= start) & (onsets < stop)]
    return extract_signatures(samples, onsets, rate), onsets
