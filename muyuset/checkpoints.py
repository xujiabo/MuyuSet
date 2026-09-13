"""Strict inference-only checkpoint loading; never enable pickle fallback."""

import numpy as np
import torch

from .diffusion import GeometryDiffusion
from .surrogate import ForwardSurrogate
from .renderer import ACTION_CODES


def _array(checkpoint, name, shape, positive=False):
    value = np.asarray(checkpoint[name], dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid checkpoint {name}; expected finite {shape}")
    if positive and np.any(value <= 0):
        raise ValueError(f"checkpoint {name} must be positive")
    return value


def load_models(diffusion_path, surrogate_path, device="cpu"):
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True)
                   for path in (diffusion_path, surrogate_path)]
    expected = ("muyu-audible-joint-conditional-diffusion-v1",
                "muyu-audible-forward-surrogate-v1")
    models = []
    for checkpoint, format_name, model_type in zip(
            checkpoints, expected, (GeometryDiffusion, ForwardSurrogate)):
        if checkpoint.get("format") != format_name:
            raise ValueError(f"expected checkpoint format {format_name}")
        _array(checkpoint, "latent_mean", (8,))
        _array(checkpoint, "latent_std", (8,), positive=True)
        model = model_type(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval().requires_grad_(False)
        models.append(model)
    diffusion, surrogate = checkpoints
    for name in ("latent_mean", "latent_std"):
        if not np.array_equal(np.asarray(diffusion[name]), np.asarray(surrogate[name])):
            raise ValueError("diffusion and surrogate latent statistics must match")
    _array(diffusion, "feature_mean", (3, 64))
    _array(diffusion, "feature_std", (3, 64), positive=True)
    if not np.array_equal(np.asarray(surrogate["action_codes"]), ACTION_CODES):
        raise ValueError("surrogate must use the frozen 45-action order")
    return models[0], models[1], diffusion
