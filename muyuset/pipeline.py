"""Diffusion -> exact score -> differentiable refinement -> exact selection."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import math
import multiprocessing

import numpy as np
import torch

from .checkpoints import load_models
from .diffusion import sample_ddim
from .frontend import normalize_for_diffusion, validate_signatures
from .refinement import refine
from .renderer import (ACTION_CODES, OOD_LATENT_LOWER, OOD_LATENT_UPPER,
                       evaluate, geometry_parameters, prepare_attack_shapes)


@dataclass(frozen=True)
class Config:
    subsets: int = 6
    samples: int = 16
    ddim_steps: int = 48
    tracks: int = 32
    steps: int = 200
    learning_rate: float = .025
    seed: int = 20261003
    workers: int = 1

    def validate(self):
        for name in ("subsets", "samples", "ddim_steps", "tracks", "steps", "workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.tracks > self.subsets * self.samples:
            raise ValueError("refinement tracks cannot exceed proposal count")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")


def stratified_subsets(hits, rounds, seed):
    if hits < 12 or rounds < 1:
        raise ValueError("at least 12 hits and one subset required")
    rng = np.random.RandomState(seed)
    strata = np.array_split(np.arange(hits), 12)
    subsets = []
    for _ in range(rounds):
        subset = np.asarray([part[rng.randint(len(part))] for part in strata])
        rng.shuffle(subset)
        subsets.append(subset)
    return np.asarray(subsets)


def run(signatures, diffusion_path, surrogate_path, config=Config(), device="cpu", progress=print):
    config.validate()
    signatures = validate_signatures(signatures)
    diffusion, surrogate, metadata = load_models(diffusion_path, surrogate_path, device)
    progress("1/4: sampling diffusion proposals")
    subsets = stratified_subsets(len(signatures), config.subsets, config.seed)
    normalized = normalize_for_diffusion(signatures, metadata)
    features = torch.as_tensor(normalized[subsets], device=device)
    proposals = sample_ddim(diffusion, {"features": features,
        "strike_mask": torch.ones(features.shape[:2], dtype=torch.bool, device=device)},
        samples=config.samples, steps=config.ddim_steps, seed=config.seed)
    proposals = proposals.cpu().numpy().reshape(-1, 3, 8)
    mean = np.asarray(metadata["latent_mean"], dtype=np.float32)
    std = np.asarray(metadata["latent_std"], dtype=np.float32)
    observed = prepare_attack_shapes(signatures)
    minimum_usage = math.ceil(.1 * len(signatures))
    context = (ProcessPoolExecutor(config.workers, mp_context=multiprocessing.get_context("spawn"))
               if config.workers > 1 else nullcontext(None))
    with context as pool:
        score = partial(evaluate, observed=observed)
        def score_batch(raw):
            return list(pool.map(score, raw) if pool is not None else map(score, raw))

        progress(f"2/4: verifying {len(proposals)} originals")
        original_scores = score_batch(proposals * std + mean)
        # Stable feasible-first ranking; ties keep the earlier original.
        elites = sorted(range(len(proposals)), key=lambda i: original_scores[i].rank(minimum_usage))[:config.tracks]
        progress(f"3/4: refining {config.tracks} tracks for {config.steps} Adam steps")
        refined, best_steps, history = refine(surrogate, proposals[elites], observed,
            ACTION_CODES, (OOD_LATENT_LOWER - mean) / std, (OOD_LATENT_UPPER - mean) / std,
            config.steps, config.learning_rate)
        progress(f"4/4: verifying {len(refined)} refinements")
        refined_scores = score_batch(refined * std + mean)
    scores = original_scores + refined_scores
    best_index = min(range(len(scores)), key=lambda i: scores[i].rank(minimum_usage))
    selected = scores[best_index]
    feasible = selected.feasible(minimum_usage)
    result = {
        "status": "ok" if feasible else "no_feasible_candidate",
        "hits": len(signatures), "minimum_slot_usage": minimum_usage,
        "exact_candidate_attempts": len(scores),
        "renderable_candidates": sum(s.valid for s in scores),
        "feasible_candidates": sum(s.feasible(minimum_usage) for s in scores),
        "surrogate_updates": config.tracks * config.steps,
        "paper_budget_settings": (config.subsets, config.samples, config.ddim_steps,
                                  config.tracks, config.steps) == (6, 16, 48, 32, 200),
        "candidate_scores": [{"objective": s.objective if s.valid else None,
                              "usage": s.usage.tolist(), "valid": s.valid,
                              "feasible": s.feasible(minimum_usage), "error": s.error} for s in scores],
        "best": None,
    }
    if feasible:
        result["best"] = {
            "candidate_index": best_index,
            "source": "original" if best_index < len(proposals) else "refined",
            "objective": selected.objective, "slot_usage": selected.usage.tolist(),
            "raw_latents": selected.raw_latents.tolist(),
            "geometry_si": geometry_parameters(selected.raw_latents),
            "strike_codes": np.column_stack((selected.slots,
                              ACTION_CODES[selected.action_indices])).tolist(),
        }
    arrays = {"originals_standardized": proposals, "refined_standardized": refined,
              "elite_indices": np.asarray(elites), "surrogate_best_steps": best_steps,
              "surrogate_objective_history": history, "subset_indices": subsets}
    return result, arrays
