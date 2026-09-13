"""Frozen forward surrogate optimizes geometry only, retaining best tracks."""

import math
import numpy as np
import torch

from .surrogate import predict_action_panel


def objective(panel, observed):
    # Avoid materializing [track,hit,slot,action,mel] differences.
    energy = observed.square().mean(-1)[None, :, None, None]
    energy = energy + panel.square().mean(-1)[:, None]
    cross = torch.einsum("nd,bsad->bnsa", observed, panel) / 64.0
    rmse = (energy - 2 * cross).clamp_min(1e-8).sqrt()
    slot_cost = -(torch.logsumexp(-rmse, -1) - math.log(45))  # T_action = 1 dB
    probabilities = torch.softmax(-slot_cost, -1)
    note_cost = -(torch.logsumexp(-slot_cost, -1) - math.log(3))  # T_slot = 1 dB
    cvar = note_cost.topk(max(1, math.ceil(.1 * len(observed))), dim=1).values.mean(1)
    minimum_fraction = math.ceil(.1 * len(observed)) / len(observed)
    penalty = 100 * (minimum_fraction - probabilities.mean(1)).clamp_min(0).square().sum(1)
    return note_cost.mean(1) + .15 * cvar + penalty


def refine(model, initial, observed, action_codes, lower, upper, steps=200, learning_rate=.025):
    device = next(model.parameters()).device
    model.eval().requires_grad_(False)
    low = torch.as_tensor(lower, dtype=torch.float32, device=device)
    high = torch.as_tensor(upper, dtype=torch.float32, device=device)
    geometry = torch.nn.Parameter(torch.as_tensor(initial, dtype=torch.float32, device=device)
                                  .clamp(low, high).clone())
    actions = torch.as_tensor(np.array(action_codes, copy=True), dtype=torch.long, device=device)
    targets = torch.as_tensor(observed, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([geometry], lr=learning_rate)
    best = geometry.detach().clone()
    best_cost = torch.full((len(initial),), torch.inf, device=device)
    best_step = torch.full((len(initial),), -1, dtype=torch.long, device=device)
    history = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        costs = objective(predict_action_panel(model, geometry, actions), targets)
        if not torch.isfinite(costs).all():
            raise FloatingPointError("nonfinite surrogate objective")
        improved = costs.detach() < best_cost
        best[improved] = geometry.detach()[improved]
        best_cost[improved] = costs.detach()[improved]
        best_step[improved] = step
        history.append(costs.detach().cpu().numpy())
        if step == steps:
            break
        costs.mean().backward()
        if geometry.grad is None or not torch.isfinite(geometry.grad).all():
            raise FloatingPointError("nonfinite geometry gradient")
        optimizer.step()
        with torch.no_grad():
            geometry.clamp_(low, high)
    return best.cpu().numpy(), best_step.cpu().numpy(), np.asarray(history)
