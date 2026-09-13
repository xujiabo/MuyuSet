# MuyuSet: Diffusion + Differentiable Refinement

Minimal inference code for inverse design of a **muyu (木鱼 / wooden fish)**, a hollow wooden percussion instrument. Given a sequence of strikes, the pipeline searches for a shared set of **three instrument geometries** and assigns an instrument and strike action to every hit.

```text
Strike features → conditional Diffusion → frozen-renderer ranking
                → differentiable surrogate refinement → frozen-renderer selection
```

The differentiable component is a **learned forward surrogate**. Gradients update geometry latents, not model weights. Final scores and discrete strike assignments always come from the original, non-differentiable renderer. This is a geometry search pipeline, not a guarantee of unique source-geometry recovery.

## Run

Use Python 3.10 or newer. From this directory:

```bash
python -m pip install -r requirements.txt
python run.py
```

Both inference checkpoints and a synthetic 48-hit input are included; no training archive is needed. The script automatically uses CUDA when available, otherwise CPU. `--workers 4` parallelizes the CPU renderer and leaves neural inference on the selected device:

```bash
python run.py --workers 4 --output runs/full
```

For a small CPU smoke test, with a deliberately reduced search budget:

```bash
python run.py --device cpu --subsets 1 --samples 4 --tracks 2 --steps 2 --output runs/smoke
```

The output directory must be empty or nonexistent. Choose a new `--output` when rerunning. See `python run.py --help` for all options.

## Default search

1. Select six temporally stratified 12-hit subsets. Sample 16 joint geometry proposals per subset using 48-step DDIM: **96 originals**.
2. Evaluate all originals with the frozen renderer and select 32 starting points, using feasible-first ranking.
3. Optimize each starting point for 200 Adam steps through the frozen surrogate. Clamp geometry to the predefined out-of-distribution (OOD) search box and retain the lowest surrogate-loss iterate, including the initial iterate. This expanded box also includes the training region, so a selected geometry is not necessarily OOD.
4. Evaluate the **32 retained refinements** with the frozen renderer. Select from **all 96 originals and 32 refinements**, so a refinement cannot discard a better original.

This gives **128 candidate evaluation attempts**, including invalid candidates, and **6,400 per-track Adam updates**. It does not mean 128 waveform renders: each candidate contains three instruments and is compared under 45 actions per instrument. Each instrument must explain at least `ceil(0.1 * N)` hits.

The frozen-renderer objective is the mean hitwise attack-spectrum RMSE after trimming the largest `floor(0.1 * N)` errors, plus `0.15 × P90`. Refinement uses soft action/slot minima, a tail term, and a soft usage penalty; the final hard usage check remains authoritative.

The default count/iteration settings match the intended 128-attempt protocol. Subset/seed handling was simplified for this standalone implementation: `paper_budget_settings: true` reports those settings only, **not exact reproduction of archived paper scores or candidates**.

## Inputs

The included `examples/input_signatures.npy` is an archived synthetic input, not real music and not an archived Diffusion prediction. Custom features can be passed as:

```bash
python run.py --features path/to/signatures.npy --output runs/custom
```

The `.npy` array must be finite with shape `[N, 3, 64]`, with **at least 12 hits**, ordered in time. Its three rows are absolute log-mel window energies in dB over **0–80 ms, 80–300 ms, and 300–1100 ms**. Use the supplied frontend to compute this representation; do not supply already standardized or mean-centered features. Diffusion uses all three windows, while ranking/refinement use floor-`−70 dB`, mean-centered attack-window shapes.

For a **44,100 Hz mono, isolated percussive WAV**, provide a JSON file containing a `notes` list of objects with `onset_s` (or `time_s`). Times must be strictly increasing and inside the complete waveform; at least 12 selected hits are required.

```bash
python run.py --wav path/to/strikes.wav --onsets path/to/onsets.json --output runs/audio
```

Optional `--start` and `--end` select onsets in `[start, end)` using absolute seconds; they do not rebase timestamps. The frontend does **not** perform onset detection, source separation, resampling, or gain/time alignment against a reconstruction. It extracts 1.1-second windows and zero-pads at the waveform end.

## Outputs

- `result.json`: configuration, candidate validity/scores, best geometry, and per-hit actions. Geometry dimensions/offsets in `best.geometry_si` are in **metres**. Each `best.strike_codes` row is zero-based `[instrument_slot, striker_material, velocity, position]`; the action vocabulary is in `muyuset/physics/actions.py`.
- `trace.npz`: original/refined standardized latents, subset and elite indices, surrogate loss history, and the retained iteration of each track. The canonical slot order used in `result.json` can differ from a latent's stored order in this trace.

If no candidate passes the geometry and usage checks, the script writes `status: "no_feasible_candidate"`, sets `best` to `null`, and exits with code **2**. It does not export an invalid geometry as a successful result.

## Code map

- `run.py`: command-line entry point.
- `muyuset/pipeline.py`: the complete four-stage workflow.
- `muyuset/diffusion.py` and `refinement.py`: DDIM sampling and geometry-only optimization.
- `muyuset/surrogate.py` and `renderer.py`: learned forward model and authoritative scoring.
- `muyuset/physics/`: the self-contained original geometry/contact/acoustic backend; not a finite-element simulator.

Flow inference, training datasets, retrieval banks, paper assets, and unrelated experiments are intentionally excluded. Training loss helpers remain for architecture compatibility, but this repository does not provide a training workflow or reproduce every paper experiment.

## Verification and checkpoints

```bash
python -m unittest discover -s tests -v
```

Verified locally with Python 3.11.15, PyTorch 2.11.0, NumPy 2.4.4, and SciPy 1.17.1: **35 tests passed**, a CPU smoke run passed, and the full default 128-attempt run passed on CUDA. The extracted backend's 135 signatures for an audited geometry triple were also identical to the archived implementation.

The two included checkpoints total approximately **30 MiB**. Inference loads them with `torch.load(..., weights_only=True)` and strict architecture/statistics checks. Training paths and unused training configuration were removed; source SHA-256 fingerprints were retained. `tools/export_weights.py` is an optional maintainer utility; it requires explicit trust confirmation before reading original pickle-enabled training checkpoints. Never use that utility on untrusted downloads.

## Before publishing

See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md). **A code/weights license has not been selected**; the repository is prepared for upload but has not been published. Choose the license and confirm asset rights before making it public. `.gitignore` excludes run outputs, caches, and local ZIP/PDF/video files; weights and the synthetic example are intentionally included.
