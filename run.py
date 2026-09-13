"""Run the minimal MuyuSet pipeline from an unpacked GitHub checkout."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

# Limit per-worker BLAS parallelism; only this process tree is affected.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from muyuset.frontend import load_audio, load_signatures
from muyuset.pipeline import Config, run

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--features", type=Path)
    source.add_argument("--wav", type=Path)
    parser.add_argument("--onsets", type=Path, help="locked-onset JSON, required with --wav")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--diffusion", type=Path, default=ROOT / "weights/diffusion.pt")
    parser.add_argument("--surrogate", type=Path, default=ROOT / "weights/surrogate.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/demo")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for name, value in (("subsets", 6), ("samples", 16), ("ddim-steps", 48),
                        ("tracks", 32), ("steps", 200), ("seed", 20261003), ("workers", 1)):
        parser.add_argument("--" + name, type=int, default=value)
    parser.add_argument("--learning-rate", type=float, default=.025)
    args = parser.parse_args()
    if args.wav is not None and args.onsets is None:
        parser.error("--wav requires --onsets")
    if args.wav is None and (args.onsets is not None or args.start != 0 or args.end is not None):
        parser.error("--onsets/--start/--end are only used with --wav")
    config = Config(**{name: getattr(args, name) for name in asdict(Config())})
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is not empty; choose a new --output to preserve existing results")
    if args.wav is not None:
        signatures, onsets = load_audio(args.wav, args.onsets, args.start, args.end)
    else:
        signatures = load_signatures(args.features or ROOT / "examples/input_signatures.npy")
        onsets = None
    result, arrays = run(signatures, args.diffusion, args.surrogate, config, device,
                         progress=lambda message: print(message, flush=True))
    result["config"] = asdict(config)
    if onsets is not None:
        result["locked_onsets_s"] = onsets.tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "result.json").open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
    np.savez_compressed(args.output / "trace.npz", **arrays)
    print(f"{result['status']}: {result['exact_candidate_attempts']} candidate attempts; {args.output}")
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
