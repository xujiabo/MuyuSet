"""Optional maintainer utility to export inference-only public assets.

Ordinary users use the included weights and example; they do not need the
original training archive. Source checkpoints are loaded with pickle enabled
ONLY after explicit --trust-local-checkpoints confirmation. Never run this
utility on untrusted downloads. Public outputs use tensors and basic Python
types and can be loaded with torch.load(..., weights_only=True).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch


COMMON_KEYS = (
    "format", "model_kwargs", "model_state", "latent_mean", "latent_std",
)
EXPECTED_FORMATS = {
    "diffusion": "muyu-audible-joint-conditional-diffusion-v1",
    "surrogate": "muyu-audible-forward-surrogate-v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_value(value: Any) -> Any:
    """Remove NumPy pickle globals and move all tensor storage to CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not allowed in public weights")
        return torch.from_numpy(np.array(value, copy=True)).cpu()
    if isinstance(value, np.generic):
        return safe_value(value.item())
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Public dictionary keys must be strings")
        return {key: safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(safe_value(item) for item in value)
    if isinstance(value, list):
        return [safe_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported public metadata type: {type(value).__name__}")


def export_checkpoint(source: Path, destination: Path, kind: str) -> None:
    # This trust boundary is intentional: archived NumPy metadata requires
    # pickle, whereas ordinary inference loads only the cleaned safe output.
    original = torch.load(source, map_location="cpu", weights_only=False)
    if original.get("format") != EXPECTED_FORMATS[kind]:
        raise ValueError(f"Unexpected {kind} source checkpoint format")
    keys = COMMON_KEYS + (
        ("feature_mean", "feature_std")
        if kind == "diffusion" else ("action_codes",)
    )
    missing = [key for key in keys if key not in original]
    if missing:
        raise ValueError(f"Missing required inference keys: {missing}")
    public = {key: safe_value(original[key]) for key in keys}
    # Retain a provenance fingerprint, never archive paths or training config.
    public["source_sha256"] = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public, destination)
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    if torch.serialization.get_unsafe_globals_in_checkpoint(destination):
        raise RuntimeError("Export unexpectedly contains unsafe pickle globals")
    for name, tensor in public["model_state"].items():
        if not torch.equal(tensor, loaded["model_state"][name]):
            raise RuntimeError(f"Export changed model tensor: {name}")
    print(f"{kind}: {destination.name}, {destination.stat().st_size:,} bytes")


def export_example(source: Path, destination: Path) -> None:
    signatures = np.load(source, allow_pickle=False)
    if signatures.dtype != np.float32 or signatures.shape != (48, 3, 64):
        raise ValueError("Expected the audited float32 [48,3,64] synthetic input")
    if not np.isfinite(signatures).all():
        raise ValueError("Synthetic input contains non-finite values")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError("Example copy checksum mismatch")
    print(f"example: {destination.name}, {destination.stat().st_size:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-source", type=Path, required=True)
    parser.add_argument("--surrogate-source", type=Path, required=True)
    parser.add_argument("--synthetic-input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--trust-local-checkpoints", action="store_true",
                        help="Confirm both sources are trusted local training files")
    args = parser.parse_args()
    if not args.trust_local_checkpoints:
        parser.error("Explicit --trust-local-checkpoints is required for pickle loading")
    sources = [args.diffusion_source, args.surrogate_source, args.synthetic_input]
    if not all(path.is_file() for path in sources):
        parser.error("All three source files must exist")
    destinations = [
        args.output_root / "weights" / "diffusion.pt",
        args.output_root / "weights" / "surrogate.pt",
        args.output_root / "examples" / "input_signatures.npy",
    ]
    if any(destination.resolve() == source.resolve()
           for destination in destinations for source in sources):
        parser.error("Outputs must not overwrite a source archive asset")
    export_checkpoint(sources[0], destinations[0], "diffusion")
    export_checkpoint(sources[1], destinations[1], "surrogate")
    export_example(sources[2], destinations[2])


if __name__ == "__main__":
    main()
