"""Frozen eight-dimensional geometry coordinates and declared OOD bounds."""
from typing import Sequence
import numpy as np
from .geometry import MuyuGeometry

GEOMETRY_LATENT_NAMES = (
    "log_outer_length_ratio",
    "log_outer_depth_ratio",
    "log_outer_height_ratio",
    "log_wall_x_ratio",
    "log_wall_y_ratio",
    "log_wall_z_ratio",
    "log_mouth_width_ratio",
    "log_mouth_height_ratio",
)

_OOD_RATIO_BOUNDS = np.asarray(
    (
        (0.78, 1.24),
        (0.78, 1.24),
        (0.78, 1.24),
        (0.58, 3.00),
        (0.58, 3.00),
        (0.58, 3.00),
        (0.52, 1.50),
        (0.52, 1.55),
    ),
    dtype=np.float64,
)

OOD_LATENT_LOWER = np.log(_OOD_RATIO_BOUNDS[:, 0])
OOD_LATENT_UPPER = np.log(_OOD_RATIO_BOUNDS[:, 1])


def factorized_latent_to_geometry(
    latent: Sequence[float],
    reference: MuyuGeometry = MuyuGeometry(),
) -> MuyuGeometry:
    """Decode the v3 latent, rejecting combinations that are not legal solids."""

    values = np.asarray(latent, dtype=np.float64)
    if values.shape != (len(GEOMETRY_LATENT_NAMES),):
        raise ValueError("latent must contain eight values")
    if not np.all(np.isfinite(values)):
        raise ValueError("latent values must be finite")

    reference_outer = np.asarray(reference.outer_dimensions_m, dtype=np.float64)
    reference_cavity = np.asarray(reference.cavity_dimensions_m, dtype=np.float64)
    reference_half_walls = 0.5 * (reference_outer - reference_cavity)
    outer = reference_outer * np.exp(values[:3])
    half_walls = reference_half_walls * np.exp(values[3:6])
    cavity = outer - 2.0 * half_walls
    if np.any(cavity <= 0.0):
        raise ValueError("latent produces non-positive cavity dimensions")

    size_scale = float(np.prod(outer / reference_outer) ** (1.0 / 3.0))
    return MuyuGeometry(
        outer_length_m=float(outer[0]),
        outer_depth_m=float(outer[1]),
        outer_height_m=float(outer[2]),
        cavity_length_m=float(cavity[0]),
        cavity_depth_m=float(cavity[1]),
        cavity_height_m=float(cavity[2]),
        cavity_offset_y_m=reference.cavity_offset_y_m * size_scale,
        cavity_offset_z_m=reference.cavity_offset_z_m * size_scale,
        mouth_width_m=reference.mouth_width_m * float(np.exp(values[6])),
        mouth_height_m=reference.mouth_height_m * float(np.exp(values[7])),
        mouth_center_z_m=reference.mouth_center_z_m * size_scale,
        minimum_wall_m=reference.minimum_wall_m,
    )

