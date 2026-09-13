"""Parameterized 3-D geometry proxy for a mokugyo.

The geometry is intentionally simple: an ellipsoidal wooden shell, an offset
ellipsoidal air cavity, and a rounded horizontal mouth slot that connects the
cavity to the exterior.  It is useful for dataset prototyping and for linking
shape variables to the lumped acoustic model.  It is not a replacement for a
CT scan, CAD model of a specific instrument, or finite-element analysis.

Only the analytic descriptors and calibrated parameter mapping are retained;
surface meshing and STL-export branches are intentionally excluded.
"""

from dataclasses import asdict, dataclass, replace


from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .model import MuyuParameters


@dataclass(frozen=True)
class MuyuGeometry:
    """Dimensions of the ellipsoidal geometry proxy, in metres.

    Coordinates are ``x`` left--right, ``y`` back--front (the mouth is on the
    negative-y side), and ``z`` bottom--top.
    """

    outer_length_m: float = 0.390
    outer_depth_m: float = 0.370
    outer_height_m: float = 0.340

    cavity_length_m: float = 0.340
    cavity_depth_m: float = 0.320
    cavity_height_m: float = 0.316
    cavity_offset_y_m: float = 0.005
    cavity_offset_z_m: float = -0.002

    mouth_width_m: float = 0.200
    mouth_height_m: float = 0.030
    mouth_center_z_m: float = -0.025

    minimum_wall_m: float = 0.006

    def __post_init__(self) -> None:
        dimension_names = (
            "outer_length_m",
            "outer_depth_m",
            "outer_height_m",
            "cavity_length_m",
            "cavity_depth_m",
            "cavity_height_m",
            "mouth_width_m",
            "mouth_height_m",
            "minimum_wall_m",
        )
        for name in dimension_names:
            if getattr(self, name) <= 0.0:
                raise ValueError("{} must be positive".format(name))

        outer = np.asarray(self.outer_dimensions_m)
        cavity = np.asarray(self.cavity_dimensions_m)
        offsets = np.asarray((0.0, self.cavity_offset_y_m, self.cavity_offset_z_m))
        remaining_half_wall = 0.5 * (outer - cavity) - np.abs(offsets)
        if np.any(remaining_half_wall < self.minimum_wall_m):
            raise ValueError(
                "The cavity violates minimum_wall_m; remaining half-walls are {} m".format(
                    np.round(remaining_half_wall, 6)
                )
            )
        if self.mouth_height_m >= self.mouth_width_m:
            raise ValueError("mouth_width_m must exceed mouth_height_m")
        if self.mouth_width_m >= self.cavity_length_m:
            raise ValueError("The mouth must be narrower than the cavity")
        if abs(self.mouth_center_z_m - self.cavity_offset_z_m) >= (
            0.5 * self.cavity_height_m - 0.5 * self.mouth_height_m
        ):
            raise ValueError("The mouth does not intersect the cavity")

    @property
    def outer_dimensions_m(self) -> Tuple[float, float, float]:
        return (self.outer_length_m, self.outer_depth_m, self.outer_height_m)

    @property
    def cavity_dimensions_m(self) -> Tuple[float, float, float]:
        return (self.cavity_length_m, self.cavity_depth_m, self.cavity_height_m)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class GeometryMetrics:
    """Deterministic shape descriptors used by the acoustic proxy mapping."""

    outer_volume_m3: float
    cavity_volume_m3: float
    solid_volume_m3: float
    physical_mouth_area_m2: float
    mouth_path_length_m: float
    mean_wall_thickness_m: float
    characteristic_span_m: float
    projected_body_area_m2: float
    implied_wood_density_kg_m3: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _ellipsoid_volume(dimensions_m: Sequence[float]) -> float:
    return float(np.pi / 6.0 * np.prod(np.asarray(dimensions_m, dtype=float)))


def _capsule_area(total_width_m: float, height_m: float) -> float:
    straight_length = total_width_m - height_m
    radius = 0.5 * height_m
    return float(straight_length * height_m + np.pi * radius ** 2)


def mouth_path_length(geometry: MuyuGeometry) -> float:
    """Return the front-wall path traversed by the centre of the mouth."""

    z = geometry.mouth_center_z_m
    outer_z = 2.0 * z / geometry.outer_height_m
    inner_z = 2.0 * (z - geometry.cavity_offset_z_m) / geometry.cavity_height_m
    outer_front_y = -0.5 * geometry.outer_depth_m * np.sqrt(
        max(0.0, 1.0 - outer_z ** 2)
    )
    inner_front_y = geometry.cavity_offset_y_m - 0.5 * geometry.cavity_depth_m * np.sqrt(
        max(0.0, 1.0 - inner_z ** 2)
    )
    path = inner_front_y - outer_front_y
    if path <= 0.0:
        raise ValueError("The mouth centre does not traverse a positive wall thickness")
    return float(path)


def geometry_metrics(
    geometry: MuyuGeometry,
    reference_body_mass_kg: float = MuyuParameters().body_mass_kg,
) -> GeometryMetrics:
    """Compute analytic descriptors of the simple ellipsoid/capsule proxy."""

    outer_volume = _ellipsoid_volume(geometry.outer_dimensions_m)
    cavity_volume = _ellipsoid_volume(geometry.cavity_dimensions_m)
    mouth_area = _capsule_area(geometry.mouth_width_m, geometry.mouth_height_m)
    path_length = mouth_path_length(geometry)
    solid_volume = outer_volume - cavity_volume - mouth_area * path_length
    if solid_volume <= 0.0:
        raise ValueError("The requested cavity and mouth remove the entire body")

    outer = np.asarray(geometry.outer_dimensions_m)
    cavity = np.asarray(geometry.cavity_dimensions_m)
    mean_wall = float(np.mean(0.5 * (outer - cavity)))
    characteristic_span = float(np.prod(outer) ** (1.0 / 3.0))
    projected_area = float(geometry.outer_length_m * geometry.outer_height_m)

    return GeometryMetrics(
        outer_volume_m3=outer_volume,
        cavity_volume_m3=cavity_volume,
        solid_volume_m3=solid_volume,
        physical_mouth_area_m2=mouth_area,
        mouth_path_length_m=path_length,
        mean_wall_thickness_m=mean_wall,
        characteristic_span_m=characteristic_span,
        projected_body_area_m2=projected_area,
        implied_wood_density_kg_m3=reference_body_mass_kg / solid_volume,
    )


def parameters_from_geometry(
    metrics: GeometryMetrics,
    reference_metrics: Optional[GeometryMetrics] = None,
    base: MuyuParameters = MuyuParameters(),
) -> MuyuParameters:
    """Map geometry ratios to the calibrated two-DOF acoustic model.

    The paper's fitted parameters define the reference point.  Ratios are used
    because physical mouth area, end correction, shell stiffness and effective
    radiating area cannot be identified exactly from this coarse geometry.
    The stiffness law is a thin-shell-inspired proxy proportional to
    ``wall_thickness**3 / span**2``; it must later be replaced or calibrated by
    FEM/measurements before making quantitative claims away from the reference.
    """

    if reference_metrics is None:
        reference_metrics = geometry_metrics(MuyuGeometry(), base.body_mass_kg)

    cavity_ratio = metrics.cavity_volume_m3 / reference_metrics.cavity_volume_m3
    mouth_area_ratio = (
        metrics.physical_mouth_area_m2 / reference_metrics.physical_mouth_area_m2
    )
    path_ratio = metrics.mouth_path_length_m / reference_metrics.mouth_path_length_m
    solid_ratio = metrics.solid_volume_m3 / reference_metrics.solid_volume_m3
    span_ratio = metrics.characteristic_span_m / reference_metrics.characteristic_span_m
    wall_ratio = (
        metrics.mean_wall_thickness_m / reference_metrics.mean_wall_thickness_m
    )
    projected_ratio = (
        metrics.projected_body_area_m2 / reference_metrics.projected_body_area_m2
    )

    body_mass = base.body_mass_kg * solid_ratio
    body_added_mass = base.body_added_mass_kg * span_ratio ** 3
    stiffness_ratio = wall_ratio ** 3 / span_ratio ** 2
    body_stiffness = base.body_stiffness_n_m * stiffness_ratio
    equivalent_mass_ratio = (
        (body_mass + body_added_mass) / base.equivalent_body_mass_kg
    )
    resistance_ratio = np.sqrt(stiffness_ratio * equivalent_mass_ratio)

    return replace(
        base,
        body_area_m2=base.body_area_m2 * projected_ratio,
        body_mass_kg=body_mass,
        body_added_mass_kg=body_added_mass,
        body_stiffness_n_m=body_stiffness,
        body_resistance_ns_m=base.body_resistance_ns_m * resistance_ratio,
        cavity_volume_m3=base.cavity_volume_m3 * cavity_ratio,
        port_area_m2=base.port_area_m2 * mouth_area_ratio,
        port_length_m=base.port_length_m * path_ratio,
    )
