"""Two-degree-of-freedom shell--cavity model for a mokugyo.

The implementation follows Eqs. (1)--(7) and the fitted values in Table I of
Sunohara et al., JASA 117(4), 2247--2258 (2005), DOI: 10.1121/1.1868192.

The model explains the two close low-frequency resonances caused by coupling
between the vibrating wooden body and the Helmholtz-like cavity/port.  It does
not contain the measured 402 Hz divided-body mode; that mode is exposed only
as an optional perceptual component in :mod:`muyu.synthesis`.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np



@dataclass(frozen=True)
class MuyuParameters:
    """Parameters of the rounded, fitted 37 cm mokugyo model.

    Geometry-sensitive quantities are referenced to the paper's fitted model.
    When cavity volume changes, cavity stiffness scales as ``1 / volume``.
    When port area changes, the fitted end-correction mass scales as
    ``area**1.5``, as implied by Eq. (9) for an equivalent circular port.
    """

    body_area_m2: float = 0.070
    body_mass_kg: float = 5.48
    body_added_mass_kg: float = 0.616
    body_stiffness_n_m: float = 6.76e6
    body_resistance_ns_m: float = 210.0

    cavity_volume_m3: float = 1.80e-2
    reference_cavity_volume_m3: float = 1.80e-2
    reference_cavity_stiffness_n_m: float = 1.00e5

    port_area_m2: float = 1.50e-2
    port_length_m: float = 0.103
    reference_port_area_m2: float = 1.50e-2
    reference_port_added_mass_kg: float = 2.36e-3

    air_density_kg_m3: float = 1.21
    sound_speed_m_s: float = 343.0
    body_observer_distance_m: float = 1.30
    port_observer_distance_m: float = 1.00
    impact_force_peak_n: float = 170.0

    measured_f1_hz: float = 155.0
    measured_f2_hz: float = 177.0
    measured_f3_hz: float = 402.0
    measured_peak_level_difference_db: float = 2.72

    def __post_init__(self) -> None:
        positive_fields = (
            "body_area_m2",
            "body_mass_kg",
            "body_added_mass_kg",
            "body_stiffness_n_m",
            "body_resistance_ns_m",
            "cavity_volume_m3",
            "reference_cavity_volume_m3",
            "reference_cavity_stiffness_n_m",
            "port_area_m2",
            "port_length_m",
            "reference_port_area_m2",
            "reference_port_added_mass_kg",
            "air_density_kg_m3",
            "sound_speed_m_s",
            "body_observer_distance_m",
            "port_observer_distance_m",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError("{} must be positive".format(name))

    @property
    def equivalent_body_mass_kg(self) -> float:
        return self.body_mass_kg + self.body_added_mass_kg

    @property
    def cavity_stiffness_n_m(self) -> float:
        return (
            self.reference_cavity_stiffness_n_m
            * self.reference_cavity_volume_m3
            / self.cavity_volume_m3
        )

    @property
    def port_air_mass_kg(self) -> float:
        return self.air_density_kg_m3 * self.port_area_m2 * self.port_length_m

    @property
    def port_added_mass_kg(self) -> float:
        area_ratio = self.port_area_m2 / self.reference_port_area_m2
        return self.reference_port_added_mass_kg * area_ratio ** 1.5

    @property
    def referred_port_mass_kg(self) -> float:
        area_ratio = self.body_area_m2 / self.port_area_m2
        return area_ratio ** 2 * (self.port_air_mass_kg + self.port_added_mass_kg)

    def derived_values(self) -> Dict[str, float]:
        """Return the quantities most useful for checking Table I."""

        fm, fc = uncoupled_frequencies(self)
        return {
            "equivalent_body_mass_kg": self.equivalent_body_mass_kg,
            "cavity_stiffness_n_m": self.cavity_stiffness_n_m,
            "port_air_mass_kg": self.port_air_mass_kg,
            "port_added_mass_kg": self.port_added_mass_kg,
            "referred_port_mass_kg": self.referred_port_mass_kg,
            "uncoupled_body_frequency_hz": fm,
            "uncoupled_cavity_frequency_hz": fc,
        }


def uncoupled_frequencies(params: MuyuParameters) -> Tuple[float, float]:
    """Return uncoupled body and cavity frequencies in Hz."""

    fm = np.sqrt(
        params.body_stiffness_n_m / params.equivalent_body_mass_kg
    ) / (2.0 * np.pi)
    fc = np.sqrt(
        params.cavity_stiffness_n_m / params.referred_port_mass_kg
    ) / (2.0 * np.pi)
    return float(fm), float(fc)

