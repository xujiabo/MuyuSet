"""Nonlinear striker--mokugyo contact and contact-driven sound rendering.

The collision uses a Hertz elastic term with Hunt--Crossley-style hysteretic
damping.  It is deliberately a compact research proxy: the mokugyo is treated
as locally isotropic at the contact point, while real wood is anisotropic and
its carved surface has position-dependent compliance.
"""

from dataclasses import asdict, dataclass
import math
from typing import Dict, Tuple

import numpy as np
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
from scipy.integrate import solve_ivp
from scipy.signal import resample_poly

from .model import MuyuParameters, uncoupled_frequencies
from .synthesis import StrikeProfile, damped_modal_frequencies


AUDIBLE_RENDERER_VERSION = "physical-contact-wood-damping-v2"


class ContactIntegrationError(RuntimeError):
    """Expected numerical failure of the Hertz separation integration."""


@dataclass(frozen=True)
class ElasticMaterial:
    key: str
    label_zh: str
    young_modulus_pa: float
    poisson_ratio: float
    density_kg_m3: float
    restitution: float

    def __post_init__(self) -> None:
        if self.young_modulus_pa <= 0.0 or self.density_kg_m3 <= 0.0:
            raise ValueError("Material modulus and density must be positive")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if not 0.0 < self.restitution <= 1.0:
            raise ValueError("restitution must be in (0, 1]")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


MOKUGYO_WOOD = ElasticMaterial(
    key="mokugyo_wood",
    label_zh="木鱼硬木",
    young_modulus_pa=9.0e9,
    poisson_ratio=0.30,
    density_kg_m3=720.0,
    restitution=0.58,
)

STRIKER_MATERIALS = (
    ElasticMaterial(
        key="hardwood",
        label_zh="硬木棒",
        young_modulus_pa=13.0e9,
        poisson_ratio=0.30,
        density_kg_m3=760.0,
        restitution=0.70,
    ),
    ElasticMaterial(
        key="softwood",
        label_zh="软木棒",
        young_modulus_pa=5.0e9,
        poisson_ratio=0.32,
        density_kg_m3=500.0,
        restitution=0.50,
    ),
    ElasticMaterial(
        key="rubber",
        label_zh="橡胶棒头",
        young_modulus_pa=12.0e6,
        poisson_ratio=0.49,
        density_kg_m3=1100.0,
        restitution=0.32,
    ),
)


@dataclass(frozen=True)
class Striker:
    material: ElasticMaterial
    mass_kg: float = 0.025
    tip_radius_m: float = 0.006
    speed_m_s: float = 0.90
    incidence_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0 or self.tip_radius_m <= 0.0 or self.speed_m_s <= 0.0:
            raise ValueError("Striker mass, radius and speed must be positive")
        if not 0.0 <= self.incidence_angle_deg < 90.0:
            raise ValueError("incidence_angle_deg must be in [0, 90)")

    @property
    def normal_speed_m_s(self) -> float:
        return float(
            self.speed_m_s * np.cos(np.deg2rad(self.incidence_angle_deg))
        )

    def to_dict(self) -> Dict[str, object]:
        values = asdict(self)
        values["normal_speed_m_s"] = self.normal_speed_m_s
        return values


@dataclass(frozen=True)
class ContactResult:
    time_s: np.ndarray
    force_n: np.ndarray
    indentation_m: np.ndarray
    indentation_velocity_m_s: np.ndarray
    contact_sample_rate_hz: int
    contact_duration_s: float
    peak_force_n: float
    impulse_ns: float
    effective_modulus_pa: float
    hertz_stiffness_n_m_3_2: float
    effective_mass_kg: float
    requested_restitution: float
    simulated_restitution: float

    def summary(self) -> Dict[str, float]:
        return {
            "contact_duration_s": self.contact_duration_s,
            "peak_force_n": self.peak_force_n,
            "impulse_ns": self.impulse_ns,
            "effective_modulus_pa": self.effective_modulus_pa,
            "hertz_stiffness_n_m_3_2": self.hertz_stiffness_n_m_3_2,
            "effective_mass_kg": self.effective_mass_kg,
            "requested_restitution": self.requested_restitution,
            "simulated_restitution": self.simulated_restitution,
        }


def effective_young_modulus(
    first: ElasticMaterial, second: ElasticMaterial
) -> float:
    """Return the Hertz effective modulus of two elastic materials."""

    compliance = (1.0 - first.poisson_ratio ** 2) / first.young_modulus_pa
    compliance += (1.0 - second.poisson_ratio ** 2) / second.young_modulus_pa
    return float(1.0 / compliance)


def simulate_hertz_impact(
    striker: Striker,
    body_material: ElasticMaterial = MOKUGYO_WOOD,
    body_effective_mass_kg: float = MuyuParameters().equivalent_body_mass_kg,
    contact_sample_rate_hz: int = 352800,
) -> ContactResult:
    """Integrate indentation until the striker separates from the body."""

    if body_effective_mass_kg <= 0.0 or contact_sample_rate_hz <= 0:
        raise ValueError("Body mass and sample rate must be positive")
    normal_speed = striker.normal_speed_m_s
    effective_mass = 1.0 / (1.0 / striker.mass_kg + 1.0 / body_effective_mass_kg)
    effective_modulus = effective_young_modulus(striker.material, body_material)
    hertz_stiffness = (
        4.0 / 3.0 * effective_modulus * np.sqrt(striker.tip_radius_m)
    )
    pair_restitution = float(
        np.sqrt(striker.material.restitution * body_material.restitution)
    )
    hysteresis = 3.0 * (1.0 - pair_restitution) / (
        2.0 * pair_restitution * normal_speed
    )

    def contact_force(indentation: float, velocity: float) -> float:
        if indentation <= 0.0:
            return 0.0
        elastic_force = hertz_stiffness * indentation ** 1.5
        return float(max(0.0, elastic_force * (1.0 + hysteresis * velocity)))

    def dynamics(_time: float, state: np.ndarray) -> Tuple[float, float]:
        force = contact_force(float(state[0]), float(state[1]))
        return float(state[1]), -force / effective_mass

    def separation_event(_time: float, state: np.ndarray) -> float:
        return float(state[0])

    separation_event.terminal = True
    separation_event.direction = -1.0

    elastic_time_estimate = 2.94 * (
        effective_mass ** 2
        / (
            max(striker.tip_radius_m, np.finfo(float).eps)
            * effective_modulus ** 2
            * normal_speed
        )
    ) ** 0.2
    integration_limit = float(np.clip(20.0 * elastic_time_estimate, 0.01, 0.20))
    initial_indentation = max(1.0e-12, normal_speed * 1.0e-10)
    solution = solve_ivp(
        dynamics,
        (0.0, integration_limit),
        (initial_indentation, normal_speed),
        events=separation_event,
        dense_output=True,
        rtol=2.0e-9,
        atol=1.0e-11,
        max_step=max(elastic_time_estimate / 80.0, 1.0e-8),
    )
    if solution.t_events[0].size == 0:
        raise ContactIntegrationError("Contact integration did not reach separation")
    duration = float(solution.t_events[0][0])
    separation_state = solution.y_events[0][0]

    sample_count = int(np.ceil(duration * contact_sample_rate_hz)) + 2
    sample_times = np.arange(sample_count, dtype=float) / contact_sample_rate_hz
    evaluation_times = np.minimum(sample_times, duration)
    states = solution.sol(evaluation_times)
    forces = np.asarray(
        [contact_force(float(value), float(speed)) for value, speed in states.T]
    )
    forces[sample_times > duration] = 0.0
    indentation = np.maximum(states[0], 0.0)
    indentation[sample_times > duration] = 0.0
    indentation_velocity = states[1]
    impulse = float(_trapezoid(forces, sample_times))

    return ContactResult(
        time_s=sample_times,
        force_n=forces,
        indentation_m=indentation,
        indentation_velocity_m_s=indentation_velocity,
        contact_sample_rate_hz=contact_sample_rate_hz,
        contact_duration_s=duration,
        peak_force_n=float(np.max(forces)),
        impulse_ns=impulse,
        effective_modulus_pa=effective_modulus,
        hertz_stiffness_n_m_3_2=float(hertz_stiffness),
        effective_mass_kg=float(effective_mass),
        requested_restitution=pair_restitution,
        simulated_restitution=float(max(0.0, -separation_state[1] / normal_speed)),
    )


_EXTRA_LOCATION_GAINS = {
    "crown_centre": (0.180, 0.140, 0.100, 0.065, 0.035),
    "mouth_rim": (0.040, 0.025, 0.015, 0.008, 0.004),
    "back_shell": (0.220, 0.170, 0.120, 0.070, 0.035),
    "side_belly": (0.060, 0.040, 0.025, 0.012, 0.006),
    "carved_ridge": (0.450, 0.340, 0.240, 0.140, 0.075),
}


def contact_modal_palette(
    params: MuyuParameters = MuyuParameters(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return body poles used by the audible contact renderer.

    The compact shell--cavity state model damps only the wooden body.  For
    some sampled geometries this leaves the cavity-dominated pole almost
    lossless (multi-second or even minute-long T60), which sounds like a bell
    rather than a wooden fish.  The audible proxy therefore applies a
    conservative wood/radiation-loss ceiling of Q=35 to its two low modes.
    This does not alter the underlying analytical response model.
    """

    low_frequencies, low_decay = damped_modal_frequencies(params)
    maximum_low_mode_quality_factor = 35.0
    low_decay = np.maximum(
        low_decay,
        np.pi * low_frequencies / maximum_low_mode_quality_factor,
    )
    reference_body_frequency = uncoupled_frequencies(MuyuParameters())[0]
    body_frequency = uncoupled_frequencies(params)[0]
    shell_scale = body_frequency / reference_body_frequency
    higher_reference_frequencies = np.asarray(
        (params.measured_f3_hz, 1.55 * params.measured_f3_hz, 2.25 * params.measured_f3_hz,
         1350.0, 2100.0, 3200.0, 4800.0, 7000.0)
    )
    higher_quality_factors = np.asarray((20.0, 12.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0))
    higher_frequencies = higher_reference_frequencies * shell_scale
    higher_decay = np.pi * higher_frequencies / higher_quality_factors
    return (
        np.concatenate((low_frequencies, higher_frequencies)),
        np.concatenate((low_decay, higher_decay)),
    )


def strike_modal_weights(profile: StrikeProfile) -> Tuple[np.ndarray, np.ndarray]:
    """Return extended modal gains and phases for a registered strike location."""

    extra_gains = _EXTRA_LOCATION_GAINS.get(profile.key)
    if extra_gains is None:
        raise ValueError("No extended modal gains registered for {}".format(profile.key))
    gains = np.asarray(profile.modal_gains + extra_gains, dtype=float)
    phases = np.asarray(profile.modal_phases_rad + (0.0,) * 5, dtype=float)
    return gains, phases


def direct_contact_gain(profile: StrikeProfile) -> float:
    return float(0.35 + 1.20 * profile.attack_level)


def normalized_force_spectrum(
    result: ContactResult, frequencies_hz: np.ndarray
) -> np.ndarray:
    """Evaluate |F(f)| / impulse at arbitrary frequencies."""

    frequencies = np.asarray(frequencies_hz, dtype=float)
    phase = np.exp(
        -2j * np.pi * result.time_s[:, None] * frequencies[None, :]
    )
    spectrum = _trapezoid(result.force_n[:, None] * phase, result.time_s, axis=0)
    return np.abs(spectrum) / max(result.impulse_ns, np.finfo(float).eps)


def render_contact_excited_sound(
    result: ContactResult,
    profile: StrikeProfile,
    params: MuyuParameters = MuyuParameters(),
    sample_rate: int = 44100,
    duration_s: float = 1.15,
    reference_impulse_ns: float = 0.035,
    reference_peak_force_n: float = 170.0,
) -> np.ndarray:
    """Render a strike from its physical force pulse without peak normalization."""

    if (
        sample_rate <= 0
        or duration_s <= 0.0
        or reference_impulse_ns <= 0.0
        or reference_peak_force_n <= 0.0
    ):
        raise ValueError("Sample rate, duration, and force references must be positive")
    frequencies, decay_rates = contact_modal_palette(params)
    force_drive = normalized_force_spectrum(result, frequencies)
    gains, phases = strike_modal_weights(profile)

    time = np.arange(int(round(duration_s * sample_rate)), dtype=float) / sample_rate
    modal_sound = np.zeros_like(time)
    for gain, phase_value, drive, frequency, decay in zip(
        gains, phases, force_drive, frequencies, decay_rates
    ):
        modal_sound += gain * drive * np.exp(-decay * time) * np.sin(
            2.0 * np.pi * frequency * time + phase_value
        )

    # Preserve the large peak-force difference between hard wood and rubber.
    # The previous per-strike peak normalization made a 14 N rubber pulse
    # nearly as loud and crisp as a 140 N hardwood pulse.
    physical_contact = result.force_n / reference_peak_force_n
    divisor = math.gcd(result.contact_sample_rate_hz, sample_rate)
    direct_contact = resample_poly(
        physical_contact,
        sample_rate // divisor,
        result.contact_sample_rate_hz // divisor,
    )
    direct_track = np.zeros_like(time)
    usable = min(direct_track.shape[0], direct_contact.shape[0])
    direct_track[:usable] = direct_contact[:usable]
    direct_gain = direct_contact_gain(profile)

    modal_strength = result.impulse_ns / reference_impulse_ns
    sound = modal_strength * modal_sound + direct_gain * direct_track
    return sound.astype(np.float32)
