"""Audible modal demonstrations derived from the lumped mokugyo model."""

from dataclasses import dataclass

from typing import Tuple

import numpy as np



from .model import MuyuParameters


@dataclass(frozen=True)
class StrikeProfile:
    """How one strike location excites the fixed modes of the same body.

    The first two modes are predicted by the two-DOF model and the third is the
    paper's measured 402 Hz body mode.  The last two short-lived modes are
    perceptual placeholders used only to demonstrate the brighter attack heard
    at a hard carved ridge; they are not claimed as measured frequencies.
    """

    key: str
    label_zh: str
    description_zh: str
    modal_gains: Tuple[float, float, float, float, float]
    modal_phases_rad: Tuple[float, float, float, float, float]
    attack_level: float
    attack_band_hz: Tuple[float, float]
    attack_decay_s: float

    def __post_init__(self) -> None:
        if len(self.modal_gains) != 5 or len(self.modal_phases_rad) != 5:
            raise ValueError("modal_gains and modal_phases_rad must each contain five values")
        if self.attack_level < 0.0 or self.attack_decay_s <= 0.0:
            raise ValueError("attack parameters must be non-negative/positive")
        if not 0.0 < self.attack_band_hz[0] < self.attack_band_hz[1]:
            raise ValueError("attack_band_hz must be a positive ascending pair")


FIVE_STRIKE_PROFILES = (
    StrikeProfile(
        key="crown_centre",
        label_zh="顶部正中",
        description_zh="两个低频耦合模态较均衡，是基准木鱼声",
        modal_gains=(1.00, 0.72, 0.22, 0.06, 0.025),
        modal_phases_rad=(0.0, 0.0, 0.15, 0.0, 0.0),
        attack_level=0.09,
        attack_band_hz=(650.0, 5000.0),
        attack_decay_s=0.0035,
    ),
    StrikeProfile(
        key="mouth_rim",
        label_zh="嘴边",
        description_zh="空腔一侧的低模态更强，声音更空、更暗",
        modal_gains=(1.25, 0.32, 0.08, 0.018, 0.006),
        modal_phases_rad=(0.0, 0.1, 0.0, 0.0, 0.0),
        attack_level=0.045,
        attack_band_hz=(350.0, 2300.0),
        attack_decay_s=0.0045,
    ),
    StrikeProfile(
        key="back_shell",
        label_zh="背部壳体",
        description_zh="壳体一侧的低模态和 402 Hz 模态更突出",
        modal_gains=(0.34, 1.18, 0.30, 0.075, 0.025),
        modal_phases_rad=(0.0, 0.0, 0.25, 0.0, 0.0),
        attack_level=0.10,
        attack_band_hz=(750.0, 5600.0),
        attack_decay_s=0.0032,
    ),
    StrikeProfile(
        key="side_belly",
        label_zh="侧腹",
        description_zh="两个邻近低模态近似反相，拍频感明显且偏闷",
        modal_gains=(0.88, 0.78, 0.10, 0.025, 0.008),
        modal_phases_rad=(0.0, np.pi, 0.0, 0.0, 0.0),
        attack_level=0.055,
        attack_band_hz=(400.0, 3000.0),
        attack_decay_s=0.0040,
    ),
    StrikeProfile(
        key="carved_ridge",
        label_zh="雕刻硬脊",
        description_zh="高阶短促振型和接触瞬态最强，听起来最亮、最硬",
        modal_gains=(0.32, 0.48, 0.72, 0.32, 0.14),
        modal_phases_rad=(0.0, 0.0, 0.10, 0.25, 0.40),
        attack_level=0.24,
        attack_band_hz=(1200.0, 8000.0),
        attack_decay_s=0.0024,
    ),
)


def damped_modal_frequencies(params: MuyuParameters) -> Tuple[np.ndarray, np.ndarray]:
    """Return damped frequencies and exponential decay rates from state poles."""

    mm = params.equivalent_body_mass_kg
    ml = params.referred_port_mass_kg
    sm = params.body_stiffness_n_m
    sc = params.cavity_stiffness_n_m
    rm = params.body_resistance_ns_m

    state_matrix = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-(sm + sc) / mm, sc / mm, -rm / mm, 0.0],
            [sc / ml, -sc / ml, 0.0, 0.0],
        ]
    )
    poles = np.linalg.eigvals(state_matrix)
    positive = poles[np.imag(poles) > 0.0]
    order = np.argsort(np.imag(positive))
    positive = positive[order]
    frequencies = np.imag(positive) / (2.0 * np.pi)
    decay_rates = -np.real(positive)
    return frequencies.astype(float), decay_rates.astype(float)
