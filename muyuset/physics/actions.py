"""The original material-major, velocity-major, position-minor 45 actions."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

from .contact import ContactResult, STRIKER_MATERIALS, Striker, simulate_hertz_impact
from .synthesis import FIVE_STRIKE_PROFILES, StrikeProfile

VELOCITY_CONDITIONS = (("light", 0.45), ("medium", 0.90), ("strong", 1.80))


@dataclass(frozen=True)
class StrikeCondition:
    condition_id: int
    material_index: int
    velocity_index: int
    position_index: int
    material_key: str
    velocity_key: str
    position_key: str
    contact: ContactResult
    profile: StrikeProfile


@lru_cache(maxsize=1)
def build_conditions() -> Tuple[StrikeCondition, ...]:
    conditions = []
    for material_index, material in enumerate(STRIKER_MATERIALS):
        for velocity_index, (velocity_key, velocity) in enumerate(VELOCITY_CONDITIONS):
            contact = simulate_hertz_impact(Striker(material=material, speed_m_s=velocity))
            for position_index, profile in enumerate(FIVE_STRIKE_PROFILES):
                conditions.append(StrikeCondition(
                    condition_id=len(conditions),
                    material_index=material_index,
                    velocity_index=velocity_index,
                    position_index=position_index,
                    material_key=material.key,
                    velocity_key=velocity_key,
                    position_key=profile.key,
                    contact=contact,
                    profile=profile,
                ))
    return tuple(conditions)
