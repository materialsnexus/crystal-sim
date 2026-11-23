# noqa: D100
from enum import StrEnum, auto

from crystal.schemas.settings import SimulationSettings


class OptimizerType(StrEnum):  # noqa: D101
    BFGS = auto()
    FIRE = auto()
    LBFGS = auto()
    FIRE2 = auto()


class RelaxOptimizerSettings(SimulationSettings):  # noqa: D101
    optimizer_type: OptimizerType = OptimizerType.FIRE
    max_steps: int = 100
    force_threshold_in_ev_per_angstrom: float = 0.05
    store_interval: int = 1
