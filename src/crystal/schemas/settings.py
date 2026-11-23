# noqa: D100
from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict, Field

from crystal.constants.process import (
    DEFAULT_NUMBER_OF_GPUS,
    DEFAULT_NUMBER_OF_PROCESSES,
    MAX_NUMBER_OF_GPUS,
)
from crystal.schemas.scratch_execution import LocalScratchPolicy
from crystal.schemas.serialization import MSONableBaseModel


class Settings(MSONableBaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SimulationSettings(Settings):
    """Base class to keep all the simulation parameters."""


class ExecutorSettings(Settings):
    """Class to keep all the run time related parameters."""

    number_of_processes: int = Field(default=DEFAULT_NUMBER_OF_PROCESSES)
    total_memory_in_gb: float = Field(default=4)
    maximum_walltime_in_minutes: float = Field(default=1440)
    walltime_safety_margin_in_minutes: float = Field(default=5, ge=0)
    executable: Path | None = Field(default=None)
    number_of_gpus: int = Field(
        default=DEFAULT_NUMBER_OF_GPUS, ge=0, le=MAX_NUMBER_OF_GPUS
    )
    local_scratch_policy: LocalScratchPolicy = Field(default_factory=LocalScratchPolicy)


class MakerSettings(Settings):  # noqa: D101
    simulation_settings: SimulationSettings
    executor_settings: ExecutorSettings = Field(default_factory=ExecutorSettings)
