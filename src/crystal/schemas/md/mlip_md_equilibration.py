"""Schemas for MLIP MD equilibration configuration and task documents."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from crystal.schemas.base import BaseTaskDocument
from crystal.schemas.md.checks import MdEquilibrationCheck
from crystal.schemas.md.properties import MdEquilibrationProperties
from crystal.schemas.md.settings import MdRunSettings
from crystal.schemas.settings import MakerSettings, SimulationSettings


class MlipMdEquilibrationSimulationSettings(SimulationSettings):  # noqa: D101
    run_settings: MdRunSettings
    checks: list[MdEquilibrationCheck] = Field(default_factory=list)
    time_averaging_window: float = Field(
        default=0.5,
        description="Time window in ps for averaging properties over time.",
        gt=0.0,
    )

    @model_validator(mode="after")
    def validate_time_averaging_window(self) -> Self:  # noqa: D102
        if self.time_averaging_window >= self.run_settings.time.simulation_time_ps:
            error_msg = (
                "Time averaging window must be smaller than total simulation time."
            )
            raise ValueError(error_msg)
        return self


class MlipMdEquilibrationMakerSettings(MakerSettings):
    """Settings for an MLIP MD equilibration maker."""

    simulation_settings: MlipMdEquilibrationSimulationSettings


class MlipMdEquilibrationTaskDocument(BaseTaskDocument):  # noqa: D101
    properties: MdEquilibrationProperties | None = Field(
        description="Computed MD equilibration properties from an MLIP task"
    )

    @property
    def is_successful_md_equilibration(self) -> bool:  # noqa: D102
        return self.properties is not None and self.properties.all_checks_passed
