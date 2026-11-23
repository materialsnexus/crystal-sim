"""Pydantic models for MLIP configuration."""

from __future__ import annotations

from typing import Literal

from monty.json import MSONable
from pydantic import Field

from crystal.schemas.backend import RelaxBackend, SinglePointBackend
from crystal.schemas.base import BaseTaskDocument, MSONableModelMixin
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.properties import SinglePointProperties
from crystal.schemas.relax_settings import RelaxOptimizerSettings
from crystal.schemas.relaxation_constraints import RelaxationConstraints
from crystal.schemas.settings import MakerSettings, SimulationSettings

MATGL_MISSING_ERROR_MESSAGE = (
    "matgl with the DGL backend is required to run MLIP calculations. "
    "Install the Linux x86_64 MLIP dependencies for this workflow."
)


class MlipSinglePointSimulationSettings(SimulationSettings):
    """Settings for MLIP single-point simulations."""

    model: MlipModelRef


class MlipSinglePointMakerSettings(MakerSettings):  # noqa: D101
    backend: Literal[SinglePointBackend.mlip] = SinglePointBackend.mlip
    simulation_settings: MlipSinglePointSimulationSettings


class MlipRelaxSimulationSettings(
    MlipSinglePointSimulationSettings, RelaxOptimizerSettings
):
    """Settings for MLIP relaxation simulations."""

    constraints: RelaxationConstraints | None = None
    store_final_step_properties_only: bool = False


class MlipRelaxMakerSettings(MakerSettings):  # noqa: D101
    backend: Literal[RelaxBackend.mlip] = RelaxBackend.mlip
    simulation_settings: MlipRelaxSimulationSettings


class MlipTaskDocument(BaseTaskDocument):
    """Base task document for MLIP calculations."""


class MlipSinglePointTaskDocument(MSONableModelMixin, MlipTaskDocument, MSONable):  # noqa: D101
    properties: SinglePointProperties | None = Field(
        description="Computed single-point properties from an MLIP task"
    )


class MlipRelaxTaskDocument(MSONableModelMixin, MlipTaskDocument, MSONable):
    """MSONable task document for an MLIP relaxation."""
