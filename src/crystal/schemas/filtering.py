"""Pydantic models for structure filtering."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import ConfigDict, Field, FiniteFloat, model_validator

from crystal.schemas.base import BaseTaskDocument
from crystal.schemas.isolated_atom_energies import (
    IsolatedAtomEnergies,
    IsolatedAtomEnergiesRef,
)
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.settings import SimulationSettings

PositiveFiniteFloat = Annotated[FiniteFloat, Field(gt=0)]


class GeometricStructureFilteringSettings(SimulationSettings):  # noqa: D101
    minimum_interatomic_distance: PositiveFiniteFloat = 0.5
    maximum_box_aspect_ratio: PositiveFiniteFloat = 6.0


class CohesiveEnergyFilteringSettings(SimulationSettings):
    """Settings for filtering structures by cohesive energy per atom."""

    model_config = ConfigDict(extra="forbid")

    evaluator_model: MlipModelRef | None = Field(
        default=None,
        description=(
            "MLIP model whose element reference energies are used when no explicit "
            "isolated_atom_energies override is supplied."
        ),
    )
    isolated_atom_energies: IsolatedAtomEnergies | None = Field(
        default=None,
        description=(
            "Explicit per-element MLIP reference energies. When provided, these "
            "values override the evaluator model references."
        ),
    )
    maximum_cohesive_energy_per_atom: FiniteFloat | None = Field(
        default=None,
        description="Maximum cohesive energy per atom; None disables this filter.",
    )

    @model_validator(mode="after")
    def _validate_reference_source(self) -> Self:
        if (
            self.maximum_cohesive_energy_per_atom is not None
            and self.isolated_atom_energies is None
            and self.evaluator_model is None
        ):
            error_msg = (
                "Cohesive-energy filtering requires either explicit isolated atom "
                "energies or an evaluator MLIP model."
            )
            raise ValueError(error_msg)
        return self


class SinglePointStructureFilteringSettings(SimulationSettings):  # noqa: D101
    isolated_atom_energies: IsolatedAtomEnergiesRef = Field(
        default_factory=IsolatedAtomEnergiesRef
    )
    maximum_cohesive_energy_per_atom: FiniteFloat
    maximum_force_norm: PositiveFiniteFloat
    maximum_stress_norm: PositiveFiniteFloat


class StructureFilteringTaskDocument(BaseTaskDocument):  # noqa: D101
    pass


class CohesiveEnergyFilteringTaskDocument(StructureFilteringTaskDocument):
    """Task document for cohesive-energy filtering results."""

    warnings: str = Field(
        default="", description="Non-fatal warnings from filtering inputs."
    )


class SinglePointStructureFilteringTaskDocument(StructureFilteringTaskDocument):  # noqa: D101
    pass
