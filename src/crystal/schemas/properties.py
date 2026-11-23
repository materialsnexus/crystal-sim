# noqa: D100
from collections.abc import Sequence
from enum import IntEnum, StrEnum, auto
from math import isfinite
from typing import Generic, Literal, Self, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PositiveFloat, model_validator
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.schemas.base_properties import Properties
from crystal.schemas.filtering_statistics import CohesiveEnergyFilteringStatistics
from crystal.schemas.serialization import MSONableBaseModel

FORCE_VECTOR_LENGTH = 3
VOIGT_STRESS_VECTOR_LENGTH = 6
ForcesLike: TypeAlias = Sequence[Sequence[float]] | NDArray[np.float64]
VoigtStressLike: TypeAlias = Sequence[float] | NDArray[np.float64]
VoigtStressesLike: TypeAlias = Sequence[Sequence[float]] | NDArray[np.float64]
StructureList: TypeAlias = list[PMG_Structure]


class ForceComponent(IntEnum):
    """Cartesian force components in ASE force-vector order."""

    x = 0
    y = 1
    z = 2


class StressComponent(IntEnum):
    """Voigt stress components in ASE order."""

    xx = 0
    yy = 1
    zz = 2
    yz = 3
    xz = 4
    xy = 5


class VectorMode(StrEnum):
    """Vector-valued property plotting modes."""

    norm = auto()
    component = auto()


class StepProperties(Properties):
    """Property values are in ASE standard units:
    energy = eV
    forces = eV/A
    stress = eV/A3
    charges = e
    """  # noqa: D415

    type: Literal["StepProperties", "SinglePointProperties"] = "StepProperties"
    step: int
    structure: PMG_Structure
    forces: list[list[float]]
    stress: list[float]
    energy: float
    charges: list[float] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_force_and_stress_shapes(self) -> Self:
        expected_force_vector_count = len(self.structure)
        if len(self.forces) != expected_force_vector_count or any(
            len(force) != FORCE_VECTOR_LENGTH for force in self.forces
        ):
            msg = (
                "forces must have shape "
                f"({expected_force_vector_count}, {FORCE_VECTOR_LENGTH}) "
                "for the associated structure."
            )
            raise ValueError(msg)

        if len(self.stress) != VOIGT_STRESS_VECTOR_LENGTH:
            msg = "stress must be a Voigt-6 vector."
            raise ValueError(msg)

        return self


class SinglePointProperties(StepProperties):  # noqa: D101
    type: Literal["SinglePointProperties"] = "SinglePointProperties"

    @model_validator(mode="after")
    def _validate_finite_labels_and_charges(self) -> Self:
        if not isfinite(self.energy):
            msg = "energy must be a finite numeric scalar."
            raise ValueError(msg)
        if not all(
            isfinite(force_component)
            for force in self.forces
            for force_component in force
        ):
            msg = "forces must contain only finite numeric values."
            raise ValueError(msg)
        if not all(isfinite(stress_component) for stress_component in self.stress):
            msg = "stress must contain only finite numeric values."
            raise ValueError(msg)
        if self.charges is None:
            return self
        expected_charge_count = len(self.structure)
        if len(self.charges) != expected_charge_count:
            msg = (
                "charges must have shape "
                f"({expected_charge_count},) for the associated structure."
            )
            raise ValueError(msg)
        if not all(isfinite(charge) for charge in self.charges):
            msg = "charges must contain only finite numeric values."
            raise ValueError(msg)
        return self


class RelaxProperties(Properties):  # noqa: D101
    type: Literal["RelaxProperties"] = "RelaxProperties"
    relax_steps: list[StepProperties]

    @property
    def relaxed_structure(self) -> PMG_Structure:  # noqa: D102
        return self.relax_steps[-1].structure

    @property
    def final_energy(self) -> float:  # noqa: D102
        return self.relax_steps[-1].energy


class StructureGenerationProperties(Properties):  # noqa: D101
    type: Literal["StructureGenerationProperties"] = "StructureGenerationProperties"
    structures: StructureList


class CohesiveEnergyFilteringProperties(Properties):
    """Surviving structures and cohesive-energy filtering statistics."""

    type: Literal["CohesiveEnergyFilteringProperties"] = (
        "CohesiveEnergyFilteringProperties"
    )
    structures: StructureList
    statistics: CohesiveEnergyFilteringStatistics

    @model_validator(mode="after")
    def _validate_statistics(self) -> Self:
        if self.statistics.passed != len(self.structures):
            msg = "statistics.passed must match the number of surviving structures."
            raise ValueError(msg)
        return self


class StructureWithFilteringMetricsBase(MSONableBaseModel):  # noqa: D101
    structure: PMG_Structure


class StructureWithSinglePointFilteringMetrics(StructureWithFilteringMetricsBase):  # noqa: D101
    cohesive_energy_per_atom: float
    maximum_force_norm: PositiveFloat
    stress_norm: PositiveFloat


StructureFilteringMetricsType = TypeVar(
    "StructureFilteringMetricsType", bound=StructureWithFilteringMetricsBase
)


class StructureFilteringPropertiesBase(
    Properties, Generic[StructureFilteringMetricsType]
):
    """Base properties for structure-list filtering results."""

    all_structures_with_metrics: list[StructureFilteringMetricsType]
    filtering_mask: list[bool]

    @property
    def structures(self) -> list[PMG_Structure]:
        """Return only the structures that passed filtering."""
        return [
            datum.structure
            for datum, include_structure in zip(
                self.all_structures_with_metrics, self.filtering_mask, strict=True
            )
            if include_structure
        ]

    @model_validator(mode="after")
    def _validate_length_of_structure_data_and_mask(self) -> Self:
        expected_length = len(self.all_structures_with_metrics)
        if len(self.filtering_mask) != expected_length:
            msg = "filtering_mask must have the same length as structure_data."
            raise ValueError(msg)
        return self


class SinglePointStructureFilteringProperties(
    StructureFilteringPropertiesBase[StructureWithSinglePointFilteringMetrics]
):
    """Filtering properties produced by single-point structure checks."""

    type: Literal["SinglePointStructureFilteringProperties"] = (
        "SinglePointStructureFilteringProperties"
    )
