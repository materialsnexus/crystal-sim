"""Schemas for generating initial crystal structures."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from crystal.constants.structure_generators import (
    DEFAULT_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE,
)
from crystal.schemas.base_properties import Properties
from crystal.schemas.formula_factories import (
    ChemicalSpaceFormulaFactorySettings,
    FormulaFactorySettings,
)
from crystal.schemas.properties import StructureList
from crystal.schemas.randomness import RandomSeed, random_seed_field
from crystal.schemas.settings import SimulationSettings
from crystal.schemas.symmetry import SpaceGroupNumber, all_space_groups_list


class InitialStructureGenerationSettings(SimulationSettings):
    """Configuration for generating initial crystal structures."""

    model_config = ConfigDict(extra="forbid")

    random_seed: RandomSeed = random_seed_field()
    formula_factory_settings: FormulaFactorySettings = Field(
        default_factory=ChemicalSpaceFormulaFactorySettings
    )
    space_groups: list[SpaceGroupNumber] = Field(
        default_factory=all_space_groups_list, min_length=1
    )
    max_generation_attempts_per_structure: PositiveInt = (
        DEFAULT_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE
    )


class StructureIndexRange(BaseModel):
    """Consecutive, inclusive range of structure indices."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    first_structure_index: NonNegativeInt = 0
    last_structure_index: NonNegativeInt

    @model_validator(mode="after")
    def _validate_indices_are_ordered(self) -> Self:
        if self.last_structure_index < self.first_structure_index:
            msg = "last_structure_index must be at least first_structure_index."
            raise ValueError(msg)
        return self

    @property
    def number_of_structures(self) -> int:
        """Return the number of indices in this range."""
        return self.last_structure_index - self.first_structure_index + 1

    @property
    def structure_indices(self) -> range:
        """Return the structure indices in this range."""
        return range(self.first_structure_index, self.last_structure_index + 1)


class InitialStructureGenerationProperties(Properties):
    """Generated structures and whether generation ended early."""

    type: Literal["InitialStructureGenerationProperties"] = (
        "InitialStructureGenerationProperties"
    )
    generation_range: StructureIndexRange
    structures: StructureList
    terminated_early: bool = Field(
        default=False,
        description=(
            "Whether generation stopped early after a failure. Partial results "
            "may still be useful for long-running generation jobs."
        ),
    )
