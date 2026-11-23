"""Pydantic models for formula factory settings."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, PositiveInt, model_validator

from crystal.schemas.elements import ElementStr
from crystal.schemas.formula import FormulaStr
from crystal.schemas.randomness import RandomSeed, random_seed_field
from crystal.schemas.settings import SimulationSettings

DEFAULT_CHEMICAL_SPACE: list[ElementStr] = ["Li", "Nb", "O", "Si"]
DEFAULT_MAX_NUMBER_OF_ATOMS: PositiveInt = 10


def _raise_if_duplicates_present(values: list[str], field_name: str) -> None:
    if len(values) != len(set(values)):
        error_msg = f"{field_name} must not contain duplicates."
        raise ValueError(error_msg)


class FormulaFactoryType(StrEnum):
    """Supported formula factory settings types."""

    chemical_space = auto()
    fixed_set = auto()


class ChemicalSpaceFormulaFactorySettings(SimulationSettings):
    """Settings used by the chemical space formula factory."""

    type: Literal[FormulaFactoryType.chemical_space] = FormulaFactoryType.chemical_space
    random_seed: RandomSeed = random_seed_field()
    chemical_space: list[ElementStr] = Field(
        default=DEFAULT_CHEMICAL_SPACE, min_length=1
    )
    max_number_of_atoms: PositiveInt = DEFAULT_MAX_NUMBER_OF_ATOMS

    @property
    def has_duplicate_elements(self) -> bool:  # noqa: D102
        return len(self.chemical_space) != len(set(self.chemical_space))

    @model_validator(mode="after")
    def _validate_chemical_space_has_no_duplicates(
        self,
    ) -> ChemicalSpaceFormulaFactorySettings:
        if self.has_duplicate_elements:
            msg = "`chemical_space` must not contain duplicates."
            raise ValueError(msg)
        return self


class FixedSetFormulaFactorySettings(SimulationSettings):
    """Settings used by the fixed set formula factory."""

    type: Literal[FormulaFactoryType.fixed_set] = FormulaFactoryType.fixed_set
    random_seed: RandomSeed = random_seed_field()
    formula_space: list[FormulaStr] = Field(min_length=1)

    @property
    def has_duplicate_elements(self) -> bool:  # noqa: D102
        return len(self.formula_space) != len(set(self.formula_space))

    @model_validator(mode="after")
    def _validate_formula_space_has_no_duplicates(
        self,
    ) -> FixedSetFormulaFactorySettings:
        if self.has_duplicate_elements:
            msg = "`formula_space` must not contain duplicates."
            raise ValueError(msg)
        return self


FormulaFactorySettings: TypeAlias = Annotated[
    ChemicalSpaceFormulaFactorySettings | FixedSetFormulaFactorySettings,
    Field(discriminator="type"),
]
