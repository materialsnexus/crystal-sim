"""Tools for random generation of chemical formulas."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cached_property
from typing import TypeAlias, assert_never

import numpy as np
from ase.formula import Formula
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    model_validator,
    validate_call,
)

from crystal.schemas.arrays import FloatArray, IntArray
from crystal.schemas.elements import ElementStr
from crystal.schemas.formula_factories import (
    ChemicalSpaceFormulaFactorySettings,
    FixedSetFormulaFactorySettings,
    FormulaFactorySettings,
)
from crystal.schemas.randomness import RandomSeed, random_seed_field

CHEMICAL_SPACE: TypeAlias = set[ElementStr]


class FormulaSpaceDistribution(BaseModel):
    """Functions controlling how we sample over the space of possible formulas."""

    chemical_space_size: PositiveInt
    max_number_of_atoms: PositiveInt

    def count_possible_formulae_for_num_elements(
        self, num_elements: PositiveInt
    ) -> PositiveInt:
        """Number of formulae that exist with the given number of elements."""
        return math.comb(self.chemical_space_size, num_elements) * math.comb(
            self.max_number_of_atoms, num_elements
        )

    @cached_property
    def num_elements_and_probabilities(self) -> tuple[IntArray, FloatArray]:
        """Supported number of elements and their probabilities in formula space."""
        num_elements = np.arange(
            1, min(self.chemical_space_size, self.max_number_of_atoms) + 1
        )
        weights = [
            self.count_possible_formulae_for_num_elements(int(num_elements))
            for num_elements in num_elements
        ]
        probabilities = weights / np.sum(weights)
        return num_elements, probabilities

    @staticmethod
    def count_possible_formulae_for_num_atoms_and_num_elements(
        num_atoms: PositiveInt, num_elements: PositiveInt
    ) -> int:
        """Number of formulae that exist with the given number of atoms and number of elements."""
        return math.comb(num_atoms - 1, num_elements - 1)

    def num_atoms_and_probabilities_for_given_num_elements(
        self, num_elements: PositiveInt
    ) -> tuple[IntArray, FloatArray]:
        """For the given number of elements, gives the supported number of atoms and their probabilities in formula space."""
        num_atoms = np.arange(num_elements, self.max_number_of_atoms + 1)
        weights = [
            self.count_possible_formulae_for_num_atoms_and_num_elements(
                num_atoms=num_atoms, num_elements=num_elements
            )
            for num_atoms in num_atoms
        ]
        probabilities = weights / np.sum(weights)
        return num_atoms, probabilities


class BaseFormulaFactory(BaseModel, ABC):
    """Base interface for random chemical formula generators."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    random_seed: RandomSeed = random_seed_field()

    @cached_property
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=self.random_seed)

    @cached_property
    @abstractmethod
    def num_possible_stoichiometries(self) -> int:
        """Return the number of stoichiometries in this search space."""
        raise NotImplementedError

    @abstractmethod
    def draw_next(self) -> Formula:
        """Draw the next chemical formula."""
        raise NotImplementedError

    @validate_call
    def draw_n_samples(self, n_samples: PositiveInt) -> Iterable[Formula]:
        """Draw n chemical formulae."""
        for _ in range(n_samples):
            yield self.draw_next()


class ChemicalSpaceFormulaFactory(BaseFormulaFactory):
    """Random chemical formula generator over a bounded chemical space."""

    chemical_space: CHEMICAL_SPACE = Field(min_length=1)
    max_number_of_atoms: PositiveInt

    @cached_property
    def _elements_as_tuple(self) -> tuple[ElementStr, ...]:
        return tuple(sorted(self.chemical_space))

    @cached_property
    def _distribution(self) -> FormulaSpaceDistribution:
        return FormulaSpaceDistribution(
            chemical_space_size=len(self.chemical_space),
            max_number_of_atoms=self.max_number_of_atoms,
        )

    @cached_property
    def num_possible_stoichiometries(self) -> int:
        """Return the number of order-invariant stoichiometries in this search space."""
        max_num_atoms = self.max_number_of_atoms
        max_num_parts = min(len(self.chemical_space), max_num_atoms)
        partitions = [
            [0 for _ in range(max_num_parts + 1)] for _ in range(max_num_atoms + 1)
        ]
        partitions[0][0] = 1

        for num_atoms in range(1, max_num_atoms + 1):
            for num_parts in range(1, min(num_atoms, max_num_parts) + 1):
                partitions[num_atoms][num_parts] = partitions[num_atoms - 1][
                    num_parts - 1
                ]
                if num_atoms >= num_parts:
                    partitions[num_atoms][num_parts] += partitions[
                        num_atoms - num_parts
                    ][num_parts]

        return sum(
            partitions[num_atoms][num_parts]
            for num_atoms in range(1, max_num_atoms + 1)
            for num_parts in range(1, min(num_atoms, max_num_parts) + 1)
        )

    def _sample_n_elements(self) -> PositiveInt:
        values, probs = self._distribution.num_elements_and_probabilities
        return int(self._rng.choice(values, p=probs))

    def _sample_elements(self, n_elements: PositiveInt) -> list[ElementStr]:
        return sorted(
            self._rng.choice(self._elements_as_tuple, size=n_elements, replace=False)
        )

    def _sample_n_total_atoms(self, n_elements: PositiveInt) -> PositiveInt:
        values, probs = (
            self._distribution.num_atoms_and_probabilities_for_given_num_elements(
                n_elements
            )
        )
        return int(self._rng.choice(values, p=probs))

    def draw_next(self) -> Formula:
        """Draw the next random chemical formula."""
        n_elements = self._sample_n_elements()
        sampled_elements = self._sample_elements(n_elements)
        n_total_atoms = self._sample_n_total_atoms(n_elements)

        if n_elements == 1:
            atom_counts = [n_total_atoms]
        else:
            split_points = np.sort(
                self._rng.choice(
                    np.arange(1, n_total_atoms), size=n_elements - 1, replace=False
                )
            )
            atom_counts = np.diff(
                split_points, prepend=0, append=n_total_atoms
            ).tolist()

        formula = "".join(f"{e}{c}" for e, c in zip(sampled_elements, atom_counts))
        return Formula(formula)


class FixedSetFormulaFactory(BaseFormulaFactory):
    """Random chemical formula generator over an explicit fixed formula set."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    formula_space: list[Formula] = Field(min_length=1)

    @staticmethod
    def _formula_tuple(formula: Formula) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(formula.count().items()))

    @model_validator(mode="after")
    def _validate_formula_space_has_no_duplicates(self) -> FixedSetFormulaFactory:
        formula_tuples = [
            self._formula_tuple(formula) for formula in self.formula_space
        ]
        if len(formula_tuples) != len(set(formula_tuples)):
            msg = "formula_space must not contain duplicate formulae."
            raise ValueError(msg)
        return self

    @cached_property
    def _sorted_formula_space(self) -> tuple[Formula, ...]:
        return tuple(sorted(self.formula_space, key=self._formula_tuple))

    @cached_property
    def num_possible_stoichiometries(self) -> int:
        """Return the number of order-invariant stoichiometries in this search space."""
        return len(
            {
                tuple(sorted(count for _, count in self._formula_tuple(formula)))
                for formula in self._sorted_formula_space
            }
        )

    def draw_next(self) -> Formula:
        """Draw the next random chemical formula from the fixed formula set."""
        formula_index = int(self._rng.integers(len(self._sorted_formula_space)))
        return self._sorted_formula_space[formula_index]


def build_formula_factory(
    formula_factory_settings: FormulaFactorySettings,
) -> BaseFormulaFactory:
    """Build a runtime formula factory from validated formula factory settings."""
    if isinstance(formula_factory_settings, ChemicalSpaceFormulaFactorySettings):
        return ChemicalSpaceFormulaFactory(
            chemical_space=set(formula_factory_settings.chemical_space),
            max_number_of_atoms=formula_factory_settings.max_number_of_atoms,
            random_seed=formula_factory_settings.random_seed,
        )

    if isinstance(formula_factory_settings, FixedSetFormulaFactorySettings):
        return FixedSetFormulaFactory(
            formula_space=[
                Formula(formula_string)
                for formula_string in formula_factory_settings.formula_space
            ],
            random_seed=formula_factory_settings.random_seed,
        )

    assert_never(formula_factory_settings)
