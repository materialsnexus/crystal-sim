"""PyXtal-backed structure generation utilities."""

from __future__ import annotations

from functools import cached_property
from logging import getLogger

import numpy as np
import pyxtal
from ase.formula import Formula
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, PrivateAttr
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.structure_generators import (
    DEFAULT_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE,
)
from crystal.schemas.randomness import RandomSeed, random_seed_field
from crystal.schemas.symmetry import SpaceGroupNumber, all_space_groups_set
from crystal.structure.generators.pyxtal.formula_factories import BaseFormulaFactory

logger = getLogger(__name__)


class PyxtalStructureGenerator(BaseModel):
    """Generate crystal structures from formula and space group constraints."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    random_seed: RandomSeed = random_seed_field()

    @staticmethod
    def _validate_formula_space_group_compatibility(
        chemical_formula: Formula, space_group: SpaceGroupNumber
    ) -> None:
        is_compatible = pyxtal.Group(space_group).check_compatible(
            chemical_formula.count().values()
        )[0]
        if not is_compatible:
            msg = f"chemical formula {chemical_formula} is incompatible with the requested space group {space_group}"
            raise ValueError(msg)

    def generate(
        self, chemical_formula: Formula, space_group: SpaceGroupNumber
    ) -> PMG_Structure:
        """Generate a structure compatible with the requested formula and space group."""
        self._validate_formula_space_group_compatibility(
            chemical_formula=chemical_formula, space_group=space_group
        )
        chemical_formula_dict = chemical_formula.count()
        px_structure = pyxtal.pyxtal()
        px_structure.from_random(
            species=tuple(str(e) for e in chemical_formula_dict),
            numIons=tuple(chemical_formula_dict.values()),
            group=space_group,
            random_state=self.random_seed,
        )
        pmg_structure = px_structure.to_pymatgen()
        assert isinstance(pmg_structure, PMG_Structure)
        return pmg_structure


class StoichiometryCachedSpaceGroupSampler(BaseModel):
    """Sample space groups from a stoichiometry-indexed compatibility cache."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    space_groups: set[SpaceGroupNumber] = Field(
        default_factory=all_space_groups_set, min_length=1
    )
    random_seed: RandomSeed = random_seed_field()
    _compatibility_cache: dict[tuple[int, ...], set[SpaceGroupNumber]] = PrivateAttr(
        default_factory=dict
    )

    @cached_property
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=self.random_seed)

    @staticmethod
    def _stoichiometry_key(formula: Formula) -> tuple[int, ...]:
        return tuple(sorted(formula.count().values()))

    def _get_candidate_space_groups(
        self, stoichiometry: tuple[int, ...]
    ) -> set[SpaceGroupNumber]:
        if stoichiometry not in self._compatibility_cache:
            self._compatibility_cache[stoichiometry] = self.space_groups.copy()
        return self._compatibility_cache[stoichiometry]

    @property
    def num_cached_stoichiometries(self) -> int:
        """Returns the total number of stoichiometries (keys)."""
        return len(self._compatibility_cache)

    @property
    def are_cached_stoichiometries_exhausted(self) -> bool:
        """Returns true if candidate space groups (values) are empty for all stoichiometries (keys)."""
        return all(
            len(candidate_space_groups) == 0
            for candidate_space_groups in self._compatibility_cache.values()
        )

    def draw_and_check_compatibility(
        self, formula: Formula
    ) -> tuple[SpaceGroupNumber | None, bool]:
        """Draw a candidate space group and prune cache entries that fail compatibility check."""
        stoichiometry = self._stoichiometry_key(formula)
        candidate_space_groups = self._get_candidate_space_groups(stoichiometry)
        if not candidate_space_groups:
            return None, False

        sampled_space_group = self._rng.choice(tuple(sorted(candidate_space_groups)))
        is_compatible = pyxtal.Group(sampled_space_group).check_compatible(
            stoichiometry
        )[0]

        if not is_compatible:
            self._compatibility_cache[stoichiometry].remove(sampled_space_group)

        return sampled_space_group, is_compatible


class StructureFactory(BaseModel):
    """Factory for sampling one or more structures from a search space."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    formula_factory: BaseFormulaFactory
    structure_generator: PyxtalStructureGenerator
    space_group_sampler: StoichiometryCachedSpaceGroupSampler
    max_generation_attempts_per_structure: PositiveInt = (
        DEFAULT_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE
    )

    def _raise_if_search_space_empty(self) -> None:
        """Raises ValueError if every possible stoichiometry has no compatible space groups."""
        if (
            self.space_group_sampler.num_cached_stoichiometries
            != self.formula_factory.num_possible_stoichiometries
        ):
            return
        if not self.space_group_sampler.are_cached_stoichiometries_exhausted:
            return

        msg = "None of the provided space-groups are compatible with the formula factory search space."
        raise ValueError(msg)

    def _generate_structure_attempt(self, attempt: int) -> PMG_Structure | None:
        chemical_formula = self.formula_factory.draw_next()
        space_group, is_compatible = (
            self.space_group_sampler.draw_and_check_compatibility(chemical_formula)
        )

        if is_compatible:
            assert space_group is not None
            return self.structure_generator.generate(
                chemical_formula=chemical_formula, space_group=space_group
            )

        if space_group is None:
            logger.warning(
                "No compatible space groups for stoichiometry %s on attempt %s/%s. "
                "Formula=%s. Retrying with a new formula.",
                tuple(sorted(chemical_formula.count().values())),
                attempt,
                self.max_generation_attempts_per_structure,
                chemical_formula,
            )
            return None

        logger.warning(
            "Incompatible formula/space group pair generated on attempt %s/%s. "
            "Formula=%s, Space Group=%s. Retrying with a new pair.",
            attempt,
            self.max_generation_attempts_per_structure,
            chemical_formula,
            space_group,
        )
        return None

    def generate_structure(self) -> PMG_Structure:
        """Generate a single structure."""
        for attempt in range(1, self.max_generation_attempts_per_structure + 1):
            generated_structure = self._generate_structure_attempt(attempt=attempt)
            if generated_structure is not None:
                return generated_structure
            self._raise_if_search_space_empty()

        msg = f"Failed to generate a compatible structure after {self.max_generation_attempts_per_structure} attempts."
        raise RuntimeError(msg)

    def generate_structures(self, n_structures: PositiveInt) -> list[PMG_Structure]:
        """Generate `n_structures` independent structures; note duplication is possible."""
        return [self.generate_structure() for _ in range(n_structures)]
