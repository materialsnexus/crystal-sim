"""Initial crystal structure generation."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, NonNegativeInt, validate_call
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationSettings,
)
from crystal.structure.generators.pyxtal.formula_factories import (
    BaseFormulaFactory,
    build_formula_factory,
)
from crystal.structure.generators.pyxtal.structure_factory import (
    PyxtalStructureGenerator,
    StoichiometryCachedSpaceGroupSampler,
)
from crystal.structure.generators.randomness import derive_random_seed

logger = logging.getLogger(__name__)


class InitialStructureGenerator(BaseModel):
    """Generate initial crystal structures from composition and symmetry settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    settings: InitialStructureGenerationSettings

    def _formula_factory(self, structure_index: int) -> BaseFormulaFactory:
        formula_settings = self.settings.formula_factory_settings.model_copy(
            update={
                "random_seed": derive_random_seed(
                    self.settings.random_seed, structure_index, "formula"
                )
            }
        )
        return build_formula_factory(formula_settings)

    def _space_group_sampler(
        self, structure_index: int
    ) -> StoichiometryCachedSpaceGroupSampler:
        return StoichiometryCachedSpaceGroupSampler(
            space_groups=set(self.settings.space_groups),
            random_seed=derive_random_seed(
                self.settings.random_seed, structure_index, "space-group"
            ),
        )

    @validate_call
    def generate(self, structure_index: NonNegativeInt) -> PMG_Structure:
        """Generate the structure at the requested index."""
        formula_factory = self._formula_factory(structure_index=structure_index)
        space_group_sampler = self._space_group_sampler(structure_index=structure_index)

        for attempt_index in range(self.settings.max_generation_attempts_per_structure):
            chemical_formula = formula_factory.draw_next()
            space_group, is_compatible = (
                space_group_sampler.draw_and_check_compatibility(chemical_formula)
            )
            if is_compatible:
                assert space_group is not None
                try:
                    return PyxtalStructureGenerator(
                        random_seed=derive_random_seed(
                            self.settings.random_seed,
                            structure_index,
                            "pyxtal",
                            attempt_index,
                        )
                    ).generate(
                        chemical_formula=chemical_formula, space_group=space_group
                    )
                except RuntimeError as exc:
                    logger.warning(
                        "PyXtal generation attempt failed: %s", exc, exc_info=True
                    )

        msg = (
            "Failed to generate a compatible structure after "
            f"{self.settings.max_generation_attempts_per_structure} attempts."
        )
        raise RuntimeError(msg)
