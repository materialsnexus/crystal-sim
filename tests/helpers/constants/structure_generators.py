"""Constants for structure-generation tests."""

from __future__ import annotations

from crystal.schemas.formula_factories import FixedSetFormulaFactorySettings
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationSettings,
)
from crystal.schemas.structure_generators import (
    RattleAtomPositionsSettings,
    StretchCellSettings,
)
from crystal.structure.generators.perturbation.rattle import RattleAtomDisplacementMode

TEST_CHEMICAL_SPACE = ("Na", "Cl")
TEST_MAX_NUMBER_OF_ATOMS = 8
TEST_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE = 25
TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS = RattleAtomPositionsSettings(
    random_seed=42,
    noise_scaling_factor=0.2,
    displacement_mode=RattleAtomDisplacementMode.normal,
)
TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS = StretchCellSettings(
    random_seed=42, max_normal_strain_fraction=0.2, max_shear_strain_fraction=0.1
)


def make_initial_structure_generation_settings(
    *, space_groups: list[int]
) -> InitialStructureGenerationSettings:
    """Build settings for deterministic initial-structure generation tests."""
    return InitialStructureGenerationSettings(
        random_seed=42,
        formula_factory_settings=FixedSetFormulaFactorySettings(
            random_seed=42, formula_space=["Na4Cl4"]
        ),
        space_groups=space_groups,
        max_generation_attempts_per_structure=5,
    )
