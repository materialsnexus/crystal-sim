"""Tests for pure indexed initial-structure generation."""

from copy import deepcopy

import pytest
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.structure.generators.pyxtal.initial_structure_generator import (
    InitialStructureGenerator,
)
from crystal.structure.generators.pyxtal.structure_factory import (
    PyxtalStructureGenerator,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.constants.structure_generators import (
    make_initial_structure_generation_settings,
)


def test_initial_structure_generator_is_repeatable_for_a_structure_index() -> None:
    settings = make_initial_structure_generation_settings(space_groups=[225])
    first_generator = InitialStructureGenerator(settings=settings)
    second_generator = InitialStructureGenerator(settings=settings)

    first_structure = first_generator.generate(structure_index=7)
    second_structure = second_generator.generate(structure_index=7)

    assert first_structure.as_dict() == second_structure.as_dict()


def test_initial_structure_generator_preserves_formula_and_space_group() -> None:
    generator = InitialStructureGenerator(
        settings=make_initial_structure_generation_settings(space_groups=[225])
    )

    structure = generator.generate(structure_index=0)

    assert structure.composition.as_dict() == {"Cl": 4, "Na": 4}
    assert structure.get_space_group_info(symprec=1e-3)[1] == 225


def test_initial_structure_generator_retries_pyxtal_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_attempts = 0

    def _generate_with_one_failure(_self: object, **_: object) -> PMG_Structure:
        nonlocal generated_attempts
        generated_attempts += 1
        if generated_attempts == 1:
            message = "PyXtal placement failed"
            raise RuntimeError(message)
        return deepcopy(TEST_SI_STRUCTURE)

    monkeypatch.setattr(
        PyxtalStructureGenerator, "generate", _generate_with_one_failure
    )
    generator = InitialStructureGenerator(
        settings=make_initial_structure_generation_settings(space_groups=[225])
    )

    structure = generator.generate(structure_index=0)

    assert structure == TEST_SI_STRUCTURE
    assert generated_attempts == 2


def test_initial_structure_generator_reports_bounded_failure_for_incompatible_formula() -> (
    None
):
    generator = InitialStructureGenerator(
        settings=make_initial_structure_generation_settings(space_groups=[220])
    )

    with pytest.raises(RuntimeError, match="after 5 attempts"):
        generator.generate(structure_index=0)
