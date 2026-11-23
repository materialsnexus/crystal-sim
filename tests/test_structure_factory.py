from typing import cast

import pytest
from ase.formula import Formula
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.schemas.symmetry import SpaceGroupNumber
from crystal.structure.generators.pyxtal.formula_factories import (
    ChemicalSpaceFormulaFactory,
)
from crystal.structure.generators.pyxtal.structure_factory import (
    PyxtalStructureGenerator,
    StoichiometryCachedSpaceGroupSampler,
    StructureFactory,
)

TEST_VALID_SPACE_GROUP_NUMBER = 225
TEST_INVALID_SPACE_GROUP_NUMBER = 220

TEST_FORMULA_FACTORY = ChemicalSpaceFormulaFactory(
    random_seed=42, chemical_space={"Na", "Cl"}, max_number_of_atoms=8
)


def test_space_group_sampler_stoichiometry_key_is_order_invariant() -> None:
    formula_a = Formula("Na2Cl1")
    formula_b = Formula("Cl1Na2")
    assert (
        StoichiometryCachedSpaceGroupSampler._stoichiometry_key(formula_a)
        == StoichiometryCachedSpaceGroupSampler._stoichiometry_key(formula_b)
        == (1, 2)
    )


def test_space_group_sampler_draw_and_check_compatibility_prunes_incompatible_space_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyGroup:
        def __init__(self, space_group: int) -> None:
            pass

        def check_compatible(self, stoichiometry: tuple[int, ...]) -> tuple[bool]:  # noqa: ARG002
            return (False,)

    monkeypatch.setattr(
        "crystal.structure.generators.pyxtal.structure_factory.pyxtal.Group", DummyGroup
    )
    sampler = StoichiometryCachedSpaceGroupSampler(space_groups={5}, random_seed=0)

    first_space_group, first_is_compatible = sampler.draw_and_check_compatibility(
        Formula("Na2")
    )
    assert first_space_group == 5
    assert first_is_compatible is False

    second_space_group, second_is_compatible = sampler.draw_and_check_compatibility(
        Formula("Na2")
    )
    assert second_space_group is None
    assert second_is_compatible is False


def test_space_group_sampler_draw_and_check_compatibility_keeps_compatible_space_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyGroup:
        def __init__(self, space_group: int) -> None:
            pass

        def check_compatible(self, stoichiometry: tuple[int, ...]) -> tuple[bool]:  # noqa: ARG002
            return (True,)

    monkeypatch.setattr(
        "crystal.structure.generators.pyxtal.structure_factory.pyxtal.Group", DummyGroup
    )
    sampler = StoichiometryCachedSpaceGroupSampler(space_groups={7}, random_seed=0)

    first_space_group, first_is_compatible = sampler.draw_and_check_compatibility(
        Formula("Na2")
    )
    assert first_space_group == 7
    assert first_is_compatible is True

    second_space_group, second_is_compatible = sampler.draw_and_check_compatibility(
        Formula("Na2")
    )
    assert second_space_group == 7
    assert second_is_compatible is True


def test_structure_factory_generate_structure_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_structure = cast("PMG_Structure", object())

    def _always_compatible(
        self: StoichiometryCachedSpaceGroupSampler, formula: Formula
    ) -> tuple[SpaceGroupNumber | None, bool]:
        _ = self, formula
        return TEST_VALID_SPACE_GROUP_NUMBER, True

    def _return_dummy_structure(
        self: PyxtalStructureGenerator,
        chemical_formula: Formula,
        space_group: SpaceGroupNumber,
    ) -> PMG_Structure:
        _ = self, chemical_formula, space_group
        return dummy_structure

    monkeypatch.setattr(
        StoichiometryCachedSpaceGroupSampler,
        "draw_and_check_compatibility",
        _always_compatible,
        raising=True,
    )
    monkeypatch.setattr(
        PyxtalStructureGenerator, "generate", _return_dummy_structure, raising=True
    )

    factory = StructureFactory(
        formula_factory=TEST_FORMULA_FACTORY,
        structure_generator=PyxtalStructureGenerator(random_seed=1),
        space_group_sampler=StoichiometryCachedSpaceGroupSampler(
            space_groups={1, 2, 3}, random_seed=2
        ),
    )
    assert factory.generate_structure() is dummy_structure


def test_structure_factory_generate_structure_raises_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _always_incompatible(
        self: StoichiometryCachedSpaceGroupSampler, formula: Formula
    ) -> tuple[SpaceGroupNumber | None, bool]:
        _ = self, formula
        return TEST_INVALID_SPACE_GROUP_NUMBER, False

    monkeypatch.setattr(
        StoichiometryCachedSpaceGroupSampler,
        "draw_and_check_compatibility",
        _always_incompatible,
        raising=True,
    )

    factory = StructureFactory(
        formula_factory=TEST_FORMULA_FACTORY,
        structure_generator=PyxtalStructureGenerator(random_seed=1),
        space_group_sampler=StoichiometryCachedSpaceGroupSampler(
            space_groups={TEST_INVALID_SPACE_GROUP_NUMBER}, random_seed=2
        ),
        max_generation_attempts_per_structure=3,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Failed to generate a compatible structure after 3 attempts.",
    ):
        _ = factory.generate_structure()


def test_structure_factory_raises_when_search_space_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyGroup:
        def __init__(self, space_group: int) -> None:
            pass

        def check_compatible(self, stoichiometry: tuple[int, ...]) -> tuple[bool]:  # noqa: ARG002
            return (False,)

    monkeypatch.setattr(
        "crystal.structure.generators.pyxtal.structure_factory.pyxtal.Group", DummyGroup
    )

    factory = StructureFactory(
        formula_factory=ChemicalSpaceFormulaFactory(
            random_seed=42, chemical_space={"Na"}, max_number_of_atoms=1
        ),
        structure_generator=PyxtalStructureGenerator(random_seed=1),
        space_group_sampler=StoichiometryCachedSpaceGroupSampler(
            space_groups={220}, random_seed=2
        ),
        max_generation_attempts_per_structure=5,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"None of the provided space-groups are compatible with the formula factory search space."
        ),
    ):
        _ = factory.generate_structure()


def test_structure_factory_logs_warning_when_no_candidate_space_groups(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _no_candidates_left(
        self: StoichiometryCachedSpaceGroupSampler, formula: Formula
    ) -> tuple[SpaceGroupNumber | None, bool]:
        _ = self, formula
        return None, False

    monkeypatch.setattr(
        StoichiometryCachedSpaceGroupSampler,
        "draw_and_check_compatibility",
        _no_candidates_left,
        raising=True,
    )
    caplog.set_level("WARNING")

    factory = StructureFactory(
        formula_factory=TEST_FORMULA_FACTORY,
        structure_generator=PyxtalStructureGenerator(random_seed=1),
        space_group_sampler=StoichiometryCachedSpaceGroupSampler(
            space_groups={1}, random_seed=2
        ),
        max_generation_attempts_per_structure=2,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Failed to generate a compatible structure after 2 attempts.",
    ):
        _ = factory.generate_structure()

    assert "No compatible space groups for stoichiometry" in caplog.text


def test_structure_factory_generate_structures_returns_expected_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_structure = cast("PMG_Structure", object())

    def _always_compatible(
        self: StoichiometryCachedSpaceGroupSampler, formula: Formula
    ) -> tuple[SpaceGroupNumber | None, bool]:
        _ = self, formula
        return TEST_VALID_SPACE_GROUP_NUMBER, True

    def _return_dummy_structure(
        self: PyxtalStructureGenerator,
        chemical_formula: Formula,
        space_group: SpaceGroupNumber,
    ) -> PMG_Structure:
        _ = self, chemical_formula, space_group
        return dummy_structure

    monkeypatch.setattr(
        StoichiometryCachedSpaceGroupSampler,
        "draw_and_check_compatibility",
        _always_compatible,
        raising=True,
    )
    monkeypatch.setattr(
        PyxtalStructureGenerator, "generate", _return_dummy_structure, raising=True
    )

    factory = StructureFactory(
        formula_factory=TEST_FORMULA_FACTORY,
        structure_generator=PyxtalStructureGenerator(random_seed=1),
        space_group_sampler=StoichiometryCachedSpaceGroupSampler(
            space_groups={1}, random_seed=2
        ),
    )

    structures = factory.generate_structures(n_structures=2)
    assert structures == [dummy_structure, dummy_structure]


def test_structure_factory_generate_structures_integration() -> None:
    """Integration test with real sampler and generator (no mocks)."""
    formula_factory = ChemicalSpaceFormulaFactory(
        random_seed=42, chemical_space={"Na", "Cl"}, max_number_of_atoms=8
    )
    factory = StructureFactory(
        formula_factory=formula_factory,
        structure_generator=PyxtalStructureGenerator(random_seed=42),
        space_group_sampler=StoichiometryCachedSpaceGroupSampler(
            space_groups={TEST_VALID_SPACE_GROUP_NUMBER}, random_seed=42
        ),
        max_generation_attempts_per_structure=200,
    )

    structures = factory.generate_structures(n_structures=2)
    assert len(structures) == 2
    for structure in structures:
        assert isinstance(structure, PMG_Structure)
        assert {site.specie.symbol for site in structure.sites} <= {"Na", "Cl"}
        assert 1 <= int(structure.composition.num_atoms) <= 8
