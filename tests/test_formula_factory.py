import json
import os
import subprocess
from functools import lru_cache
from itertools import combinations

import pytest
from ase.formula import Formula
from pydantic import ValidationError

from crystal.schemas.formula import FormulaStr
from crystal.schemas.formula_factories import (
    ChemicalSpaceFormulaFactorySettings,
    FixedSetFormulaFactorySettings,
)
from crystal.structure.generators.pyxtal.formula_factories import (
    CHEMICAL_SPACE,
    BaseFormulaFactory,
    ChemicalSpaceFormulaFactory,
    FixedSetFormulaFactory,
    FormulaSpaceDistribution,
    build_formula_factory,
)


def _positive_integer_partitions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]

    return [
        (first, *remaining)
        for first in range(1, total - parts + 2)
        for remaining in _positive_integer_partitions(total - first, parts - 1)
    ]


@lru_cache
def _count_partitions_of_n_with_up_to_k_parts(n: int, k: int, min_part: int = 1) -> int:
    if n == 0:
        return 1
    if k == 0:
        return 0
    return sum(
        _count_partitions_of_n_with_up_to_k_parts(n=n - part, k=k - 1, min_part=part)
        for part in range(min_part, n + 1)
    )


def _build_fixed_set_formula_factory(
    *formula_strings: FormulaStr,
) -> FixedSetFormulaFactory:
    return FixedSetFormulaFactory(
        random_seed=42,
        formula_space=[Formula(formula_string) for formula_string in formula_strings],
    )


@pytest.mark.parametrize(
    ("formula_factory_settings", "expected_factory_type"),
    [
        (
            ChemicalSpaceFormulaFactorySettings(
                random_seed=42, chemical_space=["Na", "Cl"], max_number_of_atoms=8
            ),
            ChemicalSpaceFormulaFactory,
        ),
        (
            FixedSetFormulaFactorySettings(
                random_seed=42, formula_space=["NaCl", "Na2Cl2"]
            ),
            FixedSetFormulaFactory,
        ),
    ],
)
def test_build_formula_factory_selects_expected_factory_branch(
    formula_factory_settings: ChemicalSpaceFormulaFactorySettings
    | FixedSetFormulaFactorySettings,
    expected_factory_type: type[BaseFormulaFactory],
) -> None:
    formula_factory = build_formula_factory(formula_factory_settings)

    assert isinstance(formula_factory, expected_factory_type)


def test_build_formula_factory_converts_fixed_set_formula_strings_to_formula_objects() -> (
    None
):
    formula_factory = build_formula_factory(
        FixedSetFormulaFactorySettings(random_seed=42, formula_space=["NaCl", "Na2Cl2"])
    )

    assert isinstance(formula_factory, FixedSetFormulaFactory)
    assert formula_factory.formula_space == [Formula("Na1Cl1"), Formula("Na2Cl2")]


def test_distribution_num_elements_support_and_probs() -> None:
    distribution = FormulaSpaceDistribution(
        chemical_space_size=3, max_number_of_atoms=4
    )

    support, probs = distribution.num_elements_and_probabilities
    assert support.tolist() == [1, 2, 3]
    assert probs == pytest.approx([6 / 17, 9 / 17, 2 / 17])


def test_distribution_num_atoms_given_num_elements_support_and_probs() -> None:
    distribution = FormulaSpaceDistribution(
        chemical_space_size=4, max_number_of_atoms=5
    )

    support, probs = distribution.num_atoms_and_probabilities_for_given_num_elements(2)
    assert support.tolist() == [2, 3, 4, 5]
    assert probs == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_chemical_space_formula_factory_rejects_empty_chemical_space() -> None:
    with pytest.raises(ValidationError, match=r" at least 1 item"):
        ChemicalSpaceFormulaFactory(
            random_seed=0, chemical_space=CHEMICAL_SPACE(set()), max_number_of_atoms=5
        )


def test_chemical_space_formula_factory_draw_next_respects_total_atoms_bounds() -> None:
    factory = ChemicalSpaceFormulaFactory(
        random_seed=123, chemical_space={"H", "O", "C"}, max_number_of_atoms=7
    )

    for _ in range(200):
        f = factory.draw_next()
        assert 1 <= len(f) <= 7


def test_chemical_space_formula_factory_draw_next_never_uses_more_elements_than_atoms_or_space() -> (
    None
):
    cs = {"H", "O", "C", "N"}
    factory = ChemicalSpaceFormulaFactory(
        random_seed=7, chemical_space=cs, max_number_of_atoms=3
    )

    for _ in range(200):
        f = factory.draw_next()
        assert len(f) <= 3
        assert len(f) <= len(cs)


def test_chemical_space_formula_factory_draw_next_counts_are_positive_ints() -> None:
    factory = ChemicalSpaceFormulaFactory(
        random_seed=999, chemical_space={"H", "O"}, max_number_of_atoms=10
    )

    for _ in range(200):
        f = factory.draw_next()
        for el, n in f.count().items():
            assert isinstance(el, str)
            assert isinstance(n, int)
            assert n > 0


def test_chemical_space_formula_factory_draw_n_returns_n_items() -> None:
    factory = ChemicalSpaceFormulaFactory(
        random_seed=1, chemical_space={"H", "O"}, max_number_of_atoms=5
    )

    out = factory.draw_n_samples(15)
    out_list = list(out)
    assert len(out_list) == 15
    assert all(isinstance(x, Formula) for x in out_list)


@pytest.mark.parametrize("max_number_of_atoms", [0, -5])
def test_chemical_space_formula_factory_rejects_non_positive_max_number_of_atoms(
    max_number_of_atoms: int,
) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        ChemicalSpaceFormulaFactory(
            random_seed=0, chemical_space={"H"}, max_number_of_atoms=max_number_of_atoms
        )


@pytest.mark.parametrize(
    ("chemical_space_size", "max_number_of_atoms"),
    [(1, 5), (2, 5), (3, 5), (4, 7), (10, 5)],
)
def test_chemical_space_formula_factory_num_possible_stoichiometries(
    chemical_space_size: int, max_number_of_atoms: int
) -> None:
    element_symbols = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
    chemical_space = CHEMICAL_SPACE(set(element_symbols[:chemical_space_size]))
    factory = ChemicalSpaceFormulaFactory(
        random_seed=0,
        chemical_space=chemical_space,
        max_number_of_atoms=max_number_of_atoms,
    )
    max_parts = min(chemical_space_size, max_number_of_atoms)
    expected = sum(
        _count_partitions_of_n_with_up_to_k_parts(n=num_atoms, k=max_parts)
        for num_atoms in range(1, max_number_of_atoms + 1)
    )
    assert factory.num_possible_stoichiometries == expected


def _run_simple_generate() -> None:
    formula_factory = ChemicalSpaceFormulaFactory(
        random_seed=42,
        chemical_space=CHEMICAL_SPACE({"H", "C", "N", "O", "F"}),
        max_number_of_atoms=8,
    )
    formulas = [str(f) for f in formula_factory.draw_n_samples(10)]
    print(json.dumps(formulas, separators=(",", ":")))  # noqa: T201


def test_chemical_space_formula_factory_is_stable_across_hashseeds() -> None:
    env = os.environ.copy()
    outputs = []
    for hash_seed in ["1", "123", "42", "999"]:
        env["PYTHONHASHSEED"] = str(hash_seed)

        cmd = [
            "uv",
            "run",
            "python",
            "-c",
            "from tests.test_formula_factory import _run_simple_generate; _run_simple_generate()",
        ]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, check=True
        )
        outputs.append(result.stdout.strip())

    assert len(set(outputs)) == 1


def test_chemical_space_formula_factory_formula_space_probabilities_are_uniform() -> (
    None
):
    formula_factory = ChemicalSpaceFormulaFactory(
        chemical_space={"H", "O"}, max_number_of_atoms=3
    )
    expected_formulae = {"H1", "H2", "H3", "O1", "O2", "O3", "H1O1", "H1O2", "H2O1"}
    expected_formula_probabilities = {
        formula: 1 / len(expected_formulae) for formula in expected_formulae
    }

    distribution = formula_factory._distribution

    formula_probabilities = {}
    num_elements_support, num_elements_probabilities = (
        distribution.num_elements_and_probabilities
    )
    for num_elements, num_elements_probability in zip(
        num_elements_support, num_elements_probabilities
    ):
        num_atoms_support, num_atoms_probabilities = (
            distribution.num_atoms_and_probabilities_for_given_num_elements(
                int(num_elements)
            )
        )
        element_combinations = list(
            combinations(formula_factory._elements_as_tuple, int(num_elements))
        )
        for num_atoms, num_atoms_probability in zip(
            num_atoms_support, num_atoms_probabilities
        ):
            atom_count_partitions = _positive_integer_partitions(
                total=int(num_atoms), parts=int(num_elements)
            )
            formula_probability = float(
                num_elements_probability
                * num_atoms_probability
                / len(element_combinations)
                / len(atom_count_partitions)
            )
            for elements in element_combinations:
                for atom_counts in atom_count_partitions:
                    formula = "".join(
                        f"{element}{count}"
                        for element, count in zip(elements, atom_counts)
                    )
                    formula_probabilities[formula] = formula_probability

    assert formula_probabilities == pytest.approx(expected_formula_probabilities)


def test_fixed_set_formula_factory_rejects_empty_formula_space() -> None:
    with pytest.raises(ValidationError, match=r" at least 1 item"):
        FixedSetFormulaFactory(random_seed=0, formula_space=[])


def test_fixed_set_formula_factory_rejects_duplicate_formulae() -> None:
    with pytest.raises(
        ValidationError, match="formula_space must not contain duplicate formulae"
    ):
        _build_fixed_set_formula_factory("NaCl", "ClNa")


def test_fixed_set_formula_factory_num_possible_stoichiometries() -> None:
    factory = _build_fixed_set_formula_factory("Na2Cl1", "Li2O1", "Li1", "C3O1")

    assert factory.num_possible_stoichiometries == 3


def test_fixed_set_formula_factory_sampling_is_stable_across_input_order() -> None:
    factory_a = FixedSetFormulaFactory(
        random_seed=42, formula_space=[Formula("Na1Cl1"), Formula("Li1"), Formula("O2")]
    )
    factory_b = FixedSetFormulaFactory(
        random_seed=42, formula_space=[Formula("O2"), Formula("Na1Cl1"), Formula("Li1")]
    )

    samples_a = [str(formula) for formula in factory_a.draw_n_samples(20)]
    samples_b = [str(formula) for formula in factory_b.draw_n_samples(20)]

    assert samples_a == samples_b
