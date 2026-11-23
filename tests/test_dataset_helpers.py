"""Tests for dataset analysis helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from numpy.typing import NDArray

from crystal.helpers.cohesive_energy import CohesiveEnergiesCalculator
from crystal.helpers.dataset import DatasetSummary
from crystal.helpers.structure_summary import ASE_CHARGES_PROPERTY, StructureSummary
from crystal.schemas.properties import ForceComponent, StressComponent

DATASET_LABEL = "mock dataset"
ISOLATED_ATOM_ENERGIES = {"H": -1.0, "He": -2.0}

MOCK_STRUCTURE_0_H_H = StructureSummary(
    atomic_numbers=np.asarray([1, 1], dtype=np.int_),
    forces=np.asarray([(3.0, 4.0, 0.0), (0.0, 0.0, 2.0)], dtype=np.float64),
    stress=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
    charges=np.asarray([0.1, -0.1], dtype=np.float64),
    total_energy=-5.0,
)
MOCK_STRUCTURE_1_H_HE = StructureSummary(
    atomic_numbers=np.asarray([1, 2], dtype=np.int_),
    forces=np.asarray([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float64),
    stress=np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=np.float64),
    charges=np.asarray([0.2, 0.3], dtype=np.float64),
    total_energy=-8.0,
)
MOCK_STRUCTURE_WITHOUT_CHARGES = StructureSummary(
    atomic_numbers=np.asarray([1, 2], dtype=np.int_),
    forces=np.asarray([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float64),
    stress=np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=np.float64),
    charges=None,
    total_energy=-8.0,
)
MOCK_STRUCTURES = (MOCK_STRUCTURE_0_H_H, MOCK_STRUCTURE_1_H_HE)
EXPECTED_ATOMIC_NUMBERS = np.concatenate(
    [structure.atomic_numbers for structure in MOCK_STRUCTURES]
)
EXPECTED_FORCES = np.vstack([structure.forces for structure in MOCK_STRUCTURES])
EXPECTED_FORCE_NORMS = np.linalg.norm(EXPECTED_FORCES, axis=1)
EXPECTED_STRESS = np.vstack([structure.stress for structure in MOCK_STRUCTURES])
EXPECTED_STRESS_NORMS = np.linalg.norm(EXPECTED_STRESS, axis=1)
EXPECTED_CHARGES = np.concatenate([structure.charges for structure in MOCK_STRUCTURES])
EXPECTED_TOTAL_ENERGIES = np.asarray(
    [structure.total_energy for structure in MOCK_STRUCTURES], dtype=np.float64
)
EXPECTED_ATOM_COUNTS = np.asarray([2, 2])
EXPECTED_UNIQUE_ELEMENT_COUNTS = np.asarray([1, 2])


def _get_atoms_with_results(structure: StructureSummary) -> Atoms:
    atoms = Atoms(
        numbers=structure.atomic_numbers,
        positions=[
            (float(index), 0.0, 0.0) for index in range(len(structure.atomic_numbers))
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    optional_charge: dict[str, Any] = (
        {ASE_CHARGES_PROPERTY: structure.charges}
        if structure.charges is not None
        else {}
    )
    calculator_kwargs: dict[str, Any] = {
        "energy": structure.total_energy,
        "forces": structure.forces,
        "stress": structure.stress,
        **optional_charge,
    }
    atoms.calc = SinglePointCalculator(atoms, **calculator_kwargs)  # type: ignore[no-untyped-call]
    return atoms


def _write_dataset(
    path: Path, structure_summaries: tuple[StructureSummary, ...] = MOCK_STRUCTURES
) -> Path:
    write(
        path,
        [_get_atoms_with_results(structure) for structure in structure_summaries],
        format="extxyz",
    )
    return path


@pytest.fixture
def dataset_summary(tmp_path: Path) -> DatasetSummary:
    return DatasetSummary.from_extxyz(
        dataset_path=_write_dataset(tmp_path / "dataset.extxyz"), label=DATASET_LABEL
    )


def test_dataset_summary_from_extxyz_stores_raw_structure_ordered_arrays(
    dataset_summary: DatasetSummary,
) -> None:
    assert dataset_summary.label == DATASET_LABEL
    assert len(dataset_summary.structure_summaries) == 2
    assert all(
        isinstance(structure, StructureSummary)
        for structure in dataset_summary.structure_summaries
    )

    np.testing.assert_array_equal(
        dataset_summary.structure_summaries[0].atomic_numbers,
        np.asarray([1, 1], dtype=np.int_),
    )
    np.testing.assert_array_equal(
        dataset_summary.structure_summaries[1].atomic_numbers,
        np.asarray([1, 2], dtype=np.int_),
    )
    np.testing.assert_allclose(
        dataset_summary.structure_summaries[0].forces,
        np.asarray(MOCK_STRUCTURE_0_H_H.forces, dtype=np.float64),
    )
    np.testing.assert_allclose(
        dataset_summary.structure_summaries[1].forces,
        np.asarray(MOCK_STRUCTURE_1_H_HE.forces, dtype=np.float64),
    )
    structure_0_charges = dataset_summary.structure_summaries[0].charges
    structure_1_charges = dataset_summary.structure_summaries[1].charges
    assert structure_0_charges is not None
    assert structure_1_charges is not None
    np.testing.assert_allclose(
        structure_0_charges, np.asarray(MOCK_STRUCTURE_0_H_H.charges, dtype=np.float64)
    )
    np.testing.assert_allclose(
        structure_1_charges, np.asarray(MOCK_STRUCTURE_1_H_HE.charges, dtype=np.float64)
    )
    np.testing.assert_allclose(
        dataset_summary.structure_summaries[0].stress, EXPECTED_STRESS[0]
    )
    np.testing.assert_allclose(
        dataset_summary.structure_summaries[1].stress, EXPECTED_STRESS[1]
    )
    assert (
        dataset_summary.structure_summaries[0].total_energy
        == MOCK_STRUCTURE_0_H_H.total_energy
    )
    assert (
        dataset_summary.structure_summaries[1].total_energy
        == MOCK_STRUCTURE_1_H_HE.total_energy
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        (
            "forces",
            np.zeros((1, 3), dtype=np.float64),
            "forces shape must match (atom_count, 3): (1, 3) != (2, 3)",
        ),
        (
            "forces",
            np.zeros((2, 2), dtype=np.float64),
            "forces shape must match (atom_count, 3): (2, 2) != (2, 3)",
        ),
        (
            "charges",
            np.zeros(1, dtype=np.float64),
            "charges shape must match (atom_count,): (1,) != (2,)",
        ),
        (
            "charges",
            np.zeros((2, 1), dtype=np.float64),
            "charges shape must match (atom_count,): (2, 1) != (2,)",
        ),
        (
            "stress",
            np.zeros(3, dtype=np.float64),
            "stress shape must match (6,): (3,) != (6,)",
        ),
        (
            "stress",
            np.zeros((1, 6), dtype=np.float64),
            "stress shape must match (6,): (1, 6) != (6,)",
        ),
    ],
)
def test_structure_summary_validates_input_shapes(
    field: str, value: NDArray[Any], expected_error: str
) -> None:
    forces = np.zeros((2, 3), dtype=np.float64)
    charges = np.zeros(2, dtype=np.float64)
    stress = np.zeros(6, dtype=np.float64)
    if field == "forces":
        forces = value
    elif field == "charges":
        charges = value
    elif field == "stress":
        stress = value

    with pytest.raises(ValueError, match=re.escape(expected_error)):
        _ = StructureSummary(
            atomic_numbers=np.asarray([1, 2], dtype=np.int_),
            forces=forces,
            charges=charges,
            stress=stress,
            total_energy=-1.0,
        )


def test_structure_summary_accepts_missing_charges() -> None:
    structure_summary = StructureSummary(
        atomic_numbers=np.asarray([1, 2], dtype=np.int_),
        forces=np.asarray([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float64),
        stress=np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=np.float64),
        charges=None,
        total_energy=-8.0,
    )

    assert structure_summary.charges is None


@pytest.mark.parametrize(
    "structure_summaries",
    [MOCK_STRUCTURES, (MOCK_STRUCTURE_WITHOUT_CHARGES, MOCK_STRUCTURE_WITHOUT_CHARGES)],
)
def test_dataset_summary_from_extxyz_accepts_consistent_charge_presence(
    tmp_path: Path, structure_summaries: tuple[StructureSummary, ...]
) -> None:
    dataset_summary = DatasetSummary.from_extxyz(
        dataset_path=_write_dataset(
            tmp_path / "dataset.extxyz", structure_summaries=structure_summaries
        ),
        label=DATASET_LABEL,
    )

    expected_has_charges = structure_summaries[0].charges is not None
    assert all(
        (structure_summary.charges is not None) == expected_has_charges
        for structure_summary in dataset_summary.structure_summaries
    )
    if expected_has_charges:
        assert dataset_summary.charges is not None
        assert dataset_summary.charges_by_element is not None
    else:
        assert dataset_summary.charges is None
        assert dataset_summary.charges_by_element is None


def test_dataset_summary_from_extxyz_rejects_mixed_charge_presence(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "DatasetSummary requires charges to be present for every structure "
            "or missing for every structure; structure 1 in dataset "
            f"{DATASET_LABEL!r} has charges present, but expected charges missing."
        ),
    ):
        _ = DatasetSummary.from_extxyz(
            dataset_path=_write_dataset(
                tmp_path / "dataset.extxyz",
                structure_summaries=(
                    MOCK_STRUCTURE_WITHOUT_CHARGES,
                    MOCK_STRUCTURE_0_H_H,
                    MOCK_STRUCTURE_1_H_HE,
                ),
            ),
            label=DATASET_LABEL,
        )


def test_dataset_summary_rejects_mixed_charge_presence() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "DatasetSummary requires charges to be present for every structure "
            "or missing for every structure."
        ),
    ):
        _ = DatasetSummary(
            label=DATASET_LABEL,
            structure_summaries=(
                MOCK_STRUCTURE_WITHOUT_CHARGES,
                MOCK_STRUCTURE_0_H_H,
                MOCK_STRUCTURE_1_H_HE,
            ),
        )


def test_dataset_summary_returns_none_for_charge_properties_when_charges_are_missing() -> (
    None
):
    dataset_summary = DatasetSummary(
        label=DATASET_LABEL,
        structure_summaries=(
            MOCK_STRUCTURE_WITHOUT_CHARGES,
            MOCK_STRUCTURE_WITHOUT_CHARGES,
        ),
    )

    assert dataset_summary.charges is None
    assert dataset_summary.charges_by_element is None


def test_dataset_summary_derives_flattened_values(
    dataset_summary: DatasetSummary,
) -> None:
    assert dataset_summary.charges is not None
    np.testing.assert_array_equal(
        dataset_summary.atomic_numbers, EXPECTED_ATOMIC_NUMBERS
    )
    np.testing.assert_allclose(dataset_summary.forces, EXPECTED_FORCES)
    np.testing.assert_allclose(dataset_summary.stress, EXPECTED_STRESS)
    np.testing.assert_allclose(dataset_summary.charges, EXPECTED_CHARGES)
    np.testing.assert_allclose(dataset_summary.total_energies, EXPECTED_TOTAL_ENERGIES)


def test_dataset_summary_derives_count_values(dataset_summary: DatasetSummary) -> None:
    np.testing.assert_array_equal(dataset_summary.atom_counts, EXPECTED_ATOM_COUNTS)
    np.testing.assert_array_equal(
        dataset_summary.unique_element_counts, EXPECTED_UNIQUE_ELEMENT_COUNTS
    )


def test_dataset_summary_derives_norms(dataset_summary: DatasetSummary) -> None:
    np.testing.assert_allclose(dataset_summary.force_norms, EXPECTED_FORCE_NORMS)
    np.testing.assert_allclose(dataset_summary.stress_norms, EXPECTED_STRESS_NORMS)


def test_dataset_summary_derives_element_grouped_values(
    dataset_summary: DatasetSummary,
) -> None:
    np.testing.assert_allclose(
        dataset_summary.force_components_by_element[1],
        EXPECTED_FORCES[EXPECTED_ATOMIC_NUMBERS == 1],
    )
    np.testing.assert_allclose(
        dataset_summary.force_norms_by_element[1],
        EXPECTED_FORCE_NORMS[EXPECTED_ATOMIC_NUMBERS == 1],
    )
    charges_by_element = dataset_summary.charges_by_element
    assert charges_by_element is not None
    np.testing.assert_allclose(
        charges_by_element[2], EXPECTED_CHARGES[EXPECTED_ATOMIC_NUMBERS == 2]
    )


def test_dataset_summary_selects_force_and_stress_components(
    dataset_summary: DatasetSummary,
) -> None:
    np.testing.assert_allclose(
        dataset_summary.select_force_components(
            components=[ForceComponent.x, ForceComponent.z]
        ),
        EXPECTED_FORCES[:, [0, 2]].ravel(),
    )
    np.testing.assert_allclose(
        dataset_summary.select_stress_components(
            components=[StressComponent.xx, StressComponent.xz]
        ),
        EXPECTED_STRESS[:, [0, 4]].ravel(),
    )
    np.testing.assert_allclose(
        dataset_summary.select_force_components(), EXPECTED_FORCES.ravel()
    )
    np.testing.assert_allclose(
        dataset_summary.select_stress_components(), EXPECTED_STRESS.ravel()
    )


def test_dataset_summary_computes_cohesive_energies_per_atom(
    dataset_summary: DatasetSummary,
) -> None:
    summary_with_isolated_atom_energies = DatasetSummary(
        label=dataset_summary.label,
        structure_summaries=dataset_summary.structure_summaries,
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES,
    )
    cohesive_energies = summary_with_isolated_atom_energies.cohesive_energies_per_atom()
    cohesive_energy_calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES
    )
    expected_cohesive_energies = np.asarray(
        cohesive_energy_calculator.from_structure_summaries(
            structure_summaries=dataset_summary.structure_summaries
        ),
        dtype=np.float64,
    )

    np.testing.assert_allclose(cohesive_energies, expected_cohesive_energies)


def test_dataset_summary_rejects_cohesive_energies_without_isolated_atom_energies(
    dataset_summary: DatasetSummary,
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "DatasetSummary requires isolated_atom_energies to compute cohesive energies."
        ),
    ):
        dataset_summary.cohesive_energies_per_atom()
