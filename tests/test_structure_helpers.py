"""Tests for structure helpers."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure  # type: ignore [attr-defined]
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.helpers.structure import (
    get_ase_atoms_midpoint_along_c_axis,
    get_centered_ase_atoms_along_c_axis,
    get_maximum_box_aspect_ratio,
    get_minimum_interatomic_distance,
    get_strained_structures,
    modify_structure_periodicity,
    split_items_into_batches,
)
from tests.helpers.constants.structure import (
    FLOATING_POINT_TOLERANCE,
    TEST_001_FE_SLAB,
    TEST_111_FE_SLAB,
    TEST_BCC_FE_STRUCTURE,
    TEST_LINBO3_SLAB,
    TEST_SI_STRUCTURE,
)

ONE_ATOM_STRUCTURE_SPECIES = ["Li"]
TWO_ATOM_STRUCTURE_SPECIES = ["Li", "Li"]


def get_structure(
    lattice_abc: tuple[float, float, float] = (5.0, 5.0, 5.0),
    species: list[str] = ONE_ATOM_STRUCTURE_SPECIES,
    coords: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
    *,
    coords_are_cartesian: bool = True,
) -> Structure:
    lattice: Lattice = Lattice.orthorhombic(*lattice_abc)
    return Structure(
        lattice=lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=coords_are_cartesian,
    )


@pytest.mark.parametrize(
    "structure",
    [TEST_LINBO3_SLAB, TEST_BCC_FE_STRUCTURE, TEST_001_FE_SLAB, TEST_111_FE_SLAB],
)
def test_centering_structure(structure: PMG_Structure) -> None:
    ase_atoms = AseAtomsAdaptor.get_atoms(structure)

    original_center = get_ase_atoms_midpoint_along_c_axis(atoms=ase_atoms)

    centered_atoms = get_centered_ase_atoms_along_c_axis(atoms=ase_atoms)
    new_center = get_ase_atoms_midpoint_along_c_axis(atoms=centered_atoms)

    assert not np.isclose(original_center, 0.0, atol=FLOATING_POINT_TOLERANCE)
    assert np.isclose(new_center, 0.0, atol=FLOATING_POINT_TOLERANCE)


def test_get_minimum_interatomic_distance_single_atom_with_self_images() -> None:
    structure = get_structure(lattice_abc=(3.0, 4.0, 5.0))

    result = get_minimum_interatomic_distance(structure)
    assert result == pytest.approx(3.0)


def test_get_minimum_interatomic_distance_two_atoms() -> None:
    structure = get_structure(
        lattice_abc=(10.0, 10.0, 10.0),
        species=TWO_ATOM_STRUCTURE_SPECIES,
        coords=((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
    )
    result = get_minimum_interatomic_distance(structure)

    assert result == pytest.approx(5.0)


def test_get_minimum_interatomic_distance_uses_all_pairwise_distances() -> None:
    structure = get_structure(
        lattice_abc=(10.0, 10.0, 10.0),
        species=["Li", "Li", "Li"],
        coords=((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (7.0, 0.0, 0.0)),
    )

    result = get_minimum_interatomic_distance(structure)

    assert result == pytest.approx(2.0)


def test_get_minimum_interatomic_distance_changes_with_modified_periodicity() -> None:
    structure = get_structure(
        lattice_abc=(10.0, 10.0, 1.0),
        species=ONE_ATOM_STRUCTURE_SPECIES,
        coords=((0.0, 0.0, 0.0),),
    )

    fully_periodic_distance = get_minimum_interatomic_distance(structure)
    slab_structure = modify_structure_periodicity(structure, (True, True, False))
    slab_distance = get_minimum_interatomic_distance(slab_structure)

    assert fully_periodic_distance == pytest.approx(1.0)
    assert slab_distance == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("lattice_abc", "expected_maximum_aspect_ratio"),
    [((4.0, 4.0, 4.0), 1.0), ((2.0, 4.0, 10.0), 5.0)],
)
def test_get_box_aspect_ratio(
    lattice_abc: tuple[float, float, float], expected_maximum_aspect_ratio: float
) -> None:
    structure = get_structure(lattice_abc=lattice_abc)
    assert get_maximum_box_aspect_ratio(structure) == pytest.approx(
        expected_maximum_aspect_ratio
    )


def test_get_strained_structures_applies_isotropic_strains() -> None:
    strains = np.array([-0.01, 0.0, 0.01], dtype=np.float64)

    strained_structures = get_strained_structures(
        structure=TEST_SI_STRUCTURE, strains=strains
    )

    assert len(strained_structures) == len(strains)
    assert [structure.volume for structure in strained_structures] == pytest.approx(
        [TEST_SI_STRUCTURE.volume * (1.0 + strain) ** 3 for strain in strains],
        abs=FLOATING_POINT_TOLERANCE,
    )
    assert all(
        np.all((structure.frac_coords >= 0.0) & (structure.frac_coords < 1.0))
        for structure in strained_structures
    )


def test_split_items_into_batches_accepts_non_structure_items() -> None:
    items = ("first", "second", "third")

    batches = split_items_into_batches(items, items_per_batch=2)

    assert batches == [["first", "second"], ["third"]]


@pytest.mark.parametrize(
    ("items", "items_per_batch", "expected_batches"),
    [
        ([], 2, []),
        ([], None, [[]]),
        (["first", "second"], None, [["first", "second"]]),
        (["first", "second", "third"], 2, [["first", "second"], ["third"]]),
    ],
)
def test_split_items_into_batches_preserves_existing_batching_behavior(
    items: list[str], items_per_batch: int | None, expected_batches: list[list[str]]
) -> None:
    assert split_items_into_batches(items, items_per_batch) == expected_batches


@pytest.mark.parametrize("items_per_batch", [0, -1])
def test_split_items_into_batches_rejects_non_positive_batch_sizes(
    items_per_batch: int,
) -> None:
    with pytest.raises(ValueError, match="items_per_batch must be a positive integer"):
        split_items_into_batches(["first"], items_per_batch)
