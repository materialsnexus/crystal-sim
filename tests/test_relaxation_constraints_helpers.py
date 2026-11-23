"""Unit tests for relaxation constraint helper functions."""

from __future__ import annotations

from ase import Atoms

from crystal.schemas.relaxation_constraints import (
    CenterFreezeSettings,
    ExternalFacesFreezeSettings,
    RegionBasedFixedAtomSelectionType,
    RelaxationConstraints,
)


def _build_layered_atoms() -> Atoms:
    return Atoms(
        "H4",
        positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 4.5], [0.0, 0.0, 5.5], [0.0, 0.0, 9.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=[True, True, True],
    )


def test_select_fixed_atom_indices_for_center_region_with_angstrom_thickness() -> None:
    atoms = _build_layered_atoms()
    settings = CenterFreezeSettings(
        region_type=RegionBasedFixedAtomSelectionType.center_region,
        thickness_in_angstrom=2.0,
    )
    indices = RelaxationConstraints.select_fixed_atom_indices_for_center_region(
        atoms, settings
    )
    assert indices == [1, 2]


def test_select_fixed_atom_indices_for_center_region_with_wide_angstrom_thickness() -> (
    None
):
    atoms = _build_layered_atoms()
    settings = CenterFreezeSettings(
        region_type=RegionBasedFixedAtomSelectionType.center_region,
        thickness_in_angstrom=4.0,
    )
    indices = RelaxationConstraints.select_fixed_atom_indices_for_center_region(
        atoms, settings
    )
    assert indices == [1, 2]


def test_select_fixed_atom_indices_for_external_faces_with_symmetric_thickness() -> (
    None
):
    atoms = _build_layered_atoms()
    settings = ExternalFacesFreezeSettings(
        region_type=RegionBasedFixedAtomSelectionType.external_faces,
        thickness_in_angstrom=1.5,
    )
    indices = RelaxationConstraints.select_fixed_atom_indices_for_external_faces(
        atoms, settings
    )
    assert indices == [0, 3]


def test_merge_fixed_atom_indices_returns_sorted_unique_indices() -> None:
    indices = RelaxationConstraints.merge_fixed_atom_indices([4, 2, 2], [3, 2, 1])
    assert indices == [1, 2, 3, 4]
