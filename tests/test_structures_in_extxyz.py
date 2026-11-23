"""Tests for MD equilibration verification structures."""

from pathlib import Path

from ase.io import read

MD_EQUILIBRATION_VERIFICATION_STRUCTURES_PATH = (
    Path(__file__).parent
    / "data"
    / "verification"
    / "md_equilibration_verification_structures.extxyz"
)


def test_md_equilibration_verification_extxyz_contains_expected_structures() -> None:
    structures = read(
        MD_EQUILIBRATION_VERIFICATION_STRUCTURES_PATH.as_posix(), index=":"
    )

    assert len(structures) == 36
