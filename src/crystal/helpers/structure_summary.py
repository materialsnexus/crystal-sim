"""Reusable summaries of calculated atomistic structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from ase import Atoms

from crystal.schemas.arrays import FloatArray, IntArray
from crystal.schemas.properties import FORCE_VECTOR_LENGTH, VOIGT_STRESS_VECTOR_LENGTH

ASE_CHARGES_PROPERTY = "charges"


@dataclass(frozen=True, slots=True)
class StructureSummary:
    """Raw calculated values for one atomistic structure."""

    atomic_numbers: IntArray
    forces: FloatArray
    charges: FloatArray | None
    stress: FloatArray
    total_energy: float

    def __post_init__(self) -> None:
        """Validate array shapes against the atom count."""
        atom_count = len(self.atomic_numbers)
        expected_forces_shape = (atom_count, FORCE_VECTOR_LENGTH)
        if self.forces.shape != expected_forces_shape:
            error_msg = (
                "StructureSummary forces shape must match "
                f"(atom_count, 3): {self.forces.shape} != {expected_forces_shape}."
            )
            raise ValueError(error_msg)
        expected_charges_shape = (atom_count,)
        if self.charges is not None and self.charges.shape != expected_charges_shape:
            error_msg = (
                "StructureSummary charges shape must match "
                f"(atom_count,): {self.charges.shape} != {expected_charges_shape}."
            )
            raise ValueError(error_msg)
        expected_stress_shape = (VOIGT_STRESS_VECTOR_LENGTH,)
        if self.stress.shape != expected_stress_shape:
            error_msg = (
                "StructureSummary stress shape must match "
                f"(6,): {self.stress.shape} != {expected_stress_shape}."
            )
            raise ValueError(error_msg)

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> Self:
        """Build a summary from ASE atoms with attached calculator results."""
        if atoms.calc is None:
            error_msg = "StructureSummary requires ASE atoms with a calculator."
            raise ValueError(error_msg)
        total_energy = float(
            atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        )
        forces = np.asarray(
            atoms.get_forces(),  # type: ignore[no-untyped-call]
            dtype=np.float64,
        )
        stress = np.asarray(
            atoms.get_stress(voigt=True),  # type: ignore[no-untyped-call]
            dtype=np.float64,
        )
        optional_charges = (
            np.asarray(
                atoms.get_charges(),  # type: ignore[no-untyped-call]
                dtype=np.float64,
            )
            if ASE_CHARGES_PROPERTY in atoms.calc.results
            else None
        )
        return cls(
            atomic_numbers=np.asarray(
                atoms.get_atomic_numbers(),  # type: ignore[no-untyped-call]
                dtype=np.int_,
            ),
            total_energy=total_energy,
            forces=forces,
            stress=stress,
            charges=optional_charges,
        )
