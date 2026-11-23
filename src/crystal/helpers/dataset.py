"""Dataset summary helpers for complete atomistic datasets."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Self, cast

import numpy as np
from ase.io import iread
from pydantic import BaseModel, ConfigDict, model_validator, validate_call

from crystal.helpers.cohesive_energy import CohesiveEnergiesCalculator
from crystal.helpers.properties import get_force_norms, get_stress_norms
from crystal.helpers.structure_summary import StructureSummary
from crystal.schemas.arrays import FloatArray, IntArray
from crystal.schemas.isolated_atom_energies import IsolatedAtomEnergies
from crystal.schemas.properties import ForceComponent, StressComponent


class DatasetSummary(BaseModel):
    """Raw and lazily derived values for one complete extxyz dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    label: str
    structure_summaries: list[StructureSummary]
    isolated_atom_energies: IsolatedAtomEnergies | None = None

    @model_validator(mode="after")
    def validate_consistent_charge_presence(self) -> Self:
        """Validate that charges are present for every structure or none."""
        charge_presence = {
            structure_summary.charges is not None
            for structure_summary in self.structure_summaries
        }
        if len(charge_presence) > 1:
            error_msg = (
                "DatasetSummary requires charges to be present for every "
                "structure or missing for every structure."
            )
            raise ValueError(error_msg)
        return self

    @classmethod
    def from_extxyz(
        cls,
        *,
        dataset_path: Path,
        label: str,
        isolated_atom_energies: IsolatedAtomEnergies | None = None,
    ) -> Self:
        """Build a dataset summary from a complete extxyz dataset."""
        structure_summaries: list[StructureSummary] = []
        expected_has_charges: bool | None = None
        for structure_index, structure_atoms in enumerate(
            iread(dataset_path, index=":")
        ):
            structure_summary = StructureSummary.from_atoms(structure_atoms)
            has_charges = structure_summary.charges is not None

            if expected_has_charges is None:
                expected_has_charges = has_charges
            elif has_charges != expected_has_charges:
                expected_state = "present" if expected_has_charges else "missing"
                actual_state = "present" if has_charges else "missing"
                error_msg = (
                    "DatasetSummary requires charges to be present for every "
                    "structure or missing for every structure; structure "
                    f"{structure_index} in dataset {label!r} has charges "
                    f"{actual_state}, but expected charges {expected_state}."
                )
                raise ValueError(error_msg)

            structure_summaries.append(structure_summary)
        return cls(
            label=label,
            structure_summaries=structure_summaries,
            isolated_atom_energies=isolated_atom_energies,
        )

    @cached_property
    def atomic_numbers(self) -> IntArray:
        """Return atomic numbers for all structures."""
        return np.concatenate(
            [
                structure_summary.atomic_numbers
                for structure_summary in self.structure_summaries
            ]
        )

    @cached_property
    def forces(self) -> FloatArray:
        """Return force components for all structures."""
        return np.vstack(
            [structure_summary.forces for structure_summary in self.structure_summaries]
        )

    @cached_property
    def stress(self) -> FloatArray:
        """Return stress components by structures."""
        return np.vstack(
            [structure_summary.stress for structure_summary in self.structure_summaries]
        )

    def _has_charges(self) -> bool:
        """Return whether charges are present for every structure."""
        return self.structure_summaries[0].charges is not None

    @cached_property
    def charges(self) -> FloatArray | None:
        """Return charges flattened across structures."""
        if not self._has_charges():
            return None
        return np.concatenate(
            [
                cast("FloatArray", structure_summary.charges)
                for structure_summary in self.structure_summaries
            ]
        )

    @cached_property
    def total_energies(self) -> FloatArray:
        """Return total energy per structure."""
        return np.asarray(
            [
                structure_summary.total_energy
                for structure_summary in self.structure_summaries
            ],
            dtype=np.float64,
        )

    @property
    def atom_counts(self) -> IntArray:
        """Return atom count per structure."""
        return np.asarray(
            [
                len(structure_summary.atomic_numbers)
                for structure_summary in self.structure_summaries
            ],
            dtype=np.int_,
        )

    @property
    def unique_element_counts(self) -> IntArray:
        """Return unique element count per structure."""
        return np.asarray(
            [
                np.unique(structure_summary.atomic_numbers).size
                for structure_summary in self.structure_summaries
            ],
            dtype=np.int_,
        )

    @property
    def force_norms(self) -> FloatArray:
        """Return per-atom force norms flattened across structures."""
        return get_force_norms(self.forces)

    @property
    def stress_norms(self) -> FloatArray:
        """Return per-structure stress norms."""
        return get_stress_norms(self.stress)

    @property
    def force_components_by_element(self) -> dict[int, FloatArray]:
        """Return force component arrays grouped by element."""
        return self._get_values_by_element(self.forces)

    @property
    def force_norms_by_element(self) -> dict[int, FloatArray]:
        """Return force norm arrays grouped by element."""
        return self._get_values_by_element(self.force_norms)

    @property
    def charges_by_element(self) -> dict[int, FloatArray] | None:
        """Return charge arrays grouped by element."""
        if not self._has_charges():
            return None
        return self._get_values_by_element(cast("FloatArray", self.charges))

    @validate_call
    def select_force_components(
        self, *, components: Sequence[ForceComponent] | None = None
    ) -> FloatArray:
        """Return selected force components flattened across atoms."""
        selected_components = list(ForceComponent) if components is None else components
        return self.forces[
            :, [component.value for component in selected_components]
        ].ravel()

    @validate_call
    def select_stress_components(
        self, *, components: Sequence[StressComponent] | None = None
    ) -> FloatArray:
        """Return selected stress components flattened across structures."""
        selected_components = (
            list(StressComponent) if components is None else components
        )
        return self.stress[
            :, [component.value for component in selected_components]
        ].ravel()

    def cohesive_energies_per_atom(self) -> FloatArray:
        """Return cohesive energies per atom in eV/atom."""
        if self.isolated_atom_energies is None:
            error_msg = (
                "DatasetSummary requires isolated_atom_energies to compute "
                "cohesive energies."
            )
            raise ValueError(error_msg)
        calculator = CohesiveEnergiesCalculator(
            isolated_atom_energies=self.isolated_atom_energies
        )
        return np.asarray(
            calculator.from_structure_summaries(
                structure_summaries=self.structure_summaries
            ),
            dtype=np.float64,
        )

    def _get_values_by_element(self, values: FloatArray) -> dict[int, FloatArray]:
        return {
            int(atomic_number): values[self.atomic_numbers == atomic_number]
            for atomic_number in np.unique(self.atomic_numbers)
        }
