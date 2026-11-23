"""Helpers for isolated atom references and cohesive-energy calculations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ase.data import chemical_symbols
from pydantic import BaseModel, ConfigDict, PositiveInt
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.mlip_runner import get_mlip_isolated_atom_energies
from crystal.schemas.filtering import CohesiveEnergyFilteringSettings
from crystal.schemas.isolated_atom_energies import IsolatedAtomEnergies

if TYPE_CHECKING:
    from crystal.helpers.structure_summary import StructureSummary


class CohesiveEnergiesCalculator(BaseModel):
    """Compute cohesive energies from isolated atom references."""

    model_config = ConfigDict(frozen=True)

    isolated_atom_energies: IsolatedAtomEnergies

    def _compute_elemental_reference_energy_total_from_symbols(
        self, *, element_symbols: Sequence[str]
    ) -> float:
        """Compute the weighted elemental reference energy total in eV."""
        return sum(self.isolated_atom_energies[symbol] for symbol in element_symbols)

    def _compute_cohesive_energy_per_atom_from_symbols(
        self, *, element_symbols: Sequence[str], total_energy: float
    ) -> float:
        """Compute cohesive energy per atom for one validated symbol list."""
        elemental_reference_energy = (
            self._compute_elemental_reference_energy_total_from_symbols(
                element_symbols=element_symbols
            )
        )
        return (total_energy - elemental_reference_energy) / len(element_symbols)

    def _compute_cohesive_energies_per_atom_from_symbols(
        self,
        *,
        element_symbols_by_structure: Sequence[Sequence[str]],
        total_energies: Sequence[float],
    ) -> list[float]:
        """Compute cohesive energies per atom in eV/atom from element symbols."""
        if len(element_symbols_by_structure) != len(total_energies):
            error_msg = (
                "Number of element symbol lists must match number of total energies: "
                f"{len(element_symbols_by_structure)} element symbol lists, "
                f"{len(total_energies)} energies."
            )
            raise ValueError(error_msg)

        unique_symbols = {
            symbol for symbols in element_symbols_by_structure for symbol in symbols
        }
        missing_symbols = unique_symbols - self.isolated_atom_energies.keys()
        if missing_symbols:
            error_msg = (
                "Missing isolated atom energy references for elements: "
                f"{', '.join(sorted(missing_symbols))}"
            )
            raise ValueError(error_msg)

        return [
            self._compute_cohesive_energy_per_atom_from_symbols(
                element_symbols=symbols, total_energy=total_energy
            )
            for symbols, total_energy in zip(
                element_symbols_by_structure, total_energies, strict=True
            )
        ]

    def from_pmg_structures(
        self, *, structures: Sequence[PMG_Structure], total_energies: Sequence[float]
    ) -> list[float]:
        """Compute cohesive energies per atom for pymatgen structures."""
        return self._compute_cohesive_energies_per_atom_from_symbols(
            element_symbols_by_structure=[
                [site.specie.symbol for site in structure] for structure in structures
            ],
            total_energies=total_energies,
        )

    def from_structure_summaries(
        self, *, structure_summaries: Sequence[StructureSummary]
    ) -> list[float]:
        """Compute cohesive energies per atom for structure summaries."""
        return self._compute_cohesive_energies_per_atom_from_symbols(
            element_symbols_by_structure=[
                [
                    chemical_symbols[int(atomic_number)]
                    for atomic_number in structure_summary.atomic_numbers
                ]
                for structure_summary in structure_summaries
            ],
            total_energies=[
                structure_summary.total_energy
                for structure_summary in structure_summaries
            ],
        )


def filter_structures_by_cohesive_energy(
    *,
    structures: Sequence[PMG_Structure],
    total_energies: Sequence[float],
    settings: CohesiveEnergyFilteringSettings,
) -> list[PMG_Structure] | None:
    """Filter structures whose cohesive energy per atom exceeds a threshold.

    Return None when the threshold is disabled.
    """
    maximum_cohesive_energy_per_atom = settings.maximum_cohesive_energy_per_atom
    if maximum_cohesive_energy_per_atom is None:
        return None
    if len(structures) != len(total_energies):
        error_msg = (
            "Number of structures and total energies must match: "
            f"{len(structures)} structures, {len(total_energies)} energies."
        )
        raise ValueError(error_msg)
    if not structures:
        return []
    isolated_atom_energies = settings.isolated_atom_energies
    if isolated_atom_energies is None:
        if settings.evaluator_model is None:
            msg = (
                "Cohesive-energy filtering requires either explicit isolated atom "
                "energies or an evaluator MLIP model."
            )
            raise ValueError(msg)
        isolated_atom_energies = get_mlip_isolated_atom_energies(
            settings.evaluator_model
        )
    cohesive_energy_calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies=isolated_atom_energies
    )
    cohesive_energies_per_atom = cohesive_energy_calculator.from_pmg_structures(
        structures=structures, total_energies=total_energies
    )
    return [
        structure
        for structure, cohesive_energy_per_atom in zip(
            structures, cohesive_energies_per_atom, strict=True
        )
        if cohesive_energy_per_atom <= maximum_cohesive_energy_per_atom
    ]


def scale_total_energy_with_num_atoms(  # noqa: D103
    total_energy: float, initial_num_atoms: PositiveInt, final_num_atoms: PositiveInt
) -> float:
    if initial_num_atoms <= 0 or final_num_atoms <= 0:
        err_msg = "Number of atoms must be > 0"
        raise ValueError(err_msg)
    return total_energy * final_num_atoms / initial_num_atoms
