"""Orchestrates QE config, pseudos, and ASE Espresso runs."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import ase
from ase.calculators.calculator import all_changes
from ase.calculators.espresso import Espresso
from pydantic import model_validator
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.base_atomistic_runner import BaseAtomisticRunner
from crystal.calculators.qe.espresso_arguments_builder import EspressoArgumentsBuilder
from crystal.constants.process import DEFAULT_NUMBER_OF_PROCESSES
from crystal.helpers.structure import (
    get_centered_ase_atoms_along_c_axis,
    is_structure_c_axis_orthogonal_to_ab_plane,
)
from crystal.schemas.elements import ElementStr
from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from crystal.schemas.pseudopotential import PseudopotentialFileName
from crystal.schemas.qe import (
    IsolatedSystemPolicy,
    QeConfig,
    QeRelaxSimulationSettings,
    QeScfSimulationSettings,
    SerializableEspressoProfile,
)
from crystal.schemas.relax_settings import RelaxOptimizerSettings


class RecenteredEspresso(Espresso):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        isolated_system_policy: IsolatedSystemPolicy | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)  # type:ignore [no-untyped-call]
        self.isolated_system_policy = isolated_system_policy

    def calculate(
        self,
        atoms: ase.Atoms | None,
        properties: list[str] | None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Wrapper around the existing calculate method to recenter the atoms object before calling QE."""
        atoms = atoms or self.atoms
        assert isinstance(atoms, ase.Atoms)
        centered_atoms = adapt_atoms_to_policy(
            atoms=atoms, isolated_system_policy=self.isolated_system_policy
        )
        atoms.positions = centered_atoms.positions
        super().calculate(  # type:ignore [no-untyped-call]
            atoms=atoms, properties=properties, system_changes=system_changes
        )
        self.results = self.results.copy()


class QeRunner(BaseAtomisticRunner):
    """High-level convenience wrapper for ASE-managed QE runs."""

    config: QeConfig
    pseudopotentials: dict[ElementStr, PseudopotentialFileName]
    espresso_profile: SerializableEspressoProfile

    @model_validator(mode="after")
    def _esm_requires_orthogonal_c_axis(self) -> QeRunner:
        if self.config.input_data.system.assume_isolated != IsolatedSystemPolicy.esm:
            return self
        if is_structure_c_axis_orthogonal_to_ab_plane(structure=self.structure):
            return self
        error_msg = (
            "ESM calculations require slab structures with c-axis orthogonal to the "
            "a and b lattice vectors."
            "Orthogonalize the slab before creating the QeRunner."
        )
        raise ValueError(error_msg)

    @cached_property
    def _calculator(self) -> RecenteredEspresso:
        """Create (once) and return an ASE Espresso calculator attached to atoms."""
        calculator_arguments = EspressoArgumentsBuilder(
            config=self.config,
            pseudopotentials=self.pseudopotentials,
            profile=self.espresso_profile,
        ).calculator_kwargs
        return RecenteredEspresso(
            isolated_system_policy=self.config.input_data.system.assume_isolated,
            **calculator_arguments,
        )


def adapt_atoms_to_policy(  # noqa: D103
    atoms: ase.Atoms, isolated_system_policy: IsolatedSystemPolicy | None
) -> ase.Atoms:
    match isolated_system_policy:
        case IsolatedSystemPolicy.esm:
            return get_centered_ase_atoms_along_c_axis(atoms)
        case _:
            return atoms


def run_qe_scf(  # noqa: D103
    structure: PMG_Structure,
    simulation_settings: QeScfSimulationSettings,
    number_of_processes: int = DEFAULT_NUMBER_OF_PROCESSES,
) -> SinglePointProperties:
    espresso_profile = (
        SerializableEspressoProfile.from_number_of_processes_and_pseudo_dir(
            number_of_processes=number_of_processes,
            pseudopotential_dir=simulation_settings.pseudopotential_directory,
        )
    )
    runner = QeRunner(
        structure=structure,
        config=simulation_settings.config,
        pseudopotentials=simulation_settings.pseudopotentials,
        espresso_profile=espresso_profile,
    )
    return runner.run_single_point()


def run_qe_relax(  # noqa: D103
    structure: PMG_Structure,
    simulation_settings: QeRelaxSimulationSettings,
    number_of_processes: int = DEFAULT_NUMBER_OF_PROCESSES,
) -> RelaxProperties:
    espresso_profile = (
        SerializableEspressoProfile.from_number_of_processes_and_pseudo_dir(
            number_of_processes=number_of_processes,
            pseudopotential_dir=simulation_settings.pseudopotential_directory,
        )
    )
    runner = QeRunner(
        structure=structure,
        config=simulation_settings.config,
        pseudopotentials=simulation_settings.pseudopotentials,
        espresso_profile=espresso_profile,
    )
    optimizer_settings = RelaxOptimizerSettings.model_validate(
        simulation_settings.model_dump()
    )
    return runner.run_relax(
        optimizer_settings=optimizer_settings,
        constraints=simulation_settings.constraints,
    )
