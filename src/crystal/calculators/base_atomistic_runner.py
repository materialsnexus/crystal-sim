"""Base classes, functions, and mappings used by all relaxers."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from copy import deepcopy
from functools import cached_property, partial
from pathlib import Path

import ase
from ase.calculators.calculator import BaseCalculator
from ase.constraints import FixAtoms
from ase.filters import UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE, FIRE2, LBFGS
from ase.optimize.optimize import Optimizer
from pydantic import ConfigDict
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.constants.qe import DEFAULT_RELAX_LOGFILE
from crystal.schemas.properties import (
    RelaxProperties,
    SinglePointProperties,
    StepProperties,
)
from crystal.schemas.relax_settings import OptimizerType, RelaxOptimizerSettings
from crystal.schemas.relaxation_constraints import RelaxationConstraints
from crystal.schemas.serialization import MSONableBaseModel

DEFAULT_RELAX_TRAJECTORY_FILE = "relax.traj"


def bfgs_optimizer(atoms: ase.Atoms, logfile: str = DEFAULT_RELAX_LOGFILE) -> BFGS:  # noqa: D103
    return BFGS(atoms, logfile=logfile)


def lbfgs_optimizer(atoms: ase.Atoms, logfile: str = DEFAULT_RELAX_LOGFILE) -> LBFGS:  # noqa: D103
    return LBFGS(atoms, logfile=logfile)


def fire_optimizer(atoms: ase.Atoms, logfile: str = DEFAULT_RELAX_LOGFILE) -> FIRE:  # noqa: D103
    return FIRE(atoms, logfile=logfile)  # type: ignore[no-any-return]


def fire2_optimizer(atoms: ase.Atoms, logfile: str = DEFAULT_RELAX_LOGFILE) -> FIRE2:  # noqa: D103
    return FIRE2(atoms, logfile=logfile)


OPTIMIZER_TYPE_TO_OPTIMIZER_CALLABLE_MAPPING: dict[
    OptimizerType, Callable[[ase.Atoms, str], Optimizer]
] = {
    OptimizerType.BFGS: bfgs_optimizer,
    OptimizerType.LBFGS: lbfgs_optimizer,
    OptimizerType.FIRE: fire_optimizer,
    OptimizerType.FIRE2: fire2_optimizer,
}


class BaseAtomisticRunner(MSONableBaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: PMG_Structure

    @cached_property
    def _ase_atoms_adaptor(self) -> AseAtomsAdaptor:
        return AseAtomsAdaptor()

    @cached_property
    def _ase_atoms(self) -> ase.Atoms:
        return self._ase_atoms_adaptor.get_atoms(self.structure)

    def _get_atoms(self, atoms_or_filter: ase.Atoms | UnitCellFilter) -> ase.Atoms:
        return (
            atoms_or_filter.atoms
            if hasattr(atoms_or_filter, "atoms")
            else atoms_or_filter
        )

    @abstractmethod
    @cached_property
    def _calculator() -> BaseCalculator:
        raise NotImplementedError

    @staticmethod
    def _optimizer(
        atoms_or_filter: ase.Atoms | UnitCellFilter,
        optimizer_type: OptimizerType = OptimizerType.BFGS,
        logfile: str = DEFAULT_RELAX_LOGFILE,
    ) -> Optimizer:
        optimizer_callable = OPTIMIZER_TYPE_TO_OPTIMIZER_CALLABLE_MAPPING.get(
            optimizer_type
        )
        if optimizer_callable is None:
            error_msg = (
                "Invalid optimizer type selected. "
                f"Valid types: {list(OPTIMIZER_TYPE_TO_OPTIMIZER_CALLABLE_MAPPING.keys())}"
            )
            raise ValueError(error_msg)
        optimizer = optimizer_callable(atoms_or_filter, logfile)  # type: ignore [arg-type]
        assert isinstance(optimizer, Optimizer)
        return optimizer

    def _initialize_optimizer(
        self,
        atoms: ase.Atoms,
        optimizer_settings: RelaxOptimizerSettings,
        fix_cell: bool,  # noqa: FBT001
        *,
        store_trajectory: bool = True,
        output_directory: Path | None = None,
    ) -> Optimizer:
        unit_cell_filter = None
        if not fix_cell:
            unit_cell_filter = UnitCellFilter(atoms=atoms)  # type: ignore[no-untyped-call]

        output_directory = output_directory or Path.cwd()
        output_directory.mkdir(parents=True, exist_ok=True)
        relaxer = BaseAtomisticRunner._optimizer(
            optimizer_type=optimizer_settings.optimizer_type,
            atoms_or_filter=unit_cell_filter or atoms,
            logfile=(output_directory / DEFAULT_RELAX_LOGFILE).as_posix(),
        )
        if store_trajectory:
            traj = Trajectory(
                output_directory / DEFAULT_RELAX_TRAJECTORY_FILE, "w", atoms
            )  # type: ignore[no-untyped-call]
            relaxer.attach(  # type: ignore[no-untyped-call]
                traj, interval=optimizer_settings.store_interval
            )
        return relaxer

    def _apply_constraints_to_atoms(
        self, atoms: ase.Atoms, constraints: RelaxationConstraints | None
    ) -> ase.Atoms:
        atoms = deepcopy(atoms)
        fixed_atom_indices = (
            [] if constraints is None else constraints.build_fixed_atom_indices(atoms)
        )
        if fixed_atom_indices:
            constraint = FixAtoms(indices=fixed_atom_indices)  # type: ignore[no-untyped-call]
            atoms.set_constraint(constraint=constraint)  # type: ignore[no-untyped-call]
        return atoms

    def _record_step(
        self, optimizer: Optimizer, step_properties: list[StepProperties]
    ) -> list[StepProperties]:
        step = optimizer.nsteps
        atoms_or_filter = optimizer.atoms
        atoms = self._get_atoms(atoms_or_filter)
        properties = self._get_output(step=step, atoms=atoms)
        step_properties.append(properties)
        return step_properties

    def _get_output(self, step: int, atoms: ase.Atoms) -> StepProperties:
        atoms.calc = self._calculator
        try:
            energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            forces = atoms.get_forces().tolist()  # type: ignore[no-untyped-call]
            stress = atoms.get_stress().tolist()  # type: ignore[no-untyped-call]
            charges = (
                atoms.get_charges()  # type: ignore[no-untyped-call]
                if "charges" in atoms.calc.implemented_properties
                else None
            )
            return StepProperties(
                step=step,
                structure=self._ase_atoms_adaptor.get_structure(atoms),
                forces=forces,
                stress=stress,
                energy=energy,
                charges=charges,
            )
        except Exception as e:
            raise RuntimeError(e) from e

    def _run_optimizer(
        self, optimizer: Optimizer, optimizer_settings: RelaxOptimizerSettings
    ) -> list[StepProperties]:
        relax_properties: list[StepProperties] = []
        optimizer.attach(  # type:ignore [no-untyped-call]
            partial(
                self._record_step, optimizer=optimizer, step_properties=relax_properties
            ),
            interval=optimizer_settings.store_interval,
        )
        optimizer.run(  # type:ignore [no-untyped-call]
            fmax=optimizer_settings.force_threshold_in_ev_per_angstrom,
            steps=optimizer_settings.max_steps,
        )
        return relax_properties

    def run_single_point(self) -> SinglePointProperties:
        """Run a single SCF and return energy, forces, stress, and (if available) charge."""
        return SinglePointProperties.model_validate(
            self._get_output(step=0, atoms=self._ase_atoms).model_dump(exclude={"type"})
        )

    def run_relax(  # noqa: D102
        self,
        optimizer_settings: RelaxOptimizerSettings,
        constraints: RelaxationConstraints | None = None,
        *,
        store_trajectory: bool = True,
        output_directory: Path | None = None,
    ) -> RelaxProperties:
        constrained_atoms = self._apply_constraints_to_atoms(
            atoms=self._ase_atoms, constraints=constraints
        )
        constrained_atoms.calc = self._calculator
        fix_cell = (
            constraints.fix_cell if constraints is not None else True
        )  # default value
        relaxer = self._initialize_optimizer(
            atoms=constrained_atoms,
            optimizer_settings=optimizer_settings,
            fix_cell=fix_cell,
            store_trajectory=store_trajectory,
            output_directory=output_directory,
        )
        relax_properties = self._run_optimizer(
            optimizer=relaxer, optimizer_settings=optimizer_settings
        )
        return RelaxProperties(relax_steps=relax_properties)
