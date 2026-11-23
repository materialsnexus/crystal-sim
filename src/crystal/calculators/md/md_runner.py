"""MLIP-backed molecular dynamics runner utilities."""

from __future__ import annotations

import logging
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import ase
import numpy as np
import torch
from ase import units
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import MaskedMTKNPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet

from crystal.schemas.md.properties import MdStageResult
from crystal.schemas.md.utils import read_last_trajectory_frame
from crystal.schemas.mlip import MATGL_MISSING_ERROR_MESSAGE

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
from pydantic import model_validator
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.helpers.file import count_args_not_none
from crystal.helpers.md_run_helpers import (
    get_md_step_count,
    load_atoms,
    pressure_gpa_to_ase_atomic_units,
    user_defined_md_logger,
)
from crystal.helpers.mlip_helpers import prepare_atoms_for_matgl
from crystal.schemas.md.settings import MdEnsemble, MdRunSettings
from crystal.schemas.serialization import MSONableBaseModel

logger = logging.getLogger(__name__)


torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")  # type:ignore[no-untyped-call]
DEFAULT_FLOATING_POINT_TOLERANCE = 1e-6


class MlipMdRunner(MSONableBaseModel):  # noqa: D101
    run_settings: MdRunSettings
    structure: PMG_Structure | None = None
    trajectory_path_for_restart: Path | None = None

    @model_validator(mode="after")
    def _validate_structure_inputs(self) -> Self:
        if count_args_not_none(self.structure, self.trajectory_path_for_restart) != 1:
            error_msg = "Only one of structure or trajectory path must be provided"
            raise ValueError(error_msg)
        return self

    @property
    def _ase_atoms(self) -> ase.Atoms:
        if self.trajectory_path_for_restart is not None:
            return read_last_trajectory_frame(self.trajectory_path_for_restart)
        assert self.structure is not None
        return load_atoms(structure_or_path=self.structure)

    @cached_property
    def _calculator(self) -> PESCalculator:
        try:
            import matgl  # noqa: PLC0415
            from matgl.ext.ase import PESCalculator  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(MATGL_MISSING_ERROR_MESSAGE) from exc

        potential = matgl.load_model(self.run_settings.potential.model.model_path)
        device = torch.get_default_device()
        potential.to(device=device)
        return PESCalculator(
            potential=potential,
            stress_unit=self.run_settings.potential.stress_unit,  # type:ignore[arg-type]
        )

    @cached_property
    def output_directory(self) -> Path:  # noqa: D102
        output_directory = self.run_settings.output_directory or Path.cwd()
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    @property
    def _timestep(self) -> float:
        return self.run_settings.time.time_step_fs * units.fs

    @property
    def _step_count(self) -> int:
        return get_md_step_count(
            simulation_time_ps=self.run_settings.time.simulation_time_ps,
            time_step_fs=self.run_settings.time.time_step_fs,
        )

    def _get_stage_result(self, *, ensemble: MdEnsemble) -> MdStageResult:
        return MdStageResult(
            ensemble=ensemble,
            trajectory_path=self._trajectory_path(ensemble=ensemble),
            default_log_path=self._default_property_log_path(ensemble=ensemble),
            user_defined_log_path=self._user_defined_property_log_path(
                ensemble=ensemble
            ),
        )

    def _trajectory_path(self, *, ensemble: MdEnsemble) -> Path:
        return self.output_directory / f"{ensemble.value}.traj"

    def _default_property_log_path(self, *, ensemble: MdEnsemble) -> Path:
        return self.output_directory / f"{ensemble.value}_default.log"

    def _user_defined_property_log_path(self, *, ensemble: MdEnsemble) -> Path:
        return self.output_directory / f"{ensemble.value}_user_defined.log"

    def _attach_calculator(self, atoms: ase.Atoms) -> None:
        atoms.calc = self._calculator

    def _apply_zero_rotation_constraint(self, atoms: ase.Atoms) -> None:
        principal_moments = np.asarray(atoms.get_moments_of_inertia())  # type:ignore[no-untyped-call]
        if np.all(principal_moments > DEFAULT_FLOATING_POINT_TOLERANCE):
            ZeroRotation(atoms)
            return
        logger.warning(
            "Skipping ZeroRotation because the structure has degenerate "
            "principal moments of inertia."
        )

    def _prepare_atoms(self, *, initialize_temperature: bool) -> ase.Atoms:
        atoms = deepcopy(self._ase_atoms)
        atoms.pbc = self.run_settings.pbc
        atoms = prepare_atoms_for_matgl(atoms)
        self._attach_calculator(atoms=atoms)
        if initialize_temperature and self.trajectory_path_for_restart is None:
            thermostat = self.run_settings.dynamics.thermostat
            assert thermostat is not None
            MaxwellBoltzmannDistribution(atoms, temperature_K=thermostat.temperature_k)
        Stationary(atoms)
        self._apply_zero_rotation_constraint(atoms)
        return atoms

    def _attach_default_logger(
        self, dynamics: Any, atoms: ase.Atoms, log_path: Path
    ) -> None:
        default_md_logger = MDLogger(
            dyn=dynamics,
            atoms=atoms,
            logfile=str(log_path),
            header=True,
            stress=True,
            peratom=False,
            mode="a",
        )
        dynamics.attach(
            default_md_logger, interval=self.run_settings.time.property_log_interval
        )

    def _attach_loggers(
        self, dynamics: Any, atoms: ase.Atoms, ensemble: MdEnsemble
    ) -> None:
        self._attach_default_logger(
            dynamics=dynamics,
            atoms=atoms,
            log_path=self._default_property_log_path(ensemble=ensemble),
        )
        dynamics.attach(
            user_defined_md_logger(
                dynamics=dynamics,
                atoms=atoms,
                log_path=self._user_defined_property_log_path(ensemble=ensemble),
            ),
            interval=self.run_settings.time.property_log_interval,
        )

    def run_nve(self) -> MdStageResult:
        """Run microcanonical MD."""
        ensemble = MdEnsemble.nve
        atoms = self._prepare_atoms(initialize_temperature=False)
        trajectory_path = self._trajectory_path(ensemble=ensemble)
        dynamics = VelocityVerlet(
            atoms,
            timestep=self._timestep,
            trajectory=str(trajectory_path),
            append_trajectory=True,
            logfile=None,
            loginterval=self.run_settings.time.trajectory_interval,
        )
        self._attach_loggers(dynamics, atoms, ensemble)
        dynamics.run(self._step_count)  # type: ignore[no-untyped-call]
        return self._get_stage_result(ensemble=ensemble)

    def run_nvt(self) -> MdStageResult:
        """Run Langevin thermostatted MD."""
        ensemble = MdEnsemble.nvt
        atoms = self._prepare_atoms(initialize_temperature=True)
        trajectory_path = self._trajectory_path(ensemble=ensemble)
        thermostat = self.run_settings.dynamics.thermostat
        assert thermostat is not None
        dynamics = Langevin(
            atoms=atoms,
            timestep=self._timestep,
            temperature_K=thermostat.temperature_k,
            friction=1.0 / (thermostat.time_constant_fs * units.fs),
            trajectory=str(trajectory_path),
            append_trajectory=True,
            logfile=None,
            loginterval=self.run_settings.time.trajectory_interval,
        )
        self._attach_loggers(dynamics, atoms, ensemble)
        dynamics.run(self._step_count)  # type: ignore[no-untyped-call]
        return self._get_stage_result(ensemble=ensemble)

    def run_npt(self) -> MdStageResult:
        """Run MTK NPT MD."""
        ensemble = MdEnsemble.npt
        atoms = self._prepare_atoms(initialize_temperature=True)
        trajectory_path = self._trajectory_path(ensemble=ensemble)
        thermostat = self.run_settings.dynamics.thermostat
        barostat = self.run_settings.dynamics.barostat
        assert thermostat is not None
        assert barostat is not None
        dynamics = MaskedMTKNPT(
            atoms=atoms,
            timestep=self._timestep,
            temperature_K=thermostat.temperature_k,
            pressure_au=pressure_gpa_to_ase_atomic_units(barostat.pressure_gpa),
            tdamp=barostat.temperature_damping_fs * units.fs,
            pdamp=barostat.pressure_damping_fs * units.fs,
            trajectory=str(trajectory_path),
            append_trajectory=True,
            logfile=None,
            loginterval=self.run_settings.time.trajectory_interval,
            # NOTE: The ASE api suggests that the mask is a tuple of bools, but it requires a numpy array of bools
            mask=np.asarray(self.run_settings.pressure_mask, dtype=bool),  # type: ignore[arg-type]
        )
        self._attach_loggers(dynamics, atoms, ensemble)
        dynamics.run(self._step_count)  # type: ignore[no-untyped-call]
        return self._get_stage_result(ensemble=ensemble)

    def run(self) -> MdStageResult:  # noqa: D102
        match self.run_settings.dynamics.ensemble:
            case MdEnsemble.nve:
                return self.run_nve()
            case MdEnsemble.nvt:
                return self.run_nvt()
            case MdEnsemble.npt:
                return self.run_npt()
