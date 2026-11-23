"""Pure helper functions for MD stability workflows."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any

import ase
import numpy as np
from ase import units
from ase.io import read
from pydantic import PositiveFloat, validate_call
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.schemas.md.checks import MdEquilibrationCheck
from crystal.schemas.md.properties import MdCheckResult, MdStageResult
from crystal.schemas.md.utils import (
    DEFAULT_USER_DEFINED_HEADERS,
    store_trajectory_site_properties,
)

AMU_TO_GRAMS = 1.66053906660e-24
ANGSTROM3_TO_CM3 = 1.0e-24


def load_ase_atoms_from_pmg_structure(structure: PMG_Structure) -> ase.Atoms:  # noqa: D103
    atoms = AseAtomsAdaptor.get_atoms(structure)
    store_trajectory_site_properties(atoms=atoms)
    return atoms


def load_ase_atoms_from_path(path: Path) -> ase.Atoms:
    """Load a single ASE atoms object from a structure file path."""
    loaded_atoms = read(path)
    if not isinstance(loaded_atoms, ase.Atoms):
        error_msg = f"Expected a single ASE Atoms object from {path}."
        raise TypeError(error_msg)
    return loaded_atoms


def load_atoms(structure_or_path: ase.Atoms | PMG_Structure | Path) -> ase.Atoms:  # noqa: D103
    match structure_or_path:
        case ase.Atoms():
            return deepcopy(structure_or_path)
        case PMG_Structure():
            return load_ase_atoms_from_pmg_structure(structure=structure_or_path)
        case Path():
            return load_ase_atoms_from_path(path=structure_or_path)
        case _:
            raise NotImplementedError


@validate_call
def get_md_step_count(
    simulation_time_ps: PositiveFloat, time_step_fs: PositiveFloat
) -> int:
    """Convert simulation time in ps and time step in fs to an MD step count."""
    return ceil((simulation_time_ps * 1000.0) / time_step_fs)


def compute_slab_volumetric_density_g_cm3(atoms: ase.Atoms) -> float:  # noqa: D103
    volume_angstrom3: float = atoms.get_volume()  # type:ignore[no-untyped-call]
    total_mass_amu = float(np.sum(atoms.get_masses()))
    total_mass_g = total_mass_amu * AMU_TO_GRAMS
    volume_cm3 = volume_angstrom3 * ANGSTROM3_TO_CM3
    return float(total_mass_g / volume_cm3)


def pressure_gpa_to_ase_atomic_units(pressure_gpa: float) -> float:  # noqa: D103
    return pressure_gpa * units.GPa


def user_defined_md_logger(
    dynamics: Any, atoms: ase.Atoms, log_path: Path
) -> Callable[[], None]:
    """Returns a lightweight custom MD logger that records time and max force."""

    def _write_log_row() -> None:
        write_header = not log_path.exists()
        # mode is set to "a" to avoid overwriting the file if it exists
        with log_path.open(mode="a", encoding="utf-8") as file_handle:
            if write_header:
                header = ", ".join(DEFAULT_USER_DEFINED_HEADERS) + "\n"
                file_handle.write(header)
            time_ps = (
                float(dynamics.get_time()) / (1000.0 * units.fs)
                if hasattr(dynamics, "get_time")
                else 0.0
            )
            density = compute_slab_volumetric_density_g_cm3(atoms=atoms)
            atoms_any: Any = atoms
            max_force_component = float(np.abs(atoms_any.get_forces()).max())
            file_handle.write(
                f"{time_ps:.6f}, {density:.6f}, {max_force_component:.6f}\n"
            )

    return _write_log_row


def run_equilibration_check(  # noqa: D103
    *,
    md_stage_result: MdStageResult,
    time_averaging_window: float,
    check: MdEquilibrationCheck,
) -> MdCheckResult:
    comparison_value = check.get_comparison_value(
        md_stage_result, time_averaging_window
    )
    passed = check.threshold >= comparison_value
    return MdCheckResult(
        check_type=check.check_type,
        passed=passed,
        threshold=check.threshold,
        observed_value=comparison_value,
    )
