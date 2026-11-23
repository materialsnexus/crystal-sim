"""Standalone GPAW script executed inside the external GPAW runtime.

Typing in this script is intentionally lower, it runs in an external
environment (separate GPAW env).
"""

from __future__ import annotations

import json
import sys
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import gpaw
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.units import Bohr
from gpaw import GPAW
from gpaw.mpi import world
from gpaw_runtime_provenance import build_gpaw_runtime_provenance_payload

DEFAULT_TEXT_ENCODING = "utf-8"


class GpawExternalScfScriptInputKey(StrEnum):
    """Keys for the external SCF input payload."""

    calculator_kwargs = auto()
    density_gridrefinement = auto()
    total_density_cube_filename = auto()


class GpawExternalScfScriptOutputKey(StrEnum):
    """Keys for the external SCF output payload."""

    energy = auto()
    forces = auto()
    stress = auto()
    pbc = auto()
    runtime_provenance = auto()


GPAW_EXTERNAL_SCF_SCRIPT_ARGUMENT_COUNT = 4
GPAW_EXTERNAL_SCF_SCRIPT_USAGE = (
    "Usage: gpaw_external_scf_script.py "
    "<input_structure_path> <kwargs_json_path> <output_json_path>"
)

if len(sys.argv) != GPAW_EXTERNAL_SCF_SCRIPT_ARGUMENT_COUNT:
    raise ValueError(GPAW_EXTERNAL_SCF_SCRIPT_USAGE)

input_structure_path = sys.argv[1]
kwargs_json_path = sys.argv[2]
output_json_path = sys.argv[3]


atoms_object = read(input_structure_path, index=0)
if not isinstance(atoms_object, Atoms):
    raise TypeError
atoms: Atoms = atoms_object
with Path(kwargs_json_path).open(encoding=DEFAULT_TEXT_ENCODING) as file_handle:
    script_input_payload = json.load(file_handle)
if not isinstance(script_input_payload, dict):
    raise TypeError
gpaw_kwargs = script_input_payload[GpawExternalScfScriptInputKey.calculator_kwargs]
if not isinstance(gpaw_kwargs, dict):
    raise TypeError
density_gridrefinement = int(
    script_input_payload[GpawExternalScfScriptInputKey.density_gridrefinement]
)
total_density_cube_filename = str(
    script_input_payload[GpawExternalScfScriptInputKey.total_density_cube_filename]
)

calculator: Any = GPAW(**gpaw_kwargs)
atoms.calc = calculator
energy = float(atoms.get_potential_energy())  # type: ignore[no-untyped-call]
forces = atoms.get_forces().tolist()  # type: ignore[no-untyped-call]
stress = atoms.get_stress().tolist()  # type: ignore[no-untyped-call]
density = np.asarray(
    calculator.get_all_electron_density(
        gridrefinement=density_gridrefinement, skip_core=False
    ),
    dtype=float,
)
write(total_density_cube_filename, atoms, data=density * Bohr**3)

runtime_provenance = build_gpaw_runtime_provenance_payload(
    gpaw_version=gpaw.__version__, setups=calculator.setups
)

if world.rank == 0:
    with Path(output_json_path).open(
        "w", encoding=DEFAULT_TEXT_ENCODING
    ) as file_handle:
        json.dump(
            {
                GpawExternalScfScriptOutputKey.energy: energy,
                GpawExternalScfScriptOutputKey.forces: forces,
                GpawExternalScfScriptOutputKey.stress: stress,
                GpawExternalScfScriptOutputKey.pbc: [
                    bool(pbc_flag) for pbc_flag in atoms.pbc
                ],
                GpawExternalScfScriptOutputKey.runtime_provenance: runtime_provenance,
            },
            file_handle,
        )

world.barrier()  # type: ignore[no-untyped-call]
