"""Standalone GPAW isolated-atom script executed inside the external runtime."""

from __future__ import annotations

import json
import sys
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import gpaw
from ase import Atoms
from ase.io import read
from gpaw import GPAW
from gpaw.mpi import world
from gpaw_runtime_provenance import build_gpaw_runtime_provenance_payload

DEFAULT_TEXT_ENCODING = "utf-8"
GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_ARGUMENT_COUNT = 4
GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_USAGE = (
    "Usage: gpaw_external_isolated_atom_script.py "
    "<input_structure_path> <kwargs_json_path> <output_json_path>"
)
GPAW_EXTERNAL_ISOLATED_ATOM_STRUCTURE_ATOM_COUNT = 1
GPAW_EXTERNAL_ISOLATED_ATOM_STRUCTURE_ATOM_COUNT_MESSAGE = (
    "The isolated-atom GPAW script requires exactly one atom."
)
GPAW_ISOLATED_ATOM_PRIMARY_EIGENSOLVER_NAME = "etdm-fdpw"
GPAW_ISOLATED_ATOM_FALLBACK_EIGENSOLVER_NAME = "ppcg"
GPAW_ISOLATED_ATOM_EMPTY_OCCUPIED_BANDS_ERROR_MESSAGE = "max() arg is an empty sequence"


class GpawExternalIsolatedAtomScriptInputKey(StrEnum):
    """Keys for the external isolated-atom input payload."""

    calculator_kwargs = auto()


class GpawExternalIsolatedAtomScriptOutputKey(StrEnum):
    """Keys for the external isolated-atom output payload."""

    energy = auto()
    magnetic_moment = auto()
    eigensolver = auto()
    runtime_provenance = auto()


def parse_cli_arguments(argv: list[str]) -> tuple[str, str, str]:
    """Extract CLI paths from argv."""
    if len(argv) != GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_ARGUMENT_COUNT:
        raise ValueError(GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_USAGE)
    return argv[1], argv[2], argv[3]


def load_isolated_atom(input_structure_path: Path | str) -> Atoms:
    """Read and validate the prebuilt one-atom input structure."""
    atoms_object = read(input_structure_path, index=0)
    if not isinstance(atoms_object, Atoms):
        raise TypeError
    if len(atoms_object) != GPAW_EXTERNAL_ISOLATED_ATOM_STRUCTURE_ATOM_COUNT:
        raise ValueError(GPAW_EXTERNAL_ISOLATED_ATOM_STRUCTURE_ATOM_COUNT_MESSAGE)
    return atoms_object


def load_script_input_payload(kwargs_json_path: Path | str) -> dict[str, object]:
    """Read and type-check the external script JSON input payload."""
    with Path(kwargs_json_path).open(encoding=DEFAULT_TEXT_ENCODING) as file_handle:
        script_input_payload_object = json.load(file_handle)
    if not isinstance(script_input_payload_object, dict):
        raise TypeError
    return script_input_payload_object


def run_isolated_atom_from_payload(
    *, atoms: Atoms, script_input_payload: dict[str, object], gpaw_version: str
) -> dict[str, object]:
    """Calculate energy, magnetic moment, and resolved PAW provenance."""
    calculator_kwargs_object = script_input_payload[
        GpawExternalIsolatedAtomScriptInputKey.calculator_kwargs
    ]
    if not isinstance(calculator_kwargs_object, dict):
        raise TypeError
    calculator_kwargs: dict[str, Any] = calculator_kwargs_object
    calculator: Any = GPAW(**calculator_kwargs)
    selected_eigensolver_name = _selected_eigensolver_name(calculator_kwargs)
    atoms.calc = calculator
    try:
        energy, magnetic_moment = _calculate_energy_and_magnetic_moment(atoms)
    except ValueError as primary_error:
        if str(primary_error) != GPAW_ISOLATED_ATOM_EMPTY_OCCUPIED_BANDS_ERROR_MESSAGE:
            raise
        if selected_eigensolver_name == GPAW_ISOLATED_ATOM_FALLBACK_EIGENSOLVER_NAME:
            raise
        fallback_eigensolver_name = GPAW_ISOLATED_ATOM_FALLBACK_EIGENSOLVER_NAME
        fallback_calculator_kwargs = {
            **calculator_kwargs,
            "eigensolver": {"name": fallback_eigensolver_name},
        }
        calculator = GPAW(**fallback_calculator_kwargs)
        selected_eigensolver_name = fallback_eigensolver_name
        atoms.calc = calculator
        energy, magnetic_moment = _calculate_energy_and_magnetic_moment(atoms)
    return {
        GpawExternalIsolatedAtomScriptOutputKey.energy: energy,
        GpawExternalIsolatedAtomScriptOutputKey.magnetic_moment: magnetic_moment,
        GpawExternalIsolatedAtomScriptOutputKey.eigensolver: selected_eigensolver_name,
        GpawExternalIsolatedAtomScriptOutputKey.runtime_provenance: (
            build_gpaw_runtime_provenance_payload(
                gpaw_version=gpaw_version, setups=calculator.setups
            )
        ),
    }


def _calculate_energy_and_magnetic_moment(atoms: Atoms) -> tuple[float, float]:
    """Evaluate the isolated atom with its currently attached calculator."""
    return (
        float(atoms.get_potential_energy()),  # type: ignore[no-untyped-call]
        float(atoms.get_magnetic_moment()),  # type: ignore[no-untyped-call]
    )


def _selected_eigensolver_name(calculator_kwargs: dict[str, Any]) -> str:
    """Return the solver represented by the primary calculator kwargs."""
    eigensolver = calculator_kwargs.get("eigensolver")
    if isinstance(eigensolver, dict):
        name = eigensolver.get("name")
        if isinstance(name, str):
            return name
    return GPAW_ISOLATED_ATOM_PRIMARY_EIGENSOLVER_NAME


def write_isolated_atom_output_payload(
    *, output_json_path: Path | str, output_payload: dict[str, object], mpi_rank: int
) -> None:
    """Write the isolated-atom payload from MPI rank zero only."""
    if mpi_rank != 0:
        return
    with Path(output_json_path).open(
        "w", encoding=DEFAULT_TEXT_ENCODING
    ) as file_handle:
        json.dump(output_payload, file_handle)


def main(argv: list[str]) -> None:
    """Run the external isolated-atom GPAW calculation from CLI-style argv."""
    input_structure_path, kwargs_json_path, output_json_path = parse_cli_arguments(argv)
    atoms = load_isolated_atom(input_structure_path)
    script_input_payload = load_script_input_payload(kwargs_json_path)
    output_payload = run_isolated_atom_from_payload(
        atoms=atoms,
        script_input_payload=script_input_payload,
        gpaw_version=gpaw.__version__,
    )
    write_isolated_atom_output_payload(
        output_json_path=output_json_path,
        output_payload=output_payload,
        mpi_rank=int(world.rank),
    )
    world.barrier()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main(sys.argv)
