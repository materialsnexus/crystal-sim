"""Standalone GPAW relax script executed inside the external GPAW runtime."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.constraints import FixAtoms
from ase.filters import UnitCellFilter
from ase.io import read
from ase.optimize import BFGS, FIRE, FIRE2, LBFGS
from gpaw import GPAW
from gpaw.mpi import world

DEFAULT_TEXT_ENCODING = "utf-8"


class GpawExternalRelaxScriptInputKey(StrEnum):
    """Keys for the external relax input payload."""

    calculator_kwargs = auto()
    optimizer_type = auto()
    max_steps = auto()
    force_threshold_in_ev_per_angstrom = auto()
    store_interval = auto()
    fix_cell = auto()
    fixed_atom_indices = auto()


class GpawExternalRelaxScriptOutputKey(StrEnum):
    """Top-level keys for the external relax output payload."""

    relax_steps = auto()


class GpawExternalRelaxScriptStepKey(StrEnum):
    """Per-step keys for the external relax output payload."""

    step = auto()
    symbols = auto()
    positions = auto()
    cell = auto()
    pbc = auto()
    forces = auto()
    stress = auto()
    energy = auto()


GPAW_EXTERNAL_RELAX_SCRIPT_ARGUMENT_COUNT = 4
GPAW_EXTERNAL_RELAX_SCRIPT_USAGE = (
    "Usage: gpaw_external_relax_script.py "
    "<input_structure_path> <kwargs_json_path> <output_json_path>"
)
GPAW_EXTERNAL_RELAX_OPTIMIZER_BFGS = "bfgs"
GPAW_EXTERNAL_RELAX_OPTIMIZER_LBFGS = "lbfgs"
GPAW_EXTERNAL_RELAX_OPTIMIZER_FIRE = "fire"
GPAW_EXTERNAL_RELAX_OPTIMIZER_FIRE2 = "fire2"


def parse_cli_arguments(argv: list[str]) -> tuple[str, str, str]:
    """Extract CLI paths from argv."""
    if len(argv) != GPAW_EXTERNAL_RELAX_SCRIPT_ARGUMENT_COUNT:
        raise ValueError(GPAW_EXTERNAL_RELAX_SCRIPT_USAGE)
    return argv[1], argv[2], argv[3]


def load_atoms(
    input_structure_path: Path | str, *, read_structure: Callable[..., object]
) -> Atoms:
    """Read and type-check the input Atoms object."""
    atoms_object = read_structure(input_structure_path, index=0)
    if not isinstance(atoms_object, Atoms):
        raise TypeError
    return atoms_object


def load_script_input_payload(kwargs_json_path: Path | str) -> dict[str, object]:
    """Read and type-check script JSON input payload."""
    with Path(kwargs_json_path).open(encoding=DEFAULT_TEXT_ENCODING) as file_handle:
        script_input_payload_object = json.load(file_handle)
    if not isinstance(script_input_payload_object, dict):
        raise TypeError
    return script_input_payload_object


def _coerce_to_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise TypeError


def _coerce_to_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError


def _coerce_to_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError
    return value


def _build_relax_step_payload(*, step: int, atoms: Atoms) -> dict[str, object]:
    return {
        GpawExternalRelaxScriptStepKey.step: step,
        GpawExternalRelaxScriptStepKey.symbols: [
            str(symbol)
            for symbol in atoms.get_chemical_symbols()  # type: ignore[no-untyped-call]
        ],
        GpawExternalRelaxScriptStepKey.positions: atoms.positions.tolist(),
        GpawExternalRelaxScriptStepKey.cell: atoms.cell.array.tolist(),
        GpawExternalRelaxScriptStepKey.pbc: [bool(pbc_flag) for pbc_flag in atoms.pbc],
        GpawExternalRelaxScriptStepKey.forces: atoms.get_forces().tolist(),  # type: ignore[no-untyped-call]
        GpawExternalRelaxScriptStepKey.stress: atoms.get_stress().tolist(),  # type: ignore[no-untyped-call]
        GpawExternalRelaxScriptStepKey.energy: float(
            atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        ),
    }


def run_relaxation_from_payload(
    *,
    atoms: Atoms,
    script_input_payload: dict[str, object],
    calculator_factory: Callable[..., Any],
    optimizer_classes: dict[str, Callable[..., Any]],
    unit_cell_filter_factory: Callable[[Atoms], object],
) -> list[dict[str, object]]:
    """Run relaxation from already-loaded payload and return serialized steps."""
    gpaw_kwargs_object = script_input_payload[
        GpawExternalRelaxScriptInputKey.calculator_kwargs
    ]
    if not isinstance(gpaw_kwargs_object, dict):
        raise TypeError
    gpaw_kwargs: dict[str, Any] = gpaw_kwargs_object
    optimizer_type = str(
        script_input_payload[GpawExternalRelaxScriptInputKey.optimizer_type]
    ).strip()
    max_steps = _coerce_to_int(
        script_input_payload[GpawExternalRelaxScriptInputKey.max_steps]
    )
    force_threshold_in_ev_per_angstrom = _coerce_to_float(
        script_input_payload[
            GpawExternalRelaxScriptInputKey.force_threshold_in_ev_per_angstrom
        ]
    )
    store_interval = _coerce_to_int(
        script_input_payload[GpawExternalRelaxScriptInputKey.store_interval]
    )
    if store_interval < 1:
        raise ValueError
    fix_cell = bool(script_input_payload[GpawExternalRelaxScriptInputKey.fix_cell])
    fixed_atom_indices_values = _coerce_to_list(
        script_input_payload[GpawExternalRelaxScriptInputKey.fixed_atom_indices]
    )
    fixed_atom_indices = [_coerce_to_int(index) for index in fixed_atom_indices_values]
    if fixed_atom_indices:
        atoms.set_constraint(FixAtoms(indices=fixed_atom_indices))  # type: ignore[no-untyped-call]
    atoms_or_filter = atoms if fix_cell else unit_cell_filter_factory(atoms)
    optimizer_class = optimizer_classes.get(optimizer_type)
    if optimizer_class is None:
        raise ValueError(optimizer_type)
    calculator: Any = calculator_factory(**gpaw_kwargs)
    atoms.calc = calculator
    optimizer = optimizer_class(atoms_or_filter, logfile=None)
    relax_steps: list[dict[str, object]] = []

    def _record_relax_step() -> None:
        current_atoms = (
            atoms_or_filter.atoms
            if hasattr(atoms_or_filter, "atoms")
            else atoms_or_filter
        )
        if not isinstance(current_atoms, Atoms):
            raise TypeError
        relax_steps.append(
            _build_relax_step_payload(step=int(optimizer.nsteps), atoms=current_atoms)
        )

    optimizer.attach(_record_relax_step, interval=store_interval)
    optimizer.run(fmax=force_threshold_in_ev_per_angstrom, steps=max_steps)
    last_recorded_step = (
        _coerce_to_int(relax_steps[-1][GpawExternalRelaxScriptStepKey.step])
        if relax_steps
        else None
    )
    if not relax_steps or last_recorded_step != optimizer.nsteps:
        _record_relax_step()
    return relax_steps


def write_relax_output_payload(
    *, output_json_path: Path | str, relax_steps: list[dict[str, object]], mpi_rank: int
) -> None:
    """Write relax output payload from MPI rank zero only."""
    if mpi_rank != 0:
        return
    with Path(output_json_path).open(
        "w", encoding=DEFAULT_TEXT_ENCODING
    ) as file_handle:
        json.dump(
            {GpawExternalRelaxScriptOutputKey.relax_steps: relax_steps}, file_handle
        )


def _build_optimizer_type_to_class() -> dict[str, Callable[..., Any]]:
    return {
        GPAW_EXTERNAL_RELAX_OPTIMIZER_BFGS: BFGS,
        GPAW_EXTERNAL_RELAX_OPTIMIZER_LBFGS: LBFGS,
        GPAW_EXTERNAL_RELAX_OPTIMIZER_FIRE: FIRE,
        GPAW_EXTERNAL_RELAX_OPTIMIZER_FIRE2: FIRE2,
    }


def main(argv: list[str]) -> None:
    """Run external GPAW relax script from CLI-style argv."""
    input_structure_path, kwargs_json_path, output_json_path = parse_cli_arguments(argv)
    atoms = load_atoms(input_structure_path, read_structure=read)
    script_input_payload = load_script_input_payload(kwargs_json_path)
    relax_steps = run_relaxation_from_payload(
        atoms=atoms,
        script_input_payload=script_input_payload,
        calculator_factory=GPAW,
        optimizer_classes=_build_optimizer_type_to_class(),
        unit_cell_filter_factory=UnitCellFilter,
    )
    write_relax_output_payload(
        output_json_path=output_json_path,
        relax_steps=relax_steps,
        mpi_rank=int(world.rank),
    )
    world.barrier()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main(sys.argv)
