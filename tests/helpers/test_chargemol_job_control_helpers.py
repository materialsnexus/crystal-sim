"""Test helpers for Chargemol job_control rendering."""

from __future__ import annotations

from pathlib import Path

from crystal.schemas.chargemol import (
    CHARGEMOL_JOB_CONTROL_TAG_ORDER,
    ChargemolJobControlInputBuilder,
    ChargemolJobControlTag,
)


def build_expected_chargemol_job_control_lines(
    input_cube_path: Path,
    atomic_densities_directory: Path,
    core_electrons_by_atomic_number: dict[int, int],
) -> list[str]:
    """Return the canonical job_control lines for Chargemol DDEC6 cube inputs."""
    core_electron_lines = [
        f"{atomic_number} {core_electrons}"
        for atomic_number, core_electrons in sorted(
            core_electrons_by_atomic_number.items()
        )
    ]
    tag_values: dict[ChargemolJobControlTag, list[str]] = {
        ChargemolJobControlTag.net_charge: ["0.0"],
        ChargemolJobControlTag.periodicity_vectors: [
            ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                fortran_bool_value=True
            ),
            ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                fortran_bool_value=True
            ),
            ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                fortran_bool_value=False
            ),
        ],
        ChargemolJobControlTag.atomic_densities_directory: [
            ChargemolJobControlInputBuilder.format_chargemol_atomic_densities_directory(
                atomic_densities_directory
            )
        ],
        ChargemolJobControlTag.input_filename: [str(input_cube_path)],
        ChargemolJobControlTag.charge_type: ["ddec6"],
        ChargemolJobControlTag.compute_bond_orders: [
            ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                fortran_bool_value=True
            )
        ],
        ChargemolJobControlTag.print_atomic_densities: [
            ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                fortran_bool_value=False
            )
        ],
        ChargemolJobControlTag.number_of_core_electrons: core_electron_lines,
    }
    job_control_lines: list[str] = []
    for tag in CHARGEMOL_JOB_CONTROL_TAG_ORDER:
        values = tag_values.get(tag)
        if values is None:
            continue
        job_control_lines.extend(
            ChargemolJobControlInputBuilder.build_chargemol_tag_block_lines(
                tag=tag, values=values
            )
        )
    return job_control_lines
