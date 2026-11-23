"""Helpers for optional extxyz artifacts from structure-generation jobs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from ase.io import write
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.constants.structure_generators import STRUCTURE_GENERATION_EXTXYZ_FILENAME

logger = logging.getLogger(__name__)


def write_generated_structures_extxyz(
    *,
    structures: Sequence[PMG_Structure],
    run_directory: Path,
    failure_message: str = "Extxyz export failed.",
) -> Path | None:
    """Write the generated structures to a job-local extxyz file."""
    try:
        run_directory.mkdir(parents=True, exist_ok=True)
        extxyz_path = run_directory / STRUCTURE_GENERATION_EXTXYZ_FILENAME
        adaptor = AseAtomsAdaptor()
        ase_atoms = [adaptor.get_atoms(structure) for structure in structures]
        write(extxyz_path.as_posix(), ase_atoms, format="extxyz")
    except Exception:
        logger.exception(failure_message)
        return None
    else:
        return extxyz_path
