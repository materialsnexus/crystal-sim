from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from jobflow.managers.local import run_locally

from crystal.calculators.gpaw import slurm_runtime_precheck


def test_external_gpaw_mpi_precheck_rejects_missing_executables(tmp_path: Path) -> None:
    precheck_job = slurm_runtime_precheck.run_external_gpaw_mpi_precheck(
        number_of_processes=1
    )

    with (
        patch.object(slurm_runtime_precheck, "which", return_value=None),
        pytest.raises(
            FileNotFoundError,
            match=slurm_runtime_precheck.GPAW_SLURM_MISSING_EXECUTABLES_MESSAGE,
        ),
    ):
        run_locally(precheck_job, root_dir=tmp_path, raise_immediately=True)


def test_external_gpaw_mpi_precheck_rejects_nonzero_mpi_return_code(
    tmp_path: Path,
) -> None:
    precheck_job = slurm_runtime_precheck.run_external_gpaw_mpi_precheck(
        number_of_processes=1
    )
    completed_process = subprocess.CompletedProcess(
        args=["mpirun"], returncode=1, stdout="", stderr="MPI failed"
    )

    with (
        patch.object(
            slurm_runtime_precheck,
            "which",
            side_effect=("/usr/bin/mpirun", "/usr/bin/gpaw"),
        ),
        patch(
            "crystal.calculators.gpaw.slurm_runtime_precheck.subprocess.run",
            return_value=completed_process,
        ),
        pytest.raises(RuntimeError, match="MPI failed"),
    ):
        run_locally(precheck_job, root_dir=tmp_path, raise_immediately=True)
