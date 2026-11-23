"""Preflight job for validating an external GPAW runtime inside Slurm."""

from __future__ import annotations

import os
import subprocess
from shutil import which

from jobflow.core.job import job

GPAW_SLURM_JOB_ID_ENVIRONMENT_VARIABLE = "SLURM_JOB_ID"
GPAW_SLURM_MISSING_EXECUTABLES_MESSAGE = (
    "The Slurm precheck could not find the configured MPI/GPAW executables."
)
GPAW_SLURM_PRECHECK_COMMAND = (
    "import gpaw; from gpaw.mpi import world; print(f'{gpaw.__version__}:{world.size}')"
)


@job
def run_external_gpaw_mpi_precheck(number_of_processes: int) -> dict[str, str]:
    """Run a one-command external GPAW MPI precheck on a Slurm compute node."""
    mpi_launcher_path = which("mpirun")
    gpaw_executable_path = which("gpaw")
    if mpi_launcher_path is None or gpaw_executable_path is None:
        raise FileNotFoundError(GPAW_SLURM_MISSING_EXECUTABLES_MESSAGE)
    completed_process = subprocess.run(
        [
            mpi_launcher_path,
            "-np",
            str(number_of_processes),
            gpaw_executable_path,
            "python",
            "-c",
            GPAW_SLURM_PRECHECK_COMMAND,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode != 0:
        raise RuntimeError(completed_process.stderr)
    return {
        "gpaw_runtime": completed_process.stdout.strip(),
        "slurm_job_id": os.environ[GPAW_SLURM_JOB_ID_ENVIRONMENT_VARIABLE],
    }
