"""Schemas for optional scratch execution of calculations."""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path

from pydantic import ConfigDict

from crystal.schemas.serialization import MSONableBaseModel

SCRATCH_EXECUTION_MANIFEST_FILENAME = "scratch_execution_manifest.json"
SLURM_TMPDIR_ENVIRONMENT_VARIABLE = "SLURM_TMPDIR"
TMPDIR_ENVIRONMENT_VARIABLE = "TMPDIR"
SLURM_JOB_ID_ENVIRONMENT_VARIABLE = "SLURM_JOB_ID"
SLURM_NODELIST_ENVIRONMENT_VARIABLE = "SLURM_NODELIST"
SCRATCH_PARENT_ENVIRONMENT_VARIABLES = (
    SLURM_TMPDIR_ENVIRONMENT_VARIABLE,
    TMPDIR_ENVIRONMENT_VARIABLE,
)


class ScratchExecutionStatus(StrEnum):
    """Status of a scratch execution run."""

    success = auto()
    failed = auto()


class LocalScratchPolicy(MSONableBaseModel):
    """Policy controlling optional execution in node-local scratch."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fallback_scratch_parent: Path | None = None


class ScratchExecutionManifest(MSONableBaseModel):
    """Manifest describing a scratch execution run."""

    final_directory: Path
    scratch_directory: Path
    hostname: str
    slurm_job_id: str | None = None
    slurm_node_name: str | None = None
    status: ScratchExecutionStatus
    error_message: str | None = None
