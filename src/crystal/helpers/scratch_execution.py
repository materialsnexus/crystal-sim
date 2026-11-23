"""Helpers for running calculations in node-local scratch."""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.schemas.scratch_execution import (
    SCRATCH_EXECUTION_MANIFEST_FILENAME,
    SCRATCH_PARENT_ENVIRONMENT_VARIABLES,
    SLURM_JOB_ID_ENVIRONMENT_VARIABLE,
    SLURM_NODELIST_ENVIRONMENT_VARIABLE,
    LocalScratchPolicy,
    ScratchExecutionManifest,
    ScratchExecutionStatus,
)
from crystal.schemas.serialization import MSONABLE_JSON_MODE


class ScratchDirectoryUnavailableError(RuntimeError):
    """Raised when scratch execution is enabled without an available scratch parent."""


@contextmanager
def local_scratch_execution_directory(
    *,
    policy: LocalScratchPolicy,
    destination_directory: Path,
    result_filenames: tuple[str, ...],
    text_result_filenames_replacing_execution_directory: tuple[str, ...] = (),
) -> Iterator[Path]:
    """Yield the directory where the calculation should run."""
    resolved_destination_directory = destination_directory.resolve()
    resolved_destination_directory.mkdir(parents=True, exist_ok=True)
    if not policy.enabled:
        yield resolved_destination_directory
        return

    scratch_run_directory = _create_scratch_directory(
        policy=policy, destination_directory=resolved_destination_directory
    )
    try:
        yield scratch_run_directory
    except Exception as exc:
        _handle_failed_scratch_run(
            destination_directory=resolved_destination_directory,
            scratch_run_directory=scratch_run_directory,
            error_message=str(exc),
            result_filenames=result_filenames,
            text_result_filenames_replacing_execution_directory=(
                text_result_filenames_replacing_execution_directory
            ),
        )
        raise

    _handle_successful_scratch_run(
        destination_directory=resolved_destination_directory,
        scratch_run_directory=scratch_run_directory,
        result_filenames=result_filenames,
        text_result_filenames_replacing_execution_directory=(
            text_result_filenames_replacing_execution_directory
        ),
    )


def _handle_successful_scratch_run(
    *,
    destination_directory: Path,
    scratch_run_directory: Path,
    result_filenames: tuple[str, ...],
    text_result_filenames_replacing_execution_directory: tuple[str, ...],
) -> None:
    try:
        _rewrite_text_result_paths(
            source_directory=scratch_run_directory,
            destination_directory=destination_directory,
            text_result_filenames=text_result_filenames_replacing_execution_directory,
        )
        _move_result_files(
            source_directory=scratch_run_directory,
            destination_directory=destination_directory,
            result_filenames=result_filenames,
        )
    except (OSError, ValueError) as exc:
        _write_scratch_execution_manifest(
            destination_directory=destination_directory,
            scratch_run_directory=scratch_run_directory,
            status=ScratchExecutionStatus.failed,
            error_message=str(exc),
        )
        raise

    _write_scratch_execution_manifest(
        destination_directory=destination_directory,
        scratch_run_directory=scratch_run_directory,
        status=ScratchExecutionStatus.success,
        error_message=None,
    )
    shutil.rmtree(scratch_run_directory, ignore_errors=True)


def _handle_failed_scratch_run(
    *,
    destination_directory: Path,
    scratch_run_directory: Path,
    error_message: str,
    result_filenames: tuple[str, ...],
    text_result_filenames_replacing_execution_directory: tuple[str, ...],
) -> None:
    manifest_error_message = error_message
    try:
        _rewrite_text_result_paths(
            source_directory=scratch_run_directory,
            destination_directory=destination_directory,
            text_result_filenames=text_result_filenames_replacing_execution_directory,
        )
        _move_result_files(
            source_directory=scratch_run_directory,
            destination_directory=destination_directory,
            result_filenames=result_filenames,
        )
    except (OSError, ValueError) as exc:
        manifest_error_message = (
            f"{error_message}; scratch result transfer failed: {exc!s}"
        )

    _write_scratch_execution_manifest(
        destination_directory=destination_directory,
        scratch_run_directory=scratch_run_directory,
        status=ScratchExecutionStatus.failed,
        error_message=manifest_error_message,
    )


def _create_scratch_directory(
    *, policy: LocalScratchPolicy, destination_directory: Path
) -> Path:
    scratch_parent = _resolve_scratch_parent(policy=policy)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f"{destination_directory.name}-", dir=scratch_parent)
    )


def _resolve_scratch_parent(*, policy: LocalScratchPolicy) -> Path:
    for environment_key in SCRATCH_PARENT_ENVIRONMENT_VARIABLES:
        scratch_parent = os.environ.get(environment_key)
        if scratch_parent:
            return Path(scratch_parent).resolve()
    if policy.fallback_scratch_parent is not None:
        return policy.fallback_scratch_parent.resolve()
    error_msg = "Scratch execution is enabled but no scratch parent is available."
    raise ScratchDirectoryUnavailableError(error_msg)


def _move_result_files(
    *,
    source_directory: Path,
    destination_directory: Path,
    result_filenames: tuple[str, ...],
) -> None:
    for result_filename in result_filenames:
        source_path = source_directory / result_filename
        if not source_path.exists():
            continue
        destination_path = destination_directory / result_filename
        temporary_destination_path = _temporary_destination_path(destination_path)
        shutil.move(source_path.as_posix(), temporary_destination_path.as_posix())
        _replace_destination_path(
            source_path=temporary_destination_path, destination_path=destination_path
        )


def _temporary_destination_path(destination_path: Path) -> Path:
    return destination_path.with_name(f".{destination_path.name}.scratch-move")


def _replace_destination_path(*, source_path: Path, destination_path: Path) -> None:
    if source_path.is_dir() or destination_path.is_dir():
        raise IsADirectoryError(source_path.as_posix())
    source_path.replace(destination_path)


def _rewrite_text_result_paths(
    *,
    source_directory: Path,
    destination_directory: Path,
    text_result_filenames: tuple[str, ...],
) -> None:
    for result_filename in text_result_filenames:
        result_path = source_directory / result_filename
        if not result_path.is_file():
            continue
        result_text = result_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
        result_path.write_text(
            result_text.replace(
                source_directory.as_posix(), destination_directory.as_posix()
            ),
            encoding=DEFAULT_TEXT_ENCODING,
        )


def _write_scratch_execution_manifest(
    *,
    destination_directory: Path,
    scratch_run_directory: Path,
    status: ScratchExecutionStatus,
    error_message: str | None,
) -> None:
    manifest = ScratchExecutionManifest(
        final_directory=destination_directory,
        scratch_directory=scratch_run_directory,
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get(SLURM_JOB_ID_ENVIRONMENT_VARIABLE),
        slurm_node_name=os.environ.get(SLURM_NODELIST_ENVIRONMENT_VARIABLE),
        status=status,
        error_message=error_message,
    )
    manifest_path = destination_directory / SCRATCH_EXECUTION_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode=MSONABLE_JSON_MODE), indent=2),
        encoding=DEFAULT_TEXT_ENCODING,
    )
