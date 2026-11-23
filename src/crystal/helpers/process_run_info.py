"""ProcessRunInfo helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from crystal.constants.process import (
    PROCESS_EXIT_CODE_COMMAND_NOT_FOUND,
    PROCESS_EXIT_CODE_GENERIC_FAILURE,
    PROCESS_RUN_INFO_UNSUPPORTED_ERROR_TYPE_MESSAGE,
)
from crystal.schemas.process import ProcessRunInfo


def build_process_run_info_for_failure(  # noqa: D103
    *,
    error: Exception,
    command_arguments: list[str],
    standard_output_file: Path | None,
    standard_error_file: Path | None,
) -> ProcessRunInfo:
    if isinstance(error, subprocess.CalledProcessError):
        exit_code = error.returncode
    elif isinstance(error, FileNotFoundError):
        exit_code = PROCESS_EXIT_CODE_COMMAND_NOT_FOUND
    elif isinstance(error, (RuntimeError, OSError)):
        exit_code = PROCESS_EXIT_CODE_GENERIC_FAILURE
    else:
        raise TypeError(PROCESS_RUN_INFO_UNSUPPORTED_ERROR_TYPE_MESSAGE)
    return ProcessRunInfo(
        command_arguments=command_arguments,
        exit_code=exit_code,
        standard_output_file=standard_output_file,
        standard_error_file=standard_error_file,
        error_message=str(error),
    )
