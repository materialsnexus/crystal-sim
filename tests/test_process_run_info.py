"""Tests for ProcessRunInfo failure helpers."""

from __future__ import annotations

import subprocess

import pytest

from crystal.constants.process import (
    PROCESS_EXIT_CODE_COMMAND_NOT_FOUND,
    PROCESS_EXIT_CODE_SUCCESS,
)
from crystal.helpers.process_run_info import build_process_run_info_for_failure
from crystal.schemas.process import ProcessRunInfo
from tests.helpers.constants.process import (
    TEST_FILE_NOT_FOUND_ERROR_MESSAGE,
    TEST_PROCESS_FAILURE_EXIT_CODE,
    TEST_UNSUPPORTED_ERROR_MESSAGE,
)
from tests.helpers.constants.qe_pp import QE_PP_EXECUTABLE


def test_build_process_run_info_for_failure_from_called_process_error() -> None:
    error = subprocess.CalledProcessError(
        TEST_PROCESS_FAILURE_EXIT_CODE, [QE_PP_EXECUTABLE]
    )
    run_info = build_process_run_info_for_failure(
        error=error,
        command_arguments=[QE_PP_EXECUTABLE],
        standard_output_file=None,
        standard_error_file=None,
    )
    assert isinstance(run_info, ProcessRunInfo)
    assert run_info.exit_code == TEST_PROCESS_FAILURE_EXIT_CODE
    assert run_info.error_message == str(error)
    assert run_info.exit_code_ok is False


def test_build_process_run_info_for_failure_from_file_not_found_error() -> None:
    error = FileNotFoundError(TEST_FILE_NOT_FOUND_ERROR_MESSAGE)
    run_info = build_process_run_info_for_failure(
        error=error,
        command_arguments=[],
        standard_output_file=None,
        standard_error_file=None,
    )
    assert isinstance(run_info, ProcessRunInfo)
    assert run_info.exit_code == PROCESS_EXIT_CODE_COMMAND_NOT_FOUND
    assert run_info.error_message == str(error)
    assert run_info.exit_code_ok is False


def test_build_process_run_info_for_failure_rejects_unexpected_exception() -> None:
    with pytest.raises(TypeError):
        build_process_run_info_for_failure(
            error=ValueError(TEST_UNSUPPORTED_ERROR_MESSAGE),
            command_arguments=[],
            standard_output_file=None,
            standard_error_file=None,
        )


def test_process_run_info_exit_code_ok_when_zero() -> None:
    run_info = ProcessRunInfo(
        command_arguments=[QE_PP_EXECUTABLE],
        exit_code=PROCESS_EXIT_CODE_SUCCESS,
        standard_output_file=None,
        standard_error_file=None,
        error_message=None,
    )
    assert run_info.exit_code_ok is True
