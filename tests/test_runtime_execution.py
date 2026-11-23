from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from fireworks.utilities.fw_serializers import load_object_from_file

from crystal.constants.fireworks_runtime import (
    RUNTIME_BASH_BINARY,
    RUNTIME_BASH_LOGIN_COMMAND_FLAG,
    RUNTIME_MPI_ENVIRONMENT_QADAPTER_FILENAME,
    RUNTIME_QADAPTER_PRE_ROCKET_KEY,
    RUNTIME_SOURCE_SHELL_COMMAND,
)
from crystal.helpers.fireworks.runtime_config import FireWorksRuntimePaths
from crystal.helpers.fireworks.runtime_execution import (
    FireWorksRuntimeExecutionManager,
    FireWorksRuntimeProcessHandle,
)
from crystal.schemas.fireworks_runtime import FireWorksRuntimeProfile
from tests.helpers.builders.fireworks_runtime import (
    build_fireworks_runtime_paths_fixture_for_profile,
    create_fireworks_runtime_launch_stub_executable,
)
from tests.helpers.constants.fireworks_runtime import (
    TEST_QLAUNCH_INFINITE_VALUE,
    TEST_QLAUNCH_MODE_RAPIDFIRE,
    TEST_RUNTIME_EXECUTION_FAKE_PID,
    TEST_RUNTIME_EXECUTION_TIMEOUT_SECONDS,
    TEST_RUNTIME_EXISTING_PRE_ROCKET_COMMAND,
    TEST_RUNTIME_MPI_ENVIRONMENT_SETUP_SCRIPT_CONTENT,
    TEST_RUNTIME_MPI_ENVIRONMENT_SETUP_SCRIPT_EXPORT_CONTENT,
    TEST_RUNTIME_MPI_ENVIRONMENT_SETUP_SCRIPT_FILENAME,
    TEST_RUNTIME_MPI_ENVIRONMENT_SETUP_SCRIPT_FILENAME_WITH_SPACE,
    TEST_RUNTIME_RLAUNCH_STUB_FILENAME,
)


def _build_queue_runtime_paths_with_mpi_environment_setup(
    tmp_path: Path, mpi_environment_setup_script_path: Path
) -> FireWorksRuntimePaths:
    runtime = build_fireworks_runtime_paths_fixture_for_profile(
        tmp_path=tmp_path,
        profile=FireWorksRuntimeProfile.SLURM,
        mpi_environment_setup_script_path=mpi_environment_setup_script_path,
    )
    qadapter_template_path = runtime.profile_paths.qadapter_template_path
    assert qadapter_template_path is not None
    qadapter_template_path.write_text(
        "_fw_name: CommonAdapter\n"
        "_fw_q_type: SLURM\n"
        "rocket_launch: rlaunch singleshot\n"
        f"logdir: {(tmp_path / 'queue-logs').as_posix()}\n"
        "pre_rocket: |\n"
        f"  {TEST_RUNTIME_EXISTING_PRE_ROCKET_COMMAND}\n",
        encoding="utf-8",
    )
    return runtime


class _FakeQlaunchProcess:
    def __init__(self, pid: int = TEST_RUNTIME_EXECUTION_FAKE_PID) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminate_called = False
        self.kill_called = False
        self.wait_timeout_raises = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_called = True

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_timeout_raises:
            self.wait_timeout_raises = False
            raise subprocess.TimeoutExpired(cmd="qlaunch", timeout=timeout or 0.0)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9


def _build_process_info_for_fake_qlaunch_process(
    fake_process: _FakeQlaunchProcess,
) -> FireWorksRuntimeProcessHandle:
    return FireWorksRuntimeProcessHandle.model_construct(
        process=cast("subprocess.Popen[str]", fake_process), command=()
    )


def test_process_info_reports_running_and_completed_success_states() -> None:
    running_process = _FakeQlaunchProcess()
    running_info = _build_process_info_for_fake_qlaunch_process(running_process)

    assert running_info.is_running is True
    assert running_info.return_code is None
    assert running_info.is_success is False

    running_process.returncode = 0
    completed_info = _build_process_info_for_fake_qlaunch_process(running_process)
    assert completed_info.is_running is False
    assert completed_info.return_code == 0
    assert completed_info.is_success is True


def test_stop_terminates_running_process_with_graceful_exit() -> None:
    fake_process = _FakeQlaunchProcess()
    process_info = _build_process_info_for_fake_qlaunch_process(fake_process)

    process_info.stop(timeout_seconds=TEST_RUNTIME_EXECUTION_TIMEOUT_SECONDS)

    assert fake_process.terminate_called is True
    assert fake_process.kill_called is False
    assert process_info.is_running is False
    assert process_info.return_code == 0
    assert process_info.is_success is True


def test_stop_kills_process_when_graceful_termination_times_out() -> None:
    fake_process = _FakeQlaunchProcess()
    fake_process.wait_timeout_raises = True
    process_info = _build_process_info_for_fake_qlaunch_process(fake_process)

    process_info.stop(timeout_seconds=TEST_RUNTIME_EXECUTION_TIMEOUT_SECONDS)

    assert fake_process.terminate_called is True
    assert fake_process.kill_called is True
    assert process_info.is_running is False
    assert process_info.return_code == -9
    assert process_info.is_success is False


def test_runtime_execution_integration_creates_validates_starts_monitors_and_stops_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = build_fireworks_runtime_paths_fixture_for_profile(
        tmp_path=tmp_path, profile=FireWorksRuntimeProfile.LOCAL
    )
    manager = FireWorksRuntimeExecutionManager(
        runtime_paths=runtime, run_directory=tmp_path / "runtime"
    )
    runtime_launch_bin_directory = tmp_path / "bin"
    _ = create_fireworks_runtime_launch_stub_executable(
        runtime_launch_bin_directory, executable_name=TEST_RUNTIME_RLAUNCH_STUB_FILENAME
    )
    monkeypatch.setenv(
        "PATH", f"{runtime_launch_bin_directory.as_posix()}:{os.environ['PATH']}"
    )

    handle = manager.start_launch(
        mode=TEST_QLAUNCH_MODE_RAPIDFIRE, nlaunches=TEST_QLAUNCH_INFINITE_VALUE
    )
    running_status_is_running = handle.is_running
    running_status_return_code = handle.return_code
    handle.stop(timeout_seconds=TEST_RUNTIME_EXECUTION_TIMEOUT_SECONDS)

    assert manager.pid_path.exists()
    assert manager.log_path.exists()
    assert running_status_is_running is True
    assert running_status_return_code is None
    assert handle.is_running is False
    assert handle.return_code is not None


def test_queue_launch_sources_mpi_environment_in_generated_qadapter(
    tmp_path: Path,
) -> None:
    mpi_environment_setup_script_path = (
        tmp_path / TEST_RUNTIME_MPI_ENVIRONMENT_SETUP_SCRIPT_FILENAME_WITH_SPACE
    )
    mpi_environment_setup_script_path.write_text(
        TEST_RUNTIME_MPI_ENVIRONMENT_SETUP_SCRIPT_CONTENT, encoding="utf-8"
    )
    runtime = _build_queue_runtime_paths_with_mpi_environment_setup(
        tmp_path=tmp_path,
        mpi_environment_setup_script_path=mpi_environment_setup_script_path,
    )
    manager = FireWorksRuntimeExecutionManager(
        runtime_paths=runtime, run_directory=tmp_path / "runtime"
    )
    runtime_launch_bin_directory = tmp_path / "bin"
    _ = create_fireworks_runtime_launch_stub_executable(runtime_launch_bin_directory)

    with patch.dict(
        os.environ, {"PATH": f"{runtime_launch_bin_directory}:{os.environ['PATH']}"}
    ):
        handle = manager.start_launch(
            mode=TEST_QLAUNCH_MODE_RAPIDFIRE, nlaunches=TEST_QLAUNCH_INFINITE_VALUE
        )
        handle.stop(timeout_seconds=TEST_RUNTIME_EXECUTION_TIMEOUT_SECONDS)

    generated_qadapter_path = (
        tmp_path / "runtime" / RUNTIME_MPI_ENVIRONMENT_QADAPTER_FILENAME
    )
    generated_qadapter = load_object_from_file(generated_qadapter_path.as_posix())
    assert generated_qadapter[RUNTIME_QADAPTER_PRE_ROCKET_KEY].splitlines() == [
        f"{RUNTIME_SOURCE_SHELL_COMMAND} "
        f"{shlex.quote(mpi_environment_setup_script_path.as_posix())}",
        TEST_RUNTIME_EXISTING_PRE_ROCKET_COMMAND,
    ]
    assert (tmp_path / "queue-logs").is_dir()
    assert generated_qadapter_path.as_posix() in handle.command


def test_local_launch_sources_mpi_environment_before_rlaunch(tmp_path: Path) -> None:
    mpi_environment_setup_script_path = (
        tmp_path / TEST_RUNTIME_MPI_ENVIRONMENT_SETUP_SCRIPT_FILENAME
    )
    mpi_environment_setup_script_path.write_text(
        TEST_RUNTIME_MPI_ENVIRONMENT_SETUP_SCRIPT_EXPORT_CONTENT, encoding="utf-8"
    )
    runtime = build_fireworks_runtime_paths_fixture_for_profile(
        tmp_path=tmp_path,
        profile=FireWorksRuntimeProfile.LOCAL,
        mpi_environment_setup_script_path=mpi_environment_setup_script_path,
    )
    manager = FireWorksRuntimeExecutionManager(
        runtime_paths=runtime, run_directory=tmp_path / "runtime"
    )
    runtime_launch_bin_directory = tmp_path / "bin"
    _ = create_fireworks_runtime_launch_stub_executable(
        runtime_launch_bin_directory, executable_name=TEST_RUNTIME_RLAUNCH_STUB_FILENAME
    )

    with patch.dict(
        os.environ, {"PATH": f"{runtime_launch_bin_directory}:{os.environ['PATH']}"}
    ):
        handle = manager.start_launch(
            mode=TEST_QLAUNCH_MODE_RAPIDFIRE, nlaunches=TEST_QLAUNCH_INFINITE_VALUE
        )
        command = handle.command
        handle.stop(timeout_seconds=TEST_RUNTIME_EXECUTION_TIMEOUT_SECONDS)

    expected_launch_arguments = runtime.launch_args(
        mode=TEST_QLAUNCH_MODE_RAPIDFIRE, nlaunches=TEST_QLAUNCH_INFINITE_VALUE
    )
    expected_mpi_environment_setup_command = (
        f"{RUNTIME_SOURCE_SHELL_COMMAND} "
        f"{shlex.quote(mpi_environment_setup_script_path.as_posix())}"
    )
    assert command == (
        RUNTIME_BASH_BINARY,
        RUNTIME_BASH_LOGIN_COMMAND_FLAG,
        f"{expected_mpi_environment_setup_command} && "
        f"exec {shlex.join(expected_launch_arguments)}",
    )
    assert handle.command == tuple(command)
