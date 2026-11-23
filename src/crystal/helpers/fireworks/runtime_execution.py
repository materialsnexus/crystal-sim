# noqa: D100
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from crystal.constants.fireworks_runtime import (
    RUNTIME_LAUNCH_LOG_FILENAME,
    RUNTIME_LAUNCH_PID_FILENAME,
    RUNTIME_MPI_ENVIRONMENT_QADAPTER_FILENAME,
)
from crystal.helpers.fireworks.runtime_config import FireWorksRuntimePaths
from crystal.helpers.fireworks.runtime_environment import (
    build_local_launch_command_with_mpi_environment,
    ensure_qadapter_log_directory,
    write_mpi_environment_qadapter,
)


class FireWorksRuntimeProcessHandle(BaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    process: subprocess.Popen[str]
    command: tuple[str, ...]

    @property
    def pid(self) -> int:  # noqa: D102
        return int(self.process.pid)

    @property
    def return_code(self) -> int | None:  # noqa: D102
        return self.process.poll()

    @property
    def is_running(self) -> bool:  # noqa: D102
        return self.return_code is None

    @property
    def is_success(self) -> bool:  # noqa: D102
        return self.return_code == 0 and self.is_running is False

    def stop(self, timeout_seconds: float = 5.0) -> None:  # noqa: D102
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout_seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class FireWorksRuntimeExecutionManager(BaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    runtime_paths: FireWorksRuntimePaths
    run_directory: Path

    @property
    def resolved_run_directory(self) -> Path:  # noqa: D102
        return self.run_directory.resolve()

    @property
    def log_path(self) -> Path:  # noqa: D102
        return self.resolved_run_directory / RUNTIME_LAUNCH_LOG_FILENAME

    @property
    def pid_path(self) -> Path:  # noqa: D102
        return self.resolved_run_directory / RUNTIME_LAUNCH_PID_FILENAME

    def _build_runtime_qadapter_path(self, run_directory: Path) -> Path | None:
        qadapter_template_path = self.runtime_paths.profile_paths.qadapter_template_path
        mpi_environment_setup_script_path = (
            self.runtime_paths.mpi_environment_setup_script_path
        )
        if qadapter_template_path is None:
            return None
        if mpi_environment_setup_script_path is None:
            ensure_qadapter_log_directory(qadapter_template_path)
            return qadapter_template_path

        generated_qadapter_path = (
            run_directory / RUNTIME_MPI_ENVIRONMENT_QADAPTER_FILENAME
        )
        qadapter_path = write_mpi_environment_qadapter(
            qadapter_template_path=qadapter_template_path,
            mpi_environment_setup_script_path=mpi_environment_setup_script_path,
            output_path=generated_qadapter_path,
        )
        ensure_qadapter_log_directory(qadapter_path)
        return qadapter_path

    def _build_launch_command(
        self, run_directory: Path, mode: str, nlaunches: int | str
    ) -> list[str]:
        qadapter_override_path = self._build_runtime_qadapter_path(run_directory)
        launch_arguments = self.runtime_paths.launch_args(
            mode=mode,
            nlaunches=nlaunches,
            qadapter_override_path=qadapter_override_path,
        )
        mpi_environment_setup_script_path = (
            self.runtime_paths.mpi_environment_setup_script_path
        )
        if (
            mpi_environment_setup_script_path is None
            or self.runtime_paths.is_queue_launch
        ):
            return launch_arguments
        return build_local_launch_command_with_mpi_environment(
            launch_arguments=launch_arguments,
            mpi_environment_setup_script_path=mpi_environment_setup_script_path,
        )

    def start_launch(
        self, mode: str = "rapidfire", nlaunches: int | str = "infinite"
    ) -> FireWorksRuntimeProcessHandle:
        """Start runtime launch command with persisted pid/log paths."""
        run_directory = self.resolved_run_directory
        run_directory.mkdir(parents=True, exist_ok=True)
        command = self._build_launch_command(
            run_directory=run_directory, mode=mode, nlaunches=nlaunches
        )
        environment = os.environ.copy()
        environment.update(self.runtime_paths.environment())
        with self.log_path.open(mode="a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=run_directory,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        self.pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        return FireWorksRuntimeProcessHandle(process=process, command=tuple(command))
