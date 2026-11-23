"""Chargemol runner utilities."""

from __future__ import annotations

import subprocess
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from crystal import CRYSTAL_CONFIG
from crystal.constants.chargemol import (
    CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE,
    CHARGEMOL_EXECUTABLE_RELATIVE_PATH,
)
from crystal.helpers.chargemol.job_control_workflow import ChargemolJobControlWriter
from crystal.helpers.process_run_info import build_process_run_info_for_failure
from crystal.schemas.chargemol import QePpToChargemolWorkflowInputs
from crystal.schemas.process import ProcessRunInfo


class ChargemolRunner(BaseModel):  # noqa: D101
    model_config = ConfigDict(frozen=True)

    run_directory: Path
    workflow_inputs: QePpToChargemolWorkflowInputs

    @cached_property
    def job_control_path(self) -> Path:  # noqa: D102
        writer = ChargemolJobControlWriter(workflow_inputs=self.workflow_inputs)
        return writer.write(self.run_directory)

    @cached_property
    def executable(self) -> Path:  # noqa: D102
        chargemol_binary = CRYSTAL_CONFIG.programs.chargemol_dir
        if chargemol_binary is None:
            raise FileNotFoundError(CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE)
        return (Path(chargemol_binary) / CHARGEMOL_EXECUTABLE_RELATIVE_PATH).resolve()

    def run(self) -> ProcessRunInfo:  # noqa: D102
        command_arguments: list[str] = []
        try:
            command_arguments = [str(self.executable), self.job_control_path.name]
            completed_process = subprocess.run(
                command_arguments, cwd=self.run_directory, check=True, text=True
            )
            return ProcessRunInfo(
                command_arguments=command_arguments,
                exit_code=completed_process.returncode,
                standard_output_file=None,
                standard_error_file=None,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            RuntimeError,
            OSError,
        ) as exc:
            return build_process_run_info_for_failure(
                error=exc,
                command_arguments=command_arguments,
                standard_output_file=None,
                standard_error_file=None,
            )
