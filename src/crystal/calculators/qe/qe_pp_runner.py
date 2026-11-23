"""QE pp.x charge density runner utilities."""

from __future__ import annotations

import subprocess
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from crystal import CRYSTAL_CONFIG
from crystal.constants.qe_pp import (
    QE_PP_CHARGE_DENSITY_BASENAME,
    QE_PP_CONFIG_NOT_FOUND_MESSAGE,
    QE_PP_EXECUTABLE_NAME,
)
from crystal.helpers.process_run_info import build_process_run_info_for_failure
from crystal.schemas.process import ProcessRunInfo
from crystal.schemas.qe_pp import (
    QePpChargeDensityCubeInputBuilder,
    QePpInputSettings,
    QePpPlotSettings,
)


class QePpChargeDensityRunner(BaseModel):  # noqa: D101
    model_config = ConfigDict(frozen=True)

    run_directory: Path
    input_settings: QePpInputSettings
    plot_settings: QePpPlotSettings

    @cached_property
    def run_files(self) -> QePpChargeDensityRunFiles:  # noqa: D102
        return QePpChargeDensityRunFiles(
            run_directory=self.run_directory, cube_filename=self._builder.cube_filename
        )

    @cached_property
    def cube_path(self) -> Path:  # noqa: D102
        return self.run_files.cube_path

    @cached_property
    def _builder(self) -> QePpChargeDensityCubeInputBuilder:
        return QePpChargeDensityCubeInputBuilder(
            input_settings=self.input_settings, plot_settings=self.plot_settings
        )

    @cached_property
    def _pp_executable(self) -> Path:
        if CRYSTAL_CONFIG.programs.quantum_espresso_dir is None:
            raise RuntimeError(QE_PP_CONFIG_NOT_FOUND_MESSAGE)
        return CRYSTAL_CONFIG.programs.quantum_espresso_dir / QE_PP_EXECUTABLE_NAME

    def run(self) -> ProcessRunInfo:  # noqa: D102
        command_arguments: list[str] = []
        try:
            command_arguments = [str(self._pp_executable)]
            self.run_files.input_path.write_text(
                self._builder.pp_charge_density_input_text
            )
            with (
                self.run_files.input_path.open("r", encoding="utf-8") as stdin_handle,
                self.run_files.output_path.open("w", encoding="utf-8") as stdout_handle,
            ):
                completed_process = subprocess.run(
                    command_arguments,
                    cwd=self.run_directory,
                    check=True,
                    text=True,
                    stdin=stdin_handle,
                    stdout=stdout_handle,
                )
            return ProcessRunInfo(
                command_arguments=command_arguments,
                exit_code=completed_process.returncode,
                standard_output_file=self.run_files.output_path,
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
                standard_output_file=self.run_files.output_path,
                standard_error_file=None,
            )


class QePpChargeDensityRunFiles(BaseModel):  # noqa: D101
    model_config = ConfigDict(frozen=True)

    run_directory: Path
    cube_filename: str

    @cached_property
    def input_path(self) -> Path:  # noqa: D102
        return self.run_directory / f"{QE_PP_CHARGE_DENSITY_BASENAME}.in"

    @cached_property
    def output_path(self) -> Path:  # noqa: D102
        return self.run_directory / f"{QE_PP_CHARGE_DENSITY_BASENAME}.out"

    @cached_property
    def cube_path(self) -> Path:  # noqa: D102
        return self.run_directory / self.cube_filename
