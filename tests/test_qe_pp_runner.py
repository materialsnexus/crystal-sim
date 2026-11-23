"""Tests for QE pp.x charge density runner utilities."""

from __future__ import annotations

from pathlib import Path

from crystal.calculators.qe.qe_pp_runner import (
    QePpChargeDensityRunFiles,
    QePpChargeDensityRunner,
)
from crystal.constants.qe_pp import QE_PP_CHARGE_DENSITY_BASENAME
from crystal.schemas.qe_pp import (
    QePpChargeDensityCubeInputBuilder,
    QePpInputSettings,
    QePpPlotSettings,
)
from tests.helpers.constants.qe_pp import (
    QE_PP_CUSTOM_CUBE_FILENAME,
    QE_PP_DEFAULT_RUN_PREFIX,
)


def test_run_files_paths_use_basename_and_cube_filename(tmp_path: Path) -> None:
    run_files = QePpChargeDensityRunFiles(
        run_directory=tmp_path, cube_filename=QE_PP_CUSTOM_CUBE_FILENAME
    )
    assert run_files.input_path == tmp_path / f"{QE_PP_CHARGE_DENSITY_BASENAME}.in"
    assert run_files.output_path == tmp_path / f"{QE_PP_CHARGE_DENSITY_BASENAME}.out"
    assert run_files.cube_path == tmp_path / QE_PP_CUSTOM_CUBE_FILENAME


def test_runner_run_files_use_builder_cube_filename(tmp_path: Path) -> None:
    input_settings = QePpInputSettings(
        prefix=QE_PP_DEFAULT_RUN_PREFIX, outdir=tmp_path.as_posix()
    )
    plot_settings = QePpPlotSettings()
    runner = QePpChargeDensityRunner(
        run_directory=tmp_path,
        input_settings=input_settings,
        plot_settings=plot_settings,
    )
    builder = QePpChargeDensityCubeInputBuilder(
        input_settings=input_settings, plot_settings=plot_settings
    )
    assert runner.run_files.cube_path.name == builder.cube_filename
