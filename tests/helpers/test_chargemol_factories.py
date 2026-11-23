"""Factories for Chargemol-related test data."""

from __future__ import annotations

from pathlib import Path

from crystal.schemas.chargemol import (
    ChargemolJobControlSettings,
    QePpToChargemolWorkflowInputs,
)
from crystal.schemas.pseudopotential import PseudopotentialConfig
from crystal.schemas.qe_pp import QePpInputSettings, QePpPlotSettings
from tests.helpers.constants.qe_pp import QE_PP_TEST_INPUT_PREFIX
from tests.helpers.test_data_paths import resolve_tests_pseudopotentials_dir
from tests.helpers.test_paths import (
    TEST_LI_PBE_PSEUDOPOTENTIAL_FILENAME,
    TEST_NB_PBE_PSEUDOPOTENTIAL_FILENAME,
    TEST_O_PBE_PSEUDOPOTENTIAL_FILENAME,
)


def build_test_pseudopotential_config(
    test_file_path: Path, pseudopotentials_file_map: dict[str, str] | None = None
) -> PseudopotentialConfig:
    """Return pseudopotential configuration for Chargemol tests."""
    if pseudopotentials_file_map is None:
        pseudopotentials_file_map = {
            "Li": TEST_LI_PBE_PSEUDOPOTENTIAL_FILENAME,
            "Nb": TEST_NB_PBE_PSEUDOPOTENTIAL_FILENAME,
            "O": TEST_O_PBE_PSEUDOPOTENTIAL_FILENAME,
        }
    return PseudopotentialConfig(
        pseudopotential_dir=resolve_tests_pseudopotentials_dir(test_file_path),
        pseudopotentials_file_map=pseudopotentials_file_map,
    )


def build_default_qe_pp_input_settings(
    *, outdir: str, prefix: str = QE_PP_TEST_INPUT_PREFIX
) -> QePpInputSettings:
    """Return default QE pp.x input settings for tests."""
    return QePpInputSettings(prefix=prefix, outdir=outdir)


def build_default_qe_pp_plot_settings(fileout: str | None = None) -> QePpPlotSettings:
    """Return default QE pp.x plot settings for tests."""
    if fileout is None:
        return QePpPlotSettings()
    return QePpPlotSettings(fileout=fileout)


def build_default_qe_pp_to_chargemol_workflow_inputs(
    cube_output_directory: Path,
    pseudopotential_config: PseudopotentialConfig,
    input_settings: QePpInputSettings | None = None,
    plot_settings: QePpPlotSettings | None = None,
) -> QePpToChargemolWorkflowInputs:
    """Return workflow inputs for QE pp.x to Chargemol tests."""
    if input_settings is None:
        input_settings = build_default_qe_pp_input_settings(
            outdir=cube_output_directory.as_posix()
        )
    if plot_settings is None:
        plot_settings = build_default_qe_pp_plot_settings()
    return QePpToChargemolWorkflowInputs(
        cube_output_directory=cube_output_directory,
        input_settings=input_settings,
        plot_settings=plot_settings,
        pseudopotential_config=pseudopotential_config,
    )


def build_expected_chargemol_job_control_settings(
    input_filename: Path,
    atomic_densities_directory: Path,
    *,
    periodic_a: bool = True,
    periodic_b: bool = True,
    periodic_c: bool = True,
    core_electrons_by_atomic_number: dict[int, int] | None = None,
) -> ChargemolJobControlSettings:
    """Return expected Chargemol job_control settings for tests."""
    if core_electrons_by_atomic_number is None:
        core_electrons_by_atomic_number = {3: 0, 8: 2, 41: 28}
    return ChargemolJobControlSettings(
        input_filename=input_filename,
        atomic_densities_directory=atomic_densities_directory,
        periodic_a=periodic_a,
        periodic_b=periodic_b,
        periodic_c=periodic_c,
        core_electrons_by_atomic_number=core_electrons_by_atomic_number,
    )
