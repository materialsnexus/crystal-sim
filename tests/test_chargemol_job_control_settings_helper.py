"""Tests for Chargemol job_control settings helper."""

from __future__ import annotations

from pathlib import Path

from crystal.schemas.chargemol import ChargemolJobControlSettings
from tests.helpers.test_chargemol_factories import (
    build_default_qe_pp_input_settings,
    build_default_qe_pp_plot_settings,
    build_expected_chargemol_job_control_settings,
    build_test_pseudopotential_config,
)
from tests.helpers.test_paths import (
    TEST_CHARGEMOL_ATOMIC_DENSITIES_DIR,
    TEST_CUBES_DIR,
    TEST_PW_CHARGE_DENSITY_CUBE_IN_CUBES,
    TEST_PW_CHARGE_DENSITY_CUBE_PATH,
)


def test_build_job_control_settings_from_qe_pseudopotentials() -> None:
    """Helper should derive core electrons and periodicity for Chargemol."""
    pseudopotential_config = build_test_pseudopotential_config(Path(__file__))
    settings = ChargemolJobControlSettings.from_qe_pseudopotentials(
        input_cube_path=TEST_PW_CHARGE_DENSITY_CUBE_PATH,
        atomic_densities_directory=TEST_CHARGEMOL_ATOMIC_DENSITIES_DIR,
        pseudopotential_config=pseudopotential_config,
    )
    expected = build_expected_chargemol_job_control_settings(
        input_filename=TEST_PW_CHARGE_DENSITY_CUBE_PATH,
        atomic_densities_directory=TEST_CHARGEMOL_ATOMIC_DENSITIES_DIR,
    )
    assert settings == expected


def test_build_job_control_settings_from_qe_pp_inputs() -> None:
    """Helper should derive cube path from QE pp.x inputs."""
    pseudopotential_config = build_test_pseudopotential_config(Path(__file__))
    settings = ChargemolJobControlSettings.from_qe_pp_inputs(
        cube_output_directory=TEST_CUBES_DIR,
        input_settings=build_default_qe_pp_input_settings(
            outdir=TEST_CUBES_DIR.as_posix()
        ),
        plot_settings=build_default_qe_pp_plot_settings(),
        atomic_densities_directory=TEST_CHARGEMOL_ATOMIC_DENSITIES_DIR,
        pseudopotential_config=pseudopotential_config,
    )
    expected = build_expected_chargemol_job_control_settings(
        input_filename=TEST_PW_CHARGE_DENSITY_CUBE_IN_CUBES,
        atomic_densities_directory=TEST_CHARGEMOL_ATOMIC_DENSITIES_DIR,
    )
    assert settings == expected
