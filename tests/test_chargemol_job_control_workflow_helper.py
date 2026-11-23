"""Tests for Chargemol job_control workflow helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from crystal import CRYSTAL_CONFIG, Programs
from crystal.constants.chargemol import CHARGEMOL_JOB_CONTROL_FILENAME
from crystal.helpers.chargemol.job_control_workflow import ChargemolJobControlWriter
from crystal.schemas.chargemol import ChargemolJobControlInputBuilder
from tests.helpers.test_chargemol_factories import (
    build_default_qe_pp_to_chargemol_workflow_inputs,
    build_expected_chargemol_job_control_settings,
    build_test_pseudopotential_config,
)
from tests.helpers.test_paths import (
    TEST_CHARGEMOL_ATOMIC_DENSITIES_DIRNAME,
    TEST_CHARGEMOL_ROOT_DIRNAME,
    TEST_CUBES_DIRNAME,
    TEST_JOB_DIRNAME,
    TEST_PW_CHARGE_DENSITY_CUBE_FILENAME,
)


def test_job_control_writer_uses_discovered_paths(tmp_path: Path) -> None:
    """Workflow helper should use crystal.toml paths to derive atomic densities."""
    chargemol_root = tmp_path / TEST_CHARGEMOL_ROOT_DIRNAME
    job_control_directory, cube_output_directory = _build_job_control_directories(
        tmp_path
    )
    pseudopotential_config = build_test_pseudopotential_config(Path(__file__))
    payload = build_default_qe_pp_to_chargemol_workflow_inputs(
        cube_output_directory=cube_output_directory,
        pseudopotential_config=pseudopotential_config,
    )

    with patch.object(
        CRYSTAL_CONFIG, "programs", Programs(chargemol_dir=chargemol_root)
    ):
        job_control_writer = ChargemolJobControlWriter(workflow_inputs=payload)
        job_control_path = job_control_writer.write(
            job_control_directory=job_control_directory
        )

    expected_text = _build_expected_job_control_text(
        cube_output_directory=cube_output_directory, chargemol_root=chargemol_root
    )
    assert job_control_path == job_control_directory / CHARGEMOL_JOB_CONTROL_FILENAME
    assert job_control_path.read_text() == expected_text


def _build_expected_job_control_text(
    cube_output_directory: Path, chargemol_root: Path
) -> str:
    expected_settings = build_expected_chargemol_job_control_settings(
        input_filename=cube_output_directory / TEST_PW_CHARGE_DENSITY_CUBE_FILENAME,
        atomic_densities_directory=(
            chargemol_root / TEST_CHARGEMOL_ATOMIC_DENSITIES_DIRNAME
        ),
    )
    return ChargemolJobControlInputBuilder(settings=expected_settings).job_control_text


def _build_job_control_directories(tmp_path: Path) -> tuple[Path, Path]:
    job_control_directory = tmp_path / TEST_JOB_DIRNAME
    job_control_directory.mkdir()
    cube_output_directory = tmp_path / TEST_CUBES_DIRNAME
    cube_output_directory.mkdir()
    return job_control_directory, cube_output_directory
