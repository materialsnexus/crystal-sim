"""Tests for GPAW Chargemol maker behavior."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jobflow.managers.local import run_locally

from crystal.calculators.gpaw.chargemol_maker import GpawChargemolMaker
from crystal.constants.chargemol import (
    CHARGEMOL_ATOMIC_DENSITIES_DIRECTORY_PLACEHOLDER,
    CHARGEMOL_EXECUTABLE_NAME,
    CHARGEMOL_JOB_CONTROL_FILENAME,
    CHARGEMOL_MISSING_OUTPUT_FILES_MESSAGE_TEMPLATE,
    CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME,
)
from crystal.constants.gpaw import (
    GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES,
    GPAW_TOTAL_DENSITY_CUBE_FILENAME,
)
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.schemas.chargemol import (
    ChargemolInstallPathsConfig,
    ChargemolJobControlInputBuilder,
    ChargemolJobControlSettings,
)
from crystal.schemas.gpaw import (
    GpawChargemolSimulationSettings,
    GpawChargemolTaskDocument,
)
from crystal.schemas.properties import SinglePointProperties
from tests.helpers.constants.gpaw_chargemol import (
    TEST_CHARGEMOL_FAILURE_EXIT_CODE,
    TEST_CHARGEMOL_INPUT_FILENAME_CLOSING_TAG,
    TEST_CHARGEMOL_INPUT_FILENAME_OPENING_TAG,
)
from tests.helpers.constants.process import TEST_PROCESS_SUCCESS_EXIT_CODE
from tests.helpers.constants.structure import TEST_SI_STRUCTURE

TEST_DDEC6_CHARGES_HEADER_LINE = "index symbol x y charge"
TEST_ZERO_CHARGE_VALUE = 0.0
TEST_NONZERO_CHARGE_VALUE = 0.1
TEST_DDEC6_DUMMY_COORD_VALUE = 0.0
TEST_ZERO_CHARGES = [TEST_ZERO_CHARGE_VALUE for _ in range(len(TEST_SI_STRUCTURE))]
TEST_NONZERO_CHARGES = [
    TEST_NONZERO_CHARGE_VALUE for _ in range(len(TEST_SI_STRUCTURE))
]
TEST_ZERO_FORCE_VECTOR = [
    TEST_ZERO_CHARGE_VALUE,
    TEST_ZERO_CHARGE_VALUE,
    TEST_ZERO_CHARGE_VALUE,
]
TEST_ZERO_FORCES = [TEST_ZERO_FORCE_VECTOR for _ in range(len(TEST_SI_STRUCTURE))]
TEST_ZERO_STRESS = [TEST_ZERO_CHARGE_VALUE] * 6
TEST_ZERO_ENERGY = TEST_ZERO_CHARGE_VALUE


def _build_minimal_job_control_text() -> str:
    return (
        f"{TEST_CHARGEMOL_INPUT_FILENAME_OPENING_TAG}\n"
        f"{GPAW_TOTAL_DENSITY_CUBE_FILENAME}\n"
        f"{TEST_CHARGEMOL_INPUT_FILENAME_CLOSING_TAG}\n"
    )


def _write_minimal_expected_chargemol_outputs(run_directory: Path) -> None:
    for output_filename in GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES:
        (run_directory / output_filename).write_text("", encoding=DEFAULT_TEXT_ENCODING)


def _write_chargemol_net_atomic_charges(
    run_directory: Path, *, charges: list[float]
) -> None:
    atom_count = len(charges)
    lines = [str(atom_count), TEST_DDEC6_CHARGES_HEADER_LINE]
    for atom_index, (site, charge) in enumerate(
        zip(TEST_SI_STRUCTURE, charges), start=1
    ):
        symbol = str(site.specie)
        lines.append(
            f"{symbol} {atom_index} {TEST_DDEC6_DUMMY_COORD_VALUE} "
            f"{TEST_DDEC6_DUMMY_COORD_VALUE} {charge}"
        )
    (run_directory / CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME).write_text(
        "\n".join(lines) + "\n", encoding=DEFAULT_TEXT_ENCODING
    )


def _build_single_point_properties() -> SinglePointProperties:
    return SinglePointProperties(
        step=0,
        structure=TEST_SI_STRUCTURE,
        forces=TEST_ZERO_FORCES,
        stress=TEST_ZERO_STRESS,
        energy=TEST_ZERO_ENERGY,
    )


def _build_job_control_with_atomic_densities_placeholder(run_directory: Path) -> str:
    settings = ChargemolJobControlSettings(
        input_filename=run_directory / GPAW_TOTAL_DENSITY_CUBE_FILENAME,
        atomic_densities_directory=Path(
            CHARGEMOL_ATOMIC_DENSITIES_DIRECTORY_PLACEHOLDER
        ),
        periodic_a=True,
        periodic_b=True,
        periodic_c=True,
        core_electrons_by_atomic_number={14: 10},
    )
    return ChargemolJobControlInputBuilder(settings=settings).job_control_text


@patch("crystal.calculators.gpaw.chargemol_maker.subprocess.run")
def test_gpaw_chargemol_maker_runs_chargemol_from_existing_job_control(
    mock_subprocess_run: MagicMock, tmp_path: Path
) -> None:
    run_directory = tmp_path
    (run_directory / CHARGEMOL_JOB_CONTROL_FILENAME).write_text(
        _build_minimal_job_control_text(), encoding=DEFAULT_TEXT_ENCODING
    )
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        [CHARGEMOL_EXECUTABLE_NAME, CHARGEMOL_JOB_CONTROL_FILENAME],
        TEST_PROCESS_SUCCESS_EXIT_CODE,
    )
    _write_minimal_expected_chargemol_outputs(run_directory=run_directory)
    _write_chargemol_net_atomic_charges(run_directory, charges=TEST_ZERO_CHARGES)

    maker = GpawChargemolMaker(simulation_settings=GpawChargemolSimulationSettings())
    job = maker.make(
        job_control_directory=run_directory,
        structure=TEST_SI_STRUCTURE,
        properties=_build_single_point_properties(),
    )
    run_job = run_locally(job, root_dir=run_directory, create_folders=True)

    output_doc = run_job[job.uuid][1].output
    assert isinstance(output_doc, GpawChargemolTaskDocument)
    assert output_doc.converged
    assert output_doc.chargemol_run_info is not None
    assert output_doc.properties is not None
    assert output_doc.structure is not None
    assert output_doc.properties.charges == TEST_ZERO_CHARGES
    assert output_doc.properties.structure == output_doc.structure
    assert "ddec6_charges" in output_doc.structure.site_properties
    assert (
        output_doc.chargemol_run_info.command_arguments[1]
        == CHARGEMOL_JOB_CONTROL_FILENAME
    )


@patch("crystal.calculators.gpaw.chargemol_maker.subprocess.run")
def test_gpaw_chargemol_maker_records_error_on_chargemol_failure(
    mock_subprocess_run: MagicMock, tmp_path: Path
) -> None:
    run_directory = tmp_path
    (run_directory / CHARGEMOL_JOB_CONTROL_FILENAME).write_text(
        _build_minimal_job_control_text(), encoding=DEFAULT_TEXT_ENCODING
    )
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        TEST_CHARGEMOL_FAILURE_EXIT_CODE,
        [CHARGEMOL_EXECUTABLE_NAME, CHARGEMOL_JOB_CONTROL_FILENAME],
    )

    maker = GpawChargemolMaker(simulation_settings=GpawChargemolSimulationSettings())
    job = maker.make(
        job_control_directory=run_directory,
        structure=TEST_SI_STRUCTURE,
        properties=_build_single_point_properties(),
    )
    run_job = run_locally(job, root_dir=run_directory, create_folders=True)

    output_doc = run_job[job.uuid][1].output
    assert isinstance(output_doc, GpawChargemolTaskDocument)
    assert not output_doc.converged
    assert output_doc.errors != ""
    assert output_doc.chargemol_run_info is None


@pytest.mark.parametrize(
    ("error_to_raise", "expected_error_text"),
    [
        (RuntimeError("runtime failure"), "runtime failure"),
        (OSError("os failure"), "os failure"),
    ],
)
@patch("crystal.calculators.gpaw.chargemol_maker.subprocess.run")
def test_gpaw_chargemol_maker_records_error_on_runtime_or_os_failure(
    mock_subprocess_run: MagicMock,
    tmp_path: Path,
    error_to_raise: Exception,
    expected_error_text: str,
) -> None:
    run_directory = tmp_path
    (run_directory / CHARGEMOL_JOB_CONTROL_FILENAME).write_text(
        _build_minimal_job_control_text(), encoding=DEFAULT_TEXT_ENCODING
    )
    mock_subprocess_run.side_effect = error_to_raise

    maker = GpawChargemolMaker(simulation_settings=GpawChargemolSimulationSettings())
    job = maker.make(
        job_control_directory=run_directory,
        structure=TEST_SI_STRUCTURE,
        properties=_build_single_point_properties(),
    )
    run_job = run_locally(job, root_dir=run_directory, create_folders=True)

    output_doc = run_job[job.uuid][1].output
    assert isinstance(output_doc, GpawChargemolTaskDocument)
    assert not output_doc.converged
    assert output_doc.chargemol_run_info is None
    assert output_doc.errors == expected_error_text


@patch("crystal.calculators.gpaw.chargemol_maker.subprocess.run")
def test_gpaw_chargemol_maker_reports_missing_outputs_after_successful_run(
    mock_subprocess_run: MagicMock, tmp_path: Path
) -> None:
    run_directory = tmp_path
    (run_directory / CHARGEMOL_JOB_CONTROL_FILENAME).write_text(
        _build_minimal_job_control_text(), encoding=DEFAULT_TEXT_ENCODING
    )
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        [CHARGEMOL_EXECUTABLE_NAME, CHARGEMOL_JOB_CONTROL_FILENAME],
        TEST_PROCESS_SUCCESS_EXIT_CODE,
    )

    maker = GpawChargemolMaker(simulation_settings=GpawChargemolSimulationSettings())
    job = maker.make(
        job_control_directory=run_directory,
        structure=TEST_SI_STRUCTURE,
        properties=_build_single_point_properties(),
    )
    run_job = run_locally(job, root_dir=run_directory, create_folders=True)

    output_doc = run_job[job.uuid][1].output
    missing_filenames = ", ".join(GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES)
    assert isinstance(output_doc, GpawChargemolTaskDocument)
    assert not output_doc.converged
    assert output_doc.chargemol_run_info is None
    assert output_doc.errors == CHARGEMOL_MISSING_OUTPUT_FILES_MESSAGE_TEMPLATE.format(
        missing_filenames=missing_filenames
    )


@patch("crystal.calculators.gpaw.chargemol_maker.subprocess.run")
@patch(
    "crystal.calculators.gpaw.chargemol_maker.ChargemolInstallPathsConfigLoader.discover"
)
def test_gpaw_chargemol_maker_replaces_atomic_densities_directory_before_run(
    mock_discover: MagicMock, mock_subprocess_run: MagicMock, tmp_path: Path
) -> None:
    run_directory = tmp_path
    job_control_path = run_directory / CHARGEMOL_JOB_CONTROL_FILENAME
    job_control_path.write_text(
        _build_job_control_with_atomic_densities_placeholder(run_directory),
        encoding=DEFAULT_TEXT_ENCODING,
    )
    atomic_densities_directory = run_directory / "atomic_densities"
    mock_discover.return_value = ChargemolInstallPathsConfig(
        chargemol_binary_path=run_directory / "bin" / CHARGEMOL_EXECUTABLE_NAME,
        atomic_densities_directory=atomic_densities_directory,
    )
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        [CHARGEMOL_EXECUTABLE_NAME, CHARGEMOL_JOB_CONTROL_FILENAME],
        TEST_PROCESS_SUCCESS_EXIT_CODE,
    )
    _write_minimal_expected_chargemol_outputs(run_directory=run_directory)
    _write_chargemol_net_atomic_charges(run_directory, charges=TEST_ZERO_CHARGES)

    maker = GpawChargemolMaker(simulation_settings=GpawChargemolSimulationSettings())
    job = maker.make(
        job_control_directory=run_directory,
        structure=TEST_SI_STRUCTURE,
        properties=_build_single_point_properties(),
    )
    run_job = run_locally(job, root_dir=run_directory, create_folders=True)

    output_doc = run_job[job.uuid][1].output
    expected_directory_line = (
        ChargemolJobControlInputBuilder.format_chargemol_atomic_densities_directory(
            atomic_densities_directory
        )
    )
    updated_job_control_text = job_control_path.read_text(
        encoding=DEFAULT_TEXT_ENCODING
    )
    assert isinstance(output_doc, GpawChargemolTaskDocument)
    assert output_doc.converged
    assert (
        CHARGEMOL_ATOMIC_DENSITIES_DIRECTORY_PLACEHOLDER not in updated_job_control_text
    )
    assert expected_directory_line in updated_job_control_text
    mock_discover.assert_called_once()


@patch("crystal.calculators.gpaw.chargemol_maker.subprocess.run")
def test_gpaw_chargemol_maker_attaches_charges_to_structure(
    mock_subprocess_run: MagicMock, tmp_path: Path
) -> None:
    run_directory = tmp_path
    (run_directory / CHARGEMOL_JOB_CONTROL_FILENAME).write_text(
        _build_minimal_job_control_text(), encoding=DEFAULT_TEXT_ENCODING
    )
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        [CHARGEMOL_EXECUTABLE_NAME, CHARGEMOL_JOB_CONTROL_FILENAME],
        TEST_PROCESS_SUCCESS_EXIT_CODE,
    )
    _write_minimal_expected_chargemol_outputs(run_directory=run_directory)
    _write_chargemol_net_atomic_charges(run_directory, charges=TEST_NONZERO_CHARGES)

    maker = GpawChargemolMaker(simulation_settings=GpawChargemolSimulationSettings())
    job = maker.make(
        job_control_directory=run_directory,
        structure=TEST_SI_STRUCTURE,
        properties=_build_single_point_properties(),
    )
    run_job = run_locally(job, root_dir=run_directory, create_folders=True)

    output_doc = run_job[job.uuid][1].output
    assert isinstance(output_doc, GpawChargemolTaskDocument)
    assert output_doc.structure is not None
    assert output_doc.structure.site_properties["ddec6_charges"] == TEST_NONZERO_CHARGES
