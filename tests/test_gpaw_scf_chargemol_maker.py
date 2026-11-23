from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

from jobflow.managers.local import run_locally
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.gpaw.gpaw_runner import GpawScfRunResult
from crystal.calculators.gpaw.scf_chargemol_maker import GpawScfChargemolMaker
from crystal.constants.gpaw import GPAW_TOTAL_DENSITY_CUBE_FILENAME
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.gpaw import (
    GpawChargemolSimulationSettings,
    GpawChargemolTaskDocument,
    GpawRuntimeProvenance,
    GpawScfChargemolMakerSettings,
    GpawScfChargemolSimulationSettings,
    GpawScfChargemolTaskDocument,
    GpawScfSimulationSettings,
)
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.builders.process_run_info import (
    build_successful_process_run_info_for_test,
)
from tests.helpers.builders.properties import build_single_point_properties_for_test
from tests.helpers.constants.structure import MOCK_TASK_DIR_NAME, TEST_SI_STRUCTURE

TEST_MOCK_GPAW_CHARGEMOL_DIR_NAME = f"{MOCK_TASK_DIR_NAME}-gpaw-chargemol"
TEST_GPAW_SCF_AND_CHARGEMOL_EXECUTOR_PROCESSES = 6
TEST_GPAW_VERSION = "26.7.0"
TEST_SILICON_SETUP_DIGEST = "a" * 64


def _build_runtime_provenance() -> GpawRuntimeProvenance:
    return GpawRuntimeProvenance(
        gpaw_version=TEST_GPAW_VERSION,
        atomic_data=(
            GpawAtomicDataInfo(
                element_symbol="Si",
                paw_setup_sha256_digest=TEST_SILICON_SETUP_DIGEST,
                paw_setup_filename="Si.PBE.gz",
                paw_setup_kind="paw",
                valence_electron_count=4,
            ),
        ),
    )


def _mock_gpaw_chargemol_task_document() -> GpawChargemolTaskDocument:
    structure = deepcopy(TEST_SI_STRUCTURE)
    return GpawChargemolTaskDocument(
        dir_name=TEST_MOCK_GPAW_CHARGEMOL_DIR_NAME,
        errors="",
        properties=build_single_point_properties_for_test(structure),
        structure=structure,
        chargemol_run_info=build_successful_process_run_info_for_test(),
    )


def _mock_failed_gpaw_chargemol_task_document() -> GpawChargemolTaskDocument:
    return GpawChargemolTaskDocument(
        dir_name=TEST_MOCK_GPAW_CHARGEMOL_DIR_NAME,
        errors="chargemol failed",
        properties=None,
        structure=None,
        chargemol_run_info=None,
    )


def _build_mock_gpaw_scf_result_with_total_density_cube(
    *, structure: PMG_Structure, run_directory: Path, **_: object
) -> GpawScfRunResult:
    (run_directory / GPAW_TOTAL_DENSITY_CUBE_FILENAME).write_text("cube")
    return GpawScfRunResult(
        properties=build_single_point_properties_for_test(structure),
        runtime_provenance=_build_runtime_provenance(),
    )


@patch(
    "crystal.calculators.gpaw.scf_chargemol_maker.GpawChargemolMaker.run_chargemol_workflow"
)
@patch("crystal.calculators.gpaw.scf_chargemol_maker.run_gpaw_scf")
def test_gpaw_scf_chargemol_maker_runs_job_and_returns_gpaw_chargemol_output(
    mock_run_gpaw_scf: MagicMock, mock_run_chargemol_workflow: MagicMock, tmp_path: Path
) -> None:
    input_structure = deepcopy(TEST_SI_STRUCTURE)
    mock_run_gpaw_scf.side_effect = _build_mock_gpaw_scf_result_with_total_density_cube
    mock_run_chargemol_workflow.return_value = _mock_gpaw_chargemol_task_document()
    maker_settings = GpawScfChargemolMakerSettings(
        simulation_settings=GpawScfChargemolSimulationSettings(
            gpaw_scf_simulation_settings=GpawScfSimulationSettings(),
            gpaw_chargemol_simulation_settings=GpawChargemolSimulationSettings(),
        ),
        executor_settings=ExecutorSettings(
            number_of_processes=TEST_GPAW_SCF_AND_CHARGEMOL_EXECUTOR_PROCESSES
        ),
    )
    maker = GpawScfChargemolMaker(
        simulation_settings=maker_settings.simulation_settings,
        executor_settings=maker_settings.executor_settings,
    )
    job = maker.make(structure=input_structure)
    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)
    mock_run_gpaw_scf.assert_called_once()
    mock_run_chargemol_workflow.assert_called_once()
    chargemol_workflow_arguments = mock_run_chargemol_workflow.call_args.kwargs
    assert chargemol_workflow_arguments["structure"] == input_structure
    assert chargemol_workflow_arguments["properties"] == (
        build_single_point_properties_for_test(input_structure)
    )
    assert "runtime_provenance" not in chargemol_workflow_arguments
    assert "gpaw_scf_simulation_settings" not in chargemol_workflow_arguments
    output_doc = run_job[job.uuid][1].output
    assert isinstance(output_doc, GpawScfChargemolTaskDocument)
    assert output_doc.converged
    assert output_doc.properties is not None
    assert output_doc.structure is not None
    assert output_doc.chargemol_run_info is not None
    assert output_doc.gpaw_scf_simulation_settings == (
        maker_settings.simulation_settings.gpaw_scf_simulation_settings
    )
    assert output_doc.runtime_provenance == _build_runtime_provenance()
    assert output_doc.dir_name == TEST_MOCK_GPAW_CHARGEMOL_DIR_NAME
    assert len(output_doc.structure) == len(TEST_SI_STRUCTURE)
    assert output_doc.structure.formula == TEST_SI_STRUCTURE.formula
    assert not list(tmp_path.rglob(GPAW_TOTAL_DENSITY_CUBE_FILENAME))


@patch(
    "crystal.calculators.gpaw.scf_chargemol_maker.GpawChargemolMaker.run_chargemol_workflow"
)
@patch("crystal.calculators.gpaw.scf_chargemol_maker.run_gpaw_scf")
def test_gpaw_scf_chargemol_maker_skips_chargemol_when_scf_fails(
    mock_run_gpaw_scf: MagicMock, mock_run_chargemol_workflow: MagicMock, tmp_path: Path
) -> None:
    mock_run_gpaw_scf.side_effect = RuntimeError("scf failed")
    maker = GpawScfChargemolMaker(
        simulation_settings=GpawScfChargemolSimulationSettings(
            gpaw_scf_simulation_settings=GpawScfSimulationSettings(),
            gpaw_chargemol_simulation_settings=GpawChargemolSimulationSettings(),
        ),
        executor_settings=ExecutorSettings(
            number_of_processes=TEST_GPAW_SCF_AND_CHARGEMOL_EXECUTOR_PROCESSES
        ),
    )

    job = maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)

    mock_run_gpaw_scf.assert_called_once()
    mock_run_chargemol_workflow.assert_not_called()
    output_doc = run_job[job.uuid][1].output
    assert not output_doc.converged
    assert output_doc.errors != ""
    assert output_doc.properties is None
    assert output_doc.structure is None
    assert output_doc.chargemol_run_info is None
    assert output_doc.gpaw_scf_simulation_settings is None
    assert output_doc.runtime_provenance is None


@patch(
    "crystal.calculators.gpaw.scf_chargemol_maker.GpawChargemolMaker.run_chargemol_workflow"
)
@patch("crystal.calculators.gpaw.scf_chargemol_maker.run_gpaw_scf")
def test_gpaw_scf_chargemol_maker_retains_total_density_cube_on_chargemol_failure(
    mock_run_gpaw_scf: MagicMock, mock_run_chargemol_workflow: MagicMock, tmp_path: Path
) -> None:
    mock_run_gpaw_scf.side_effect = _build_mock_gpaw_scf_result_with_total_density_cube
    mock_run_chargemol_workflow.return_value = (
        _mock_failed_gpaw_chargemol_task_document()
    )
    maker = GpawScfChargemolMaker(
        simulation_settings=GpawScfChargemolSimulationSettings(
            gpaw_scf_simulation_settings=GpawScfSimulationSettings(),
            gpaw_chargemol_simulation_settings=GpawChargemolSimulationSettings(),
        )
    )

    job = maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)

    output_doc = run_job[job.uuid][1].output
    assert not output_doc.converged
    assert list(tmp_path.rglob(GPAW_TOTAL_DENSITY_CUBE_FILENAME))
