"""Tests for the MLIP MD equilibration maker."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from jobflow.core.job import Job
from jobflow.managers.local import run_locally
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.md.equilibration_maker import MlipMdEquilibrationMaker
from crystal.calculators.md.md_runner import MlipMdRunner
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.md.checks import AverageEnergyDriftCheck
from crystal.schemas.md.mlip_md_equilibration import (
    MlipMdEquilibrationSimulationSettings,
    MlipMdEquilibrationTaskDocument,
)
from crystal.schemas.md.properties import MdEquilibrationProperties, MdStageResult
from crystal.schemas.md.settings import (
    MdDynamicsSettings,
    MdEnsemble,
    MdPotentialSettings,
    MdRunSettings,
    MdThermostatSettings,
    MdTimeSettings,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from tests.helpers.constants.md import (
    MOCK_DEFAULT_LOG_CONTENT,
    MOCK_USER_DEFINED_LOG_CONTENT,
    TEST_TRAJECTORY_PATH,
)
from tests.helpers.constants.structure import TEST_BCC_FE_STRUCTURE

DEFAULT_MD_NVT_RUN_SETTINGS = MdRunSettings(
    time=MdTimeSettings(
        time_step_fs=2.0,
        simulation_time_ps=0.004,
        trajectory_interval=2,
        property_log_interval=2,
    ),
    dynamics=MdDynamicsSettings(
        ensemble=MdEnsemble.nvt,
        thermostat=MdThermostatSettings(temperature_k=300.0, time_constant_fs=2.0),
    ),
    potential=MdPotentialSettings(
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    ),
)

DEFAULT_PATCHED_MD_STAGE_RESULT = MdStageResult(
    ensemble=MdEnsemble.nvt,
    trajectory_path=TEST_TRAJECTORY_PATH,
    default_log_path=Path("nvt_default.log"),
    user_defined_log_path=Path("nvt_user_defined.log"),
)

DEFAULT_MLIP_MD_EQUILIBRATION_MAKER = MlipMdEquilibrationMaker(
    simulation_settings=MlipMdEquilibrationSimulationSettings(
        run_settings=DEFAULT_MD_NVT_RUN_SETTINGS, checks=[], time_averaging_window=0.002
    )
)


@run_in_temp_dir
def test_mlip_md_equilibration_maker_returns_properties_and_artifacts(
    tmp_path: Path,
) -> None:
    default_log_path = tmp_path / "nvt_default.log"
    default_log_path.write_text(MOCK_DEFAULT_LOG_CONTENT, encoding="utf-8")
    user_defined_log_path = tmp_path / "nvt_user_defined.log"
    user_defined_log_path.write_text(MOCK_USER_DEFINED_LOG_CONTENT, encoding="utf-8")

    maker = MlipMdEquilibrationMaker(
        simulation_settings=MlipMdEquilibrationSimulationSettings(
            run_settings=DEFAULT_MD_NVT_RUN_SETTINGS,
            checks=[AverageEnergyDriftCheck(max_absolute_drift_slope=5.0)],
            time_averaging_window=0.002,
        )
    )
    with patch.object(
        MlipMdRunner,
        "run",
        return_value=DEFAULT_PATCHED_MD_STAGE_RESULT.model_copy(
            update={
                "default_log_path": default_log_path,
                "user_defined_log_path": user_defined_log_path,
            }
        ),
    ):
        job = maker.make(structure=deepcopy(TEST_BCC_FE_STRUCTURE))
        run_output = run_locally(job, root_dir=Path.cwd(), create_folders=True)

    task_document = run_output[job.uuid][1].output
    assert isinstance(job, Job)
    assert isinstance(task_document, MlipMdEquilibrationTaskDocument)
    assert isinstance(task_document.properties, MdEquilibrationProperties)
    assert task_document.properties.check_results[0].passed
    assert isinstance(task_document.properties.average_energy, float)
    assert task_document.properties.last_frame_structure.formula == "Li4 Nb4 O12"
    assert isinstance(task_document.properties.last_frame_structure, PMG_Structure)
    assert task_document.properties.plots.energy_plot_path is not None
    assert task_document.properties.plots.temperature_plot_path is not None
    assert task_document.properties.plots.density_plot_path is not None
    assert Path(task_document.properties.plots.energy_plot_path).is_file()
    assert Path(task_document.properties.plots.temperature_plot_path).is_file()
    assert Path(task_document.properties.plots.density_plot_path).is_file()


@run_in_temp_dir
def test_mlip_md_equilibration_maker_returns_error_document_on_runner_failure() -> None:
    maker = deepcopy(DEFAULT_MLIP_MD_EQUILIBRATION_MAKER)
    with patch.object(MlipMdRunner, "run", side_effect=RuntimeError("mock md failure")):
        job = maker.make(structure=deepcopy(TEST_BCC_FE_STRUCTURE))
        run_output = run_locally(job, root_dir=Path.cwd(), create_folders=True)

    task_document = run_output[job.uuid][1].output
    assert isinstance(task_document, MlipMdEquilibrationTaskDocument)
    assert task_document.properties is None
    assert "mock md failure" in task_document.errors


@run_in_temp_dir
def test_run_mlip_md_equilibration_maker_locally() -> None:
    maker = MlipMdEquilibrationMaker(
        simulation_settings=MlipMdEquilibrationSimulationSettings(
            run_settings=DEFAULT_MD_NVT_RUN_SETTINGS,
            checks=[AverageEnergyDriftCheck(max_absolute_drift_slope=10.0)],
            time_averaging_window=0.002,
        )
    )

    job = maker.make(structure=deepcopy(TEST_BCC_FE_STRUCTURE))
    run_output = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    task_document = run_output[job.uuid][1].output
    assert isinstance(task_document, MlipMdEquilibrationTaskDocument)
    assert isinstance(task_document.properties, MdEquilibrationProperties)
    assert Path(task_document.properties.md_stage_result.default_log_path).is_file()
    assert Path(
        task_document.properties.md_stage_result.user_defined_log_path
    ).is_file()
    assert task_document.properties.plots.energy_plot_path is not None
    assert Path(task_document.properties.plots.energy_plot_path).is_file()
    assert len(task_document.properties.check_results) == 1
    assert isinstance(task_document.properties.average_energy, float)


@pytest.mark.parametrize(
    ("structure", "trajectory_path"),
    [(deepcopy(TEST_BCC_FE_STRUCTURE), deepcopy(TEST_TRAJECTORY_PATH)), (None, None)],
)
def test_mlip_md_equilibration_raises_for_both_invalid_structure_and_trajectory_path(
    structure: PMG_Structure | None, trajectory_path: Path | None
) -> None:
    maker = deepcopy(DEFAULT_MLIP_MD_EQUILIBRATION_MAKER)
    with pytest.raises(
        ValueError,
        match="Exactly one of structure or trajectory path should be provided",
    ):
        _ = maker.make(structure=structure, trajectory_path=trajectory_path)


def test_mlip_md_equilibration_raises_for_structure_with_previous_output_trajectory_reference() -> (
    None
):
    maker = deepcopy(DEFAULT_MLIP_MD_EQUILIBRATION_MAKER)
    md_equilibration_job = maker.make(structure=deepcopy(TEST_BCC_FE_STRUCTURE))
    with pytest.raises(
        ValueError,
        match="Exactly one of structure or trajectory path should be provided",
    ):
        _ = maker.make(
            structure=deepcopy(TEST_BCC_FE_STRUCTURE),
            trajectory_path=md_equilibration_job.output.properties.md_stage_result.trajectory_path,  # type:ignore[arg-type]
        )
