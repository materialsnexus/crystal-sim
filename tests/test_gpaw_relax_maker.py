"""Tests for GPAW relax maker behavior."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

from jobflow.managers.local import run_locally

from crystal.calculators.gpaw.relax_maker import GpawRelaxMaker
from crystal.constants.fireworks import (
    QUEUEADAPTER_MEM_KEY,
    QUEUEADAPTER_NTASKS_KEY,
    QUEUEADAPTER_SPEC_KEY,
    QUEUEADAPTER_WALLTIME_KEY,
)
from crystal.schemas.gpaw import (
    GpawRelaxSimulationSettings,
    GpawRelaxTaskDocument,
    GpawTaskStatus,
)
from crystal.schemas.properties import RelaxProperties, StepProperties
from crystal.schemas.relax_settings import OptimizerType
from crystal.schemas.scratch_execution import LocalScratchPolicy
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.constants.queueadapter import (
    QUEUEADAPTER_EXPECTED_MEM_GB,
    QUEUEADAPTER_EXPECTED_WALLTIME,
    QUEUEADAPTER_TEST_MEM_GB,
    QUEUEADAPTER_TEST_WALLTIME_MIN,
)
from tests.helpers.constants.structure import (
    MOCK_ERROR_MESSAGE,
    TEST_RELAX_ENERGY_LOW,
    TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
    TEST_RELAX_MAX_STEPS,
    TEST_RELAX_STEP_INDEX,
    TEST_RELAX_STRESS_VECTOR_ZERO,
    TEST_RELAX_ZERO_FORCE_VECTOR,
    TEST_SI_STRUCTURE,
)

TEST_GPAW_RELAX_EXECUTOR_PROCESS_COUNT = 3


def _build_mock_gpaw_relax_properties() -> RelaxProperties:
    return RelaxProperties(
        relax_steps=[
            StepProperties(
                step=TEST_RELAX_STEP_INDEX,
                structure=deepcopy(TEST_SI_STRUCTURE),
                forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in TEST_SI_STRUCTURE],
                stress=deepcopy(TEST_RELAX_STRESS_VECTOR_ZERO),
                energy=TEST_RELAX_ENERGY_LOW,
            )
        ]
    )


@patch("crystal.calculators.gpaw.relax_maker.run_gpaw_relax")
def test_mock_gpaw_relax_calculation(
    mock_run_gpaw_relax: MagicMock, tmp_path: Path
) -> None:
    expected_properties = _build_mock_gpaw_relax_properties()
    mock_run_gpaw_relax.return_value = expected_properties
    simulation_settings = GpawRelaxSimulationSettings(
        optimizer_type=OptimizerType.BFGS,
        max_steps=TEST_RELAX_MAX_STEPS,
        force_threshold_in_ev_per_angstrom=TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
    )
    executor_settings = ExecutorSettings(
        number_of_processes=TEST_GPAW_RELAX_EXECUTOR_PROCESS_COUNT,
        total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB,
        maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN,
    )
    maker = GpawRelaxMaker(
        simulation_settings=simulation_settings, executor_settings=executor_settings
    )
    job = maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    assert (
        job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY]
        == TEST_GPAW_RELAX_EXECUTOR_PROCESS_COUNT
    )
    assert (
        job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_MEM_KEY]
        == QUEUEADAPTER_EXPECTED_MEM_GB
    )
    assert (
        job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_WALLTIME_KEY]
        == QUEUEADAPTER_EXPECTED_WALLTIME
    )

    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)
    task_document = run_job[job.uuid][1].output
    assert isinstance(task_document, GpawRelaxTaskDocument)
    assert task_document.status is GpawTaskStatus.success
    run_directory = Path(task_document.dir_name).resolve()
    assert run_directory.is_dir()
    assert run_directory.is_relative_to(tmp_path.resolve())
    assert task_document.converged
    assert task_document.errors == ""
    properties = task_document.properties
    assert isinstance(properties, RelaxProperties)
    assert len(properties.relax_steps) == len(expected_properties.relax_steps)
    assert properties.final_energy == expected_properties.final_energy
    assert (
        properties.relaxed_structure.composition
        == expected_properties.relaxed_structure.composition
    )


@patch("crystal.calculators.gpaw.relax_maker.run_gpaw_relax")
def test_gpaw_relax_maker_passes_local_scratch_policy_to_runner(
    mock_run_gpaw_relax: MagicMock, tmp_path: Path
) -> None:
    expected_properties = _build_mock_gpaw_relax_properties()
    local_scratch_policy = LocalScratchPolicy(
        enabled=True, fallback_scratch_parent=tmp_path / "scratch"
    )
    mock_run_gpaw_relax.return_value = expected_properties
    maker = GpawRelaxMaker(
        simulation_settings=GpawRelaxSimulationSettings(
            optimizer_type=OptimizerType.BFGS,
            max_steps=TEST_RELAX_MAX_STEPS,
            force_threshold_in_ev_per_angstrom=(
                TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM
            ),
        ),
        executor_settings=ExecutorSettings(local_scratch_policy=local_scratch_policy),
    )
    job = maker.make(structure=deepcopy(TEST_SI_STRUCTURE))

    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)

    task_document = run_job[job.uuid][1].output
    assert isinstance(task_document, GpawRelaxTaskDocument)
    run_directory = Path(task_document.dir_name).resolve()
    assert run_directory.is_dir()
    assert run_directory.is_relative_to(tmp_path.resolve())
    assert (
        mock_run_gpaw_relax.call_args.kwargs["local_scratch_policy"]
        == local_scratch_policy
    )


@patch("crystal.calculators.gpaw.relax_maker.run_gpaw_relax")
def test_unsuccessful_mock_gpaw_relax_calculation(
    mock_run_gpaw_relax: MagicMock, tmp_path: Path
) -> None:
    mock_run_gpaw_relax.side_effect = RuntimeError(MOCK_ERROR_MESSAGE)
    simulation_settings = GpawRelaxSimulationSettings(
        optimizer_type=OptimizerType.BFGS,
        max_steps=TEST_RELAX_MAX_STEPS,
        force_threshold_in_ev_per_angstrom=TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
    )
    maker = GpawRelaxMaker(simulation_settings=simulation_settings)
    job = maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)
    task_document = run_job[job.uuid][1].output
    assert isinstance(task_document, GpawRelaxTaskDocument)
    assert task_document.status is GpawTaskStatus.failed
    run_directory = Path(task_document.dir_name).resolve()
    assert run_directory.is_dir()
    assert run_directory.is_relative_to(tmp_path.resolve())
    assert not task_document.converged
    assert task_document.properties is None
    assert MOCK_ERROR_MESSAGE in task_document.errors
