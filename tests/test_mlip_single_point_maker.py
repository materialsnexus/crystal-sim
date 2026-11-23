"""Tests for MLIP single-point maker."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from jobflow.core.job import Job
from jobflow.managers.local import run_locally

from crystal.calculators.mlip.single_point_maker import MlipSinglePointMaker
from crystal.constants.fireworks import QUEUEADAPTER_GRES_KEY, QUEUEADAPTER_SPEC_KEY
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.mlip import (
    MlipSinglePointSimulationSettings,
    MlipSinglePointTaskDocument,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.properties import SinglePointProperties
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.constants.mlip import MOCK_TORCH
from tests.helpers.constants.properties import MOCK_MLIP_SINGLE_POINT_PROPERTIES
from tests.helpers.constants.structure import TEST_SI_STRUCTURE

BAD_MODEL_PATH = Path("this/model/does/not/exist")


@pytest.mark.parametrize(
    ("registry", "minimum_energy"),
    [
        (QetModelRegistry.foundational_matq, -5.58),
        (QetModelRegistry.foundational_matpes_pbe, -5.89),
        (QetModelRegistry.finetuned_linbosi, -4.184148),
        (QetModelRegistry.finetuned_litaosi, -6.061215),
    ],
)
@run_in_temp_dir
def test_run_mlip_single_point_maker_locally(
    registry: QetModelRegistry, minimum_energy: float
) -> None:
    single_point_maker = MlipSinglePointMaker(
        simulation_settings=MlipSinglePointSimulationSettings(
            model=MlipModelRef(registry_model=registry)
        )
    )
    structure = deepcopy(TEST_SI_STRUCTURE)
    job = single_point_maker.make(structure=structure)

    assert isinstance(job, Job)
    # using default memory store
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    assert isinstance(run_job, dict)
    single_point_doc = run_job[job.uuid][1].output
    assert isinstance(single_point_doc, MlipSinglePointTaskDocument)
    single_point_properties = single_point_doc.properties
    assert isinstance(single_point_properties, SinglePointProperties)
    assert math.isclose(single_point_properties.energy, minimum_energy, abs_tol=1e-1)
    assert single_point_properties.charges is not None


@run_in_temp_dir
def test_unsuccessful_mlip_calculation_bad_model() -> None:
    single_point_maker = MlipSinglePointMaker(
        simulation_settings=MlipSinglePointSimulationSettings(
            model=MlipModelRef(path=BAD_MODEL_PATH)
        )
    )

    structure = deepcopy(TEST_SI_STRUCTURE)
    job = single_point_maker.make(structure=structure)

    assert isinstance(job, Job)
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    assert isinstance(run_job, dict)
    single_point_doc = run_job[job.uuid][1].output
    assert isinstance(single_point_doc, MlipSinglePointTaskDocument)
    assert single_point_doc.properties is None
    assert (
        f"Directory {BAD_MODEL_PATH.as_posix()} does not exist"
        in single_point_doc.errors
    )


def test_mlip_single_point_maker_adds_gpu_gres_metadata() -> None:
    single_point_maker = MlipSinglePointMaker(
        simulation_settings=MlipSinglePointSimulationSettings(
            model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
        ),
        executor_settings=ExecutorSettings(number_of_gpus=1),
    )

    job = single_point_maker.make(structure=deepcopy(TEST_SI_STRUCTURE))

    assert job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_GRES_KEY] == "gpu:1"


@patch("crystal.calculators.mlip.single_point_maker.run_mlip_single_point")
@run_in_temp_dir
def test_mlip_single_point_maker_sets_cuda_device_when_gpu_requested(
    mock_run_mlip_single_point: Mock,
) -> None:
    mock_run_mlip_single_point.return_value = MOCK_MLIP_SINGLE_POINT_PROPERTIES
    single_point_maker = MlipSinglePointMaker(
        simulation_settings=MlipSinglePointSimulationSettings(
            model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
        ),
        executor_settings=ExecutorSettings(number_of_gpus=1),
    )
    mock_torch = deepcopy(MOCK_TORCH)
    with patch.dict("sys.modules", {"torch": mock_torch}):
        job = single_point_maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
        run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)

    single_point_doc = run_job[job.uuid][1].output
    assert isinstance(single_point_doc, MlipSinglePointTaskDocument)
    assert isinstance(single_point_doc.properties, SinglePointProperties)
    mock_torch.set_default_device.assert_called_once_with("cuda")
