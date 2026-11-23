"""Tests for MLIP relax maker."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from jobflow.core.job import Job
from jobflow.managers.local import run_locally

from crystal.calculators.mlip.relax_maker import MlipRelaxMaker
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.mlip import MlipRelaxSimulationSettings, MlipRelaxTaskDocument
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.properties import RelaxProperties
from crystal.schemas.relax_settings import OptimizerType
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.constants.mlip import MOCK_TORCH, TEST_MLIP_RELAX_SIMULATION_SETTINGS
from tests.helpers.constants.properties import MOCK_MLIP_RELAX_PROPERTIES
from tests.helpers.constants.structure import TEST_SI_STRUCTURE


@pytest.mark.parametrize(
    ("registry", "final_energy"),
    [
        (QetModelRegistry.foundational_matq, -5.65),
        (QetModelRegistry.foundational_matpes_pbe, -5.94),
        (QetModelRegistry.finetuned_linbosi, -4.279855),
        (QetModelRegistry.finetuned_litaosi, -5.945069),
    ],
)
@run_in_temp_dir
def test_run_mlip_relax_maker(registry: QetModelRegistry, final_energy: float) -> None:
    simulation_settings = MlipRelaxSimulationSettings(
        optimizer_type=OptimizerType.BFGS,
        max_steps=2,
        force_threshold_in_ev_per_angstrom=0.0,
        model=MlipModelRef(registry_model=registry),
    )
    relax_maker = MlipRelaxMaker(simulation_settings=simulation_settings)
    structure = deepcopy(TEST_SI_STRUCTURE)
    job = relax_maker.make(structure=structure)
    assert isinstance(job, Job)

    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    assert isinstance(run_job, dict)

    relax_doc = run_job[job.uuid][1].output
    assert isinstance(relax_doc, MlipRelaxTaskDocument)

    relax_properties = relax_doc.properties
    assert isinstance(relax_properties, RelaxProperties)
    assert math.isclose(relax_properties.final_energy, final_energy, abs_tol=1e-1)
    assert len(relax_properties.relax_steps) == 3


@patch("crystal.calculators.mlip.relax_maker.run_mlip_relax")
@run_in_temp_dir
def test_mlip_relax_maker_sets_cuda_device_when_gpu_requested(
    mock_run_mlip_relax: Mock,
) -> None:
    mock_run_mlip_relax.return_value = MOCK_MLIP_RELAX_PROPERTIES
    relax_maker = MlipRelaxMaker(
        simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS,
        executor_settings=ExecutorSettings(number_of_gpus=1),
    )
    mock_torch = deepcopy(MOCK_TORCH)
    with patch.dict("sys.modules", {"torch": mock_torch}):
        job = relax_maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
        run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)

    relax_doc = run_job[job.uuid][1].output
    assert isinstance(relax_doc, MlipRelaxTaskDocument)
    assert isinstance(relax_doc.properties, RelaxProperties)
    mock_torch.set_default_device.assert_called_once_with("cuda")
