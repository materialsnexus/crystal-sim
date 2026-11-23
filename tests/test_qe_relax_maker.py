import math
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fireworks.core.firework import Firework
from fireworks.core.fworker import FWorker
from fireworks.core.launchpad import LaunchPad
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from jobflow.core.store import JobStore
from jobflow.managers.fireworks import job_to_firework
from jobflow.managers.local import run_locally

from crystal import (
    LOCAL_FWORKER_YAML,
    SLURM_FWORKER_YAML,
    SLURM_JOBSTORE_YAML,
    SLURM_LAUNCHPAD_YAML,
)
from crystal.calculators.qe.relax_maker import QeRelaxMaker
from crystal.constants.fireworks import (
    QUEUEADAPTER_MEM_KEY,
    QUEUEADAPTER_NTASKS_KEY,
    QUEUEADAPTER_SPEC_KEY,
    QUEUEADAPTER_WALLTIME_KEY,
)
from crystal.firework_helpers import run_queue_launcher, run_rapidfire_launcher
from crystal.helpers.fireworks.manager import JobManager
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.properties import RelaxProperties
from crystal.schemas.qe import QeConfig, QeRelaxSimulationSettings, QeRelaxTaskDocument
from crystal.schemas.relax_settings import OptimizerType
from crystal.schemas.relaxation_constraints import RelaxationConstraints
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.constants.properties import MOCK_RELAX_PROPERTIES
from tests.helpers.constants.queueadapter import (
    QUEUEADAPTER_EXPECTED_MEM_GB,
    QUEUEADAPTER_EXPECTED_WALLTIME,
    QUEUEADAPTER_TEST_MEM_GB,
    QUEUEADAPTER_TEST_WALLTIME_MIN,
)
from tests.helpers.constants.structure import MOCK_ERROR_MESSAGE, TEST_SI_STRUCTURE
from tests.helpers.test_paths import TEST_SI_PSEUDOPOTENTIAL_FILENAME

TEST_QE_RELAX_SIMULATION_SETTINGS = QeRelaxSimulationSettings(
    config=QeConfig(),
    pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME},
    optimizer_type=OptimizerType.BFGS,
    max_steps=2,
    force_threshold_in_ev_per_angstrom=2,
)

TEST_QE_RELAX_SIMULATION_SETTINGS_WITH_CONSTRAINTS = QeRelaxSimulationSettings(
    config=QeConfig(),
    pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME},
    optimizer_type=OptimizerType.BFGS,
    max_steps=2,
    force_threshold_in_ev_per_angstrom=2,
    constraints=RelaxationConstraints(fix_cell=False),
)


@pytest.mark.needs_third_party_executables
@run_in_temp_dir
@pytest.mark.parametrize(
    ("simulation_settings", "expected_minimum_energy"),
    [
        (TEST_QE_RELAX_SIMULATION_SETTINGS, -304.008),
        (TEST_QE_RELAX_SIMULATION_SETTINGS_WITH_CONSTRAINTS, -303.926),
    ],
)
def test_qe_relax_calculation_with_firework_local(
    simulation_settings: QeRelaxSimulationSettings,
    expected_minimum_energy: float,
    mock_launchpad: LaunchPad,
    mock_jobstore: JobStore,
) -> None:
    maker = QeRelaxMaker(
        simulation_settings=simulation_settings,
        executor_settings=ExecutorSettings(number_of_processes=4),
    )
    structure = deepcopy(TEST_SI_STRUCTURE)
    job = maker.make(structure=structure)
    manager = JobManager(
        launchpad=mock_launchpad,
        jobstore=mock_jobstore,
        fireworker=FWorker.from_file(LOCAL_FWORKER_YAML.as_posix()),
    )
    [firework_id] = manager.add_to_launchpad(job, queueadapter_settings=None)
    run_rapidfire_launcher(mock_launchpad, manager.fireworker)
    fw_from_db = manager.launchpad.get_fw_by_id(firework_id)
    job = fw_from_db.spec["_tasks"][0]["job"]
    if not isinstance(job, dict):
        job = job.as_dict()
    uuid = job["uuid"]
    [job_doc] = list(manager.jobstore.docs_store.query({"uuid": uuid}))
    job_task_doc = QeRelaxTaskDocument.model_validate(job_doc["output"])
    assert job_task_doc.converged
    relax_output = job_task_doc.properties
    assert isinstance(relax_output, RelaxProperties)
    assert math.isclose(
        relax_output.final_energy, expected_minimum_energy, abs_tol=1e-1
    )


@patch("crystal.schemas.qe.extract_error_messages")
@patch("crystal.calculators.qe.relax_maker.run_qe_relax")
@run_in_temp_dir
def test_mock_qe_calculation(
    mock_run: MagicMock, mock_error_messages: MagicMock
) -> None:
    mock_run.return_value = MOCK_RELAX_PROPERTIES
    mock_error_messages.return_value = []

    executor_settings = ExecutorSettings(
        number_of_processes=4,
        total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB,
        maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN,
    )
    maker = QeRelaxMaker(
        simulation_settings=TEST_QE_RELAX_SIMULATION_SETTINGS,
        executor_settings=executor_settings,
    )
    structure = deepcopy(TEST_SI_STRUCTURE)
    job = maker.make(structure=structure)
    assert job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 4
    assert (
        job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_MEM_KEY]
        == QUEUEADAPTER_EXPECTED_MEM_GB
    )
    assert (
        job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_WALLTIME_KEY]
        == QUEUEADAPTER_EXPECTED_WALLTIME
    )
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    assert isinstance(run_job, dict)
    relax_doc = run_job[job.uuid][1].output
    assert isinstance(relax_doc, QeRelaxTaskDocument)
    assert relax_doc.converged
    relax_properties = relax_doc.properties
    assert isinstance(relax_properties, RelaxProperties)
    assert relax_properties.final_energy == 0.2


@patch("crystal.schemas.qe.extract_error_messages")
@patch("crystal.calculators.qe.relax_maker.run_qe_relax")
@run_in_temp_dir
def test_unsuccessful_mock_qe_calculation(
    mock_run: MagicMock, mock_error_messages: MagicMock
) -> None:
    mock_run.side_effect = RuntimeError("Mock error")
    mock_error_messages.return_value = [MOCK_ERROR_MESSAGE]

    maker = QeRelaxMaker(
        simulation_settings=TEST_QE_RELAX_SIMULATION_SETTINGS,
        executor_settings=ExecutorSettings(number_of_processes=4),
    )
    structure = deepcopy(TEST_SI_STRUCTURE)
    job = maker.make(structure=structure)
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    assert isinstance(run_job, dict)
    relax_doc = run_job[job.uuid][1].output
    assert isinstance(relax_doc, QeRelaxTaskDocument)
    assert not relax_doc.converged
    assert MOCK_ERROR_MESSAGE in relax_doc.errors


def test_qe_relax_job_firework_roundtrip_with_pseudopotential_directory() -> None:
    simulation_settings = QeRelaxSimulationSettings(
        config=QeConfig(),
        pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME},
        set_pseudopotential_directory=Path.cwd(),
        optimizer_type=OptimizerType.BFGS,
        max_steps=2,
        force_threshold_in_ev_per_angstrom=2,
    )
    maker = QeRelaxMaker(simulation_settings=simulation_settings)
    job = maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    firework = job_to_firework(job)
    roundtrip_firework = Firework.from_dict(firework.to_dict())
    assert roundtrip_firework.name == firework.name


@pytest.mark.needs_slurm
@pytest.mark.needs_mongodb
@pytest.mark.needs_third_party_executables
@run_in_temp_dir
def test_qe_relax_calculation_with_firework_on_slurm(
    default_slurm_queue_adapter: CommonAdapter,
) -> None:
    maker = QeRelaxMaker(simulation_settings=TEST_QE_RELAX_SIMULATION_SETTINGS)
    structure = deepcopy(TEST_SI_STRUCTURE)
    job = maker.make(structure=structure)
    manager = JobManager.from_configs(
        launchpad_config=SLURM_LAUNCHPAD_YAML,
        jobstore_config=SLURM_JOBSTORE_YAML,
        fireworker=FWorker.from_file(SLURM_FWORKER_YAML.as_posix()),
    )
    # for safety and checking if launchpad is running
    manager.launchpad.reset(password="", require_password=False)
    [firework_id] = manager.add_to_launchpad(
        job, queueadapter_settings=ExecutorSettings(number_of_processes=1)
    )
    run_queue_launcher(
        manager.launchpad, manager.fireworker, qadapter=default_slurm_queue_adapter
    )
    time.sleep(3)
    firework = manager.launchpad.get_fw_by_id(firework_id)
    assert firework.state in ("WAITING", "RUNNING", "COMPLETED", "RESERVED")
