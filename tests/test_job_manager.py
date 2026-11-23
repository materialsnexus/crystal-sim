import time
from pathlib import Path

import pytest
from fireworks.core.fworker import FWorker
from fireworks.core.launchpad import LaunchPad
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from jobflow.core.flow import Flow
from jobflow.core.job import job
from jobflow.core.maker import Maker
from jobflow.core.store import JobStore

from crystal import (
    LOCAL_JOBSTORE_YAML,
    LOCAL_LAUNCHPAD_YAML,
    SLURM_JOBSTORE_YAML,
    SLURM_LAUNCHPAD_YAML,
)
from crystal.constants.fireworks import (
    QUEUEADAPTER_GRES_KEY,
    QUEUEADAPTER_MEM_KEY,
    QUEUEADAPTER_NTASKS_KEY,
    QUEUEADAPTER_SPEC_KEY,
    QUEUEADAPTER_WALLTIME_KEY,
)
from crystal.firework_helpers import run_queue_launcher, run_rapidfire_launcher
from crystal.helpers.fireworks.manager import JobManager
from crystal.helpers.fireworks.slurm_settings import QueueAdapterSettingsMapping
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.constants.queueadapter import (
    QUEUEADAPTER_EXPECTED_MEM_GB,
    QUEUEADAPTER_EXPECTED_WALLTIME,
    QUEUEADAPTER_TEST_MEM_GB,
    QUEUEADAPTER_TEST_WALLTIME_MIN,
)
from tests.helpers.test_paths import (
    TEST_JOBSTORE_CONFIG_MISSING_NAME,
    TEST_LAUNCHPAD_CONFIG_MISSING_NAME,
)

TEST_FIREWORKER = FWorker(name="fireworker", query={})


class SleepMaker(Maker):
    name: str = "sleeper"

    @job
    def make(self, sleep_time: float = 120.0) -> int:
        time.sleep(sleep_time)
        return int(sleep_time)


@job
def example_double_job(number: int) -> int:
    return 2 * number


class AlternateSleepMaker(Maker):
    name: str = "alternate_sleeper"

    @job
    def make(self, sleep_time: float = 120.0) -> int:
        time.sleep(sleep_time)
        return int(sleep_time)


def test_job_manager_fails_in_absence_of_config_files() -> None:
    with pytest.raises(FileNotFoundError, match=TEST_LAUNCHPAD_CONFIG_MISSING_NAME):
        _ = JobManager.from_configs(
            fireworker=FWorker(name="worker_1"),
            launchpad_config=Path.cwd() / TEST_LAUNCHPAD_CONFIG_MISSING_NAME,
            jobstore_config=LOCAL_JOBSTORE_YAML,
        )

    with pytest.raises(FileNotFoundError, match=TEST_JOBSTORE_CONFIG_MISSING_NAME):
        _ = JobManager.from_configs(
            fireworker=FWorker(name="worker_1"),
            launchpad_config=LOCAL_LAUNCHPAD_YAML,
            jobstore_config=Path.cwd() / TEST_JOBSTORE_CONFIG_MISSING_NAME,
        )


@pytest.mark.needs_mongodb
@run_in_temp_dir
def test_job_runs_with_fireworks(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    running_fireworker = FWorker(name="running-fireworker", query={})
    maker = SleepMaker()
    job_1 = maker.make(sleep_time=1)
    manager = JobManager(
        fireworker=running_fireworker, launchpad=mock_launchpad, jobstore=mock_jobstore
    )
    [firework_id] = manager.add_to_launchpad(job_1)
    run_rapidfire_launcher(mock_launchpad, running_fireworker)
    time.sleep(3)
    uuid = manager.launchpad.get_fw_by_id(firework_id).spec["_tasks"][0]["job"].uuid
    [job_doc] = list(manager.jobstore.docs_store.query({"uuid": uuid}))
    assert job_doc["output"] == 1

    [launch] = manager.launchpad.launches.find({"fw_id": firework_id})
    fworker = launch["fworker"]
    assert fworker["name"] == running_fireworker.name


@pytest.mark.needs_mongodb
@pytest.mark.needs_slurm
@run_in_temp_dir
def test_job_runs_with_slurm(default_slurm_queue_adapter: CommonAdapter) -> None:
    # Needs to be tested over slurm
    fireworker = FWorker(name="fireworker", query={})
    maker = SleepMaker()
    job_1 = maker.make(sleep_time=1)

    manager = JobManager.from_configs(
        fireworker=fireworker,
        launchpad_config=SLURM_LAUNCHPAD_YAML,
        jobstore_config=SLURM_JOBSTORE_YAML,
    )
    [firework_id] = manager.add_to_launchpad(job_1)
    run_queue_launcher(
        manager.launchpad, fireworker, qadapter=default_slurm_queue_adapter
    )
    time.sleep(2)
    firework = manager.launchpad.get_fw_by_id(firework_id)
    assert firework.state == "COMPLETED"


@run_in_temp_dir
def test_run_job_with_different_runtime_settings(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    maker = SleepMaker()
    fast_job = maker.make(sleep_time=1)

    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FIREWORKER
    )
    [fast_firework_id] = manager.add_to_launchpad(
        fast_job, queueadapter_settings=ExecutorSettings(number_of_processes=4)
    )
    slow_job = maker.make(sleep_time=2)
    [slow_firework_id] = manager.add_to_launchpad(
        slow_job, queueadapter_settings=ExecutorSettings(number_of_processes=8)
    )
    fast_firework = manager.launchpad.get_fw_by_id(fast_firework_id)
    slow_firework = manager.launchpad.get_fw_by_id(slow_firework_id)

    assert QUEUEADAPTER_SPEC_KEY in slow_firework.spec
    assert slow_firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 8
    assert QUEUEADAPTER_SPEC_KEY in fast_firework.spec
    assert fast_firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 4


@run_in_temp_dir
def test_job_manager_accepts_executor_settings_for_queueadapter(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    maker = SleepMaker()
    job = maker.make(sleep_time=1)
    executor_settings = ExecutorSettings(
        number_of_processes=4,
        total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB,
        maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN,
    )
    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FIREWORKER
    )

    [firework_id] = manager.add_to_launchpad(
        job, queueadapter_settings=executor_settings
    )
    firework = manager.launchpad.get_fw_by_id(firework_id)

    assert QUEUEADAPTER_SPEC_KEY in firework.spec
    assert firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 4
    assert (
        firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_MEM_KEY]
        == QUEUEADAPTER_EXPECTED_MEM_GB
    )
    assert (
        firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_WALLTIME_KEY]
        == QUEUEADAPTER_EXPECTED_WALLTIME
    )


@run_in_temp_dir
def test_job_manager_accepts_gpu_queueadapter_settings(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    maker = SleepMaker()
    job = maker.make(sleep_time=1)
    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FIREWORKER
    )

    [firework_id] = manager.add_to_launchpad(
        job,
        queueadapter_settings=ExecutorSettings(number_of_processes=4, number_of_gpus=2),
    )
    firework = manager.launchpad.get_fw_by_id(firework_id)

    assert firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_GRES_KEY] == "gpu:2"


@run_in_temp_dir
def test_run_flow_with_independent_jobs_local(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    maker = SleepMaker()
    fast_job = maker.make(sleep_time=1)
    slow_job = maker.make(sleep_time=2)

    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FIREWORKER
    )
    # independent jobs
    flow = Flow(jobs=[fast_job, slow_job], name="two independent sleeper")

    queueadapter_settings_mapping = {"sleeper": ExecutorSettings(number_of_processes=8)}
    firework_ids = manager.add_to_launchpad(
        flow, queueadapter_settings=queueadapter_settings_mapping
    )
    run_rapidfire_launcher(mock_launchpad, TEST_FIREWORKER)
    time.sleep(3)
    for firework_id in firework_ids:
        firework = manager.launchpad.get_fw_by_id(firework_id)
        assert firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 8


@run_in_temp_dir
def test_flow_manager_accepts_executor_settings_mapping(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    maker = SleepMaker()
    fast_job = maker.make(sleep_time=1)
    slow_job = maker.make(sleep_time=2)
    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FIREWORKER
    )
    flow = Flow(jobs=[fast_job, slow_job], name="two independent sleepers")
    executor_settings = ExecutorSettings(
        number_of_processes=6,
        total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB,
        maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN,
    )
    queueadapter_settings_mapping = {"sleeper": executor_settings}

    firework_ids = manager.add_to_launchpad(
        flow, queueadapter_settings=queueadapter_settings_mapping
    )

    for firework_id in firework_ids:
        firework = manager.launchpad.get_fw_by_id(firework_id)
        assert firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 6
        assert (
            firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_MEM_KEY]
            == QUEUEADAPTER_EXPECTED_MEM_GB
        )
        assert (
            firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_WALLTIME_KEY]
            == QUEUEADAPTER_EXPECTED_WALLTIME
        )


@run_in_temp_dir
def test_flow_manager_accepts_named_queueadapter_settings_mapping(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    fast_job = SleepMaker().make(sleep_time=1)
    slow_job = AlternateSleepMaker().make(sleep_time=2)
    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FIREWORKER
    )
    flow = Flow(jobs=[fast_job, slow_job], name="mixed sleeper settings")
    queueadapter_settings_mapping: QueueAdapterSettingsMapping = {
        "sleeper": ExecutorSettings(number_of_processes=8),
        "alternate_sleeper": ExecutorSettings(
            number_of_processes=6,
            total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB,
            maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN,
        ),
    }

    firework_ids = manager.add_to_launchpad(
        flow, queueadapter_settings=queueadapter_settings_mapping
    )

    for firework_id in firework_ids:
        firework = manager.launchpad.get_fw_by_id(firework_id)
        if firework.name == "sleeper":
            assert firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 8
        if firework.name == "alternate_sleeper":
            assert firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 6
            assert (
                firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_MEM_KEY]
                == QUEUEADAPTER_EXPECTED_MEM_GB
            )
            assert (
                firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_WALLTIME_KEY]
                == QUEUEADAPTER_EXPECTED_WALLTIME
            )


@pytest.mark.needs_mongodb
@pytest.mark.needs_slurm
@run_in_temp_dir
def test_run_flow_with_independent_jobs_in_slurm(
    default_slurm_queue_adapter: CommonAdapter,
) -> None:
    # test in slurm
    maker = SleepMaker()
    fast_job = maker.make(sleep_time=1)
    slow_job = maker.make(sleep_time=2)

    manager = JobManager.from_configs(
        fireworker=TEST_FIREWORKER,
        launchpad_config=SLURM_LAUNCHPAD_YAML,
        jobstore_config=SLURM_JOBSTORE_YAML,
    )
    # independent jobs
    flow = Flow(jobs=[fast_job, slow_job], name="two independent sleeper")
    queueadapter_settings_mapping = {"sleeper": ExecutorSettings(number_of_processes=8)}
    firework_ids = manager.add_to_launchpad(
        flow, queueadapter_settings=queueadapter_settings_mapping
    )
    run_queue_launcher(
        manager.launchpad, TEST_FIREWORKER, qadapter=default_slurm_queue_adapter
    )
    for firework_id in firework_ids:
        firework = manager.launchpad.get_fw_by_id(firework_id)
        assert firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 8


@run_in_temp_dir
def test_run_flow_with_dependent_jobs(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    maker = SleepMaker()
    fast_job = maker.make(sleep_time=1)
    double_job = example_double_job(fast_job.output)
    slow_job = maker.make(sleep_time=double_job.output)

    manager = JobManager(
        fireworker=TEST_FIREWORKER, launchpad=mock_launchpad, jobstore=mock_jobstore
    )
    flow = Flow(
        jobs=[fast_job, double_job, slow_job],
        name="two dependent sleepers",
        output=slow_job.output,
    )
    queueadapter_settings_mapping = {"sleeper": ExecutorSettings(number_of_processes=8)}
    firework_ids = manager.add_to_launchpad(
        flow, queueadapter_settings=queueadapter_settings_mapping
    )
    run_rapidfire_launcher(mock_launchpad, TEST_FIREWORKER)
    job_outputs = []
    time.sleep(10)
    for firework_id in firework_ids:
        firework = manager.launchpad.get_fw_by_id(firework_id)
        if firework.name == "sleeper":
            assert firework.spec[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 8
        uuid = firework.spec["_tasks"][0]["job"].uuid
        [job_doc] = list(manager.jobstore.docs_store.query({"uuid": uuid}))
        job_outputs.append(job_doc["output"])
    assert sorted(job_outputs) == [1, 2, 2]
