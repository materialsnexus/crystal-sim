import pytest

from crystal.constants.fireworks import QUEUEADAPTER_GRES_KEY, QUEUEADAPTER_SPEC_KEY
from crystal.helpers.fireworks.slurm_settings import SlurmSettingsGenerator
from crystal.schemas.fireworks import PartitionType
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.constants.queueadapter import (
    QUEUEADAPTER_EXPECTED_MEM_GB,
    QUEUEADAPTER_EXPECTED_MEM_GB_ROUND,
    QUEUEADAPTER_EXPECTED_WALLTIME_ROUND,
    QUEUEADAPTER_TEST_MEM_GB,
    QUEUEADAPTER_TEST_MEM_GB_ROUND,
    QUEUEADAPTER_TEST_WALLTIME_MIN_ROUND,
)


def test_slurm_settings_from_executor_settings() -> None:
    executor_settings = ExecutorSettings(
        number_of_processes=8,
        total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB_ROUND,
        maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN_ROUND,
    )

    slurm_settings = SlurmSettingsGenerator(
        executor_settings=executor_settings
    ).settings

    assert slurm_settings.ntasks == 8
    assert slurm_settings.mem == QUEUEADAPTER_EXPECTED_MEM_GB_ROUND
    assert slurm_settings.walltime == QUEUEADAPTER_EXPECTED_WALLTIME_ROUND


def test_build_queueadapter_metadata_from_executor_settings() -> None:
    executor_settings = ExecutorSettings(
        number_of_processes=8,
        total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB_ROUND,
        maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN_ROUND,
    )
    slurm_settings = SlurmSettingsGenerator(
        executor_settings=executor_settings
    ).settings
    queueadapter_spec = slurm_settings.to_queueadapter_spec()

    assert QUEUEADAPTER_SPEC_KEY in queueadapter_spec
    assert queueadapter_spec[QUEUEADAPTER_SPEC_KEY]["ntasks"] == 8
    assert (
        queueadapter_spec[QUEUEADAPTER_SPEC_KEY]["mem"]
        == QUEUEADAPTER_EXPECTED_MEM_GB_ROUND
    )
    assert (
        queueadapter_spec[QUEUEADAPTER_SPEC_KEY]["walltime"]
        == QUEUEADAPTER_EXPECTED_WALLTIME_ROUND
    )


def test_slurm_settings_adds_gres_when_gpus_requested() -> None:
    slurm_settings = SlurmSettingsGenerator(
        executor_settings=ExecutorSettings(number_of_gpus=2)
    ).settings

    assert slurm_settings.gres == "gpu:2"


def test_slurm_settings_uses_gpu_partition_when_gpus_requested() -> None:
    slurm_settings = SlurmSettingsGenerator(
        executor_settings=ExecutorSettings(number_of_gpus=2)
    ).settings

    assert slurm_settings.queue is PartitionType.gpu


def test_queueadapter_metadata_omits_gres_when_no_gpus_requested() -> None:
    queueadapter_spec = SlurmSettingsGenerator(
        executor_settings=ExecutorSettings(number_of_gpus=0)
    ).settings.to_queueadapter_spec()

    assert QUEUEADAPTER_GRES_KEY not in queueadapter_spec[QUEUEADAPTER_SPEC_KEY]


@pytest.mark.parametrize(
    ("maximum_walltime_in_minutes", "expected_walltime"),
    [(0.0, "00:00:00"), (0.01, "00:00:01"), (59.999, "01:00:00")],
)
def test_format_slurm_walltime_from_minutes(
    maximum_walltime_in_minutes: float, expected_walltime: str
) -> None:
    executor_settings = ExecutorSettings(
        number_of_processes=8,
        total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB_ROUND,
        maximum_walltime_in_minutes=maximum_walltime_in_minutes,
    )
    slurm_settings = SlurmSettingsGenerator(
        executor_settings=executor_settings
    ).settings
    assert slurm_settings.walltime == expected_walltime


@pytest.mark.parametrize(
    ("total_memory_in_gb", "expected_memory"),
    [
        (16.0, "16G"),
        (QUEUEADAPTER_TEST_MEM_GB, QUEUEADAPTER_EXPECTED_MEM_GB),
        (16.555, "16.56G"),
    ],
)
def test_format_slurm_memory_from_gb(
    total_memory_in_gb: float, expected_memory: str
) -> None:
    executor_settings = ExecutorSettings(
        number_of_processes=8,
        total_memory_in_gb=total_memory_in_gb,
        maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN_ROUND,
    )
    slurm_settings = SlurmSettingsGenerator(
        executor_settings=executor_settings
    ).settings
    assert slurm_settings.mem == expected_memory


@pytest.mark.parametrize("maximum_walltime_in_minutes", [-0.01, -1.0])
def test_format_slurm_walltime_from_minutes_rejects_negative_inputs(
    maximum_walltime_in_minutes: float,
) -> None:
    executor_settings = ExecutorSettings(
        number_of_processes=8,
        total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB_ROUND,
        maximum_walltime_in_minutes=maximum_walltime_in_minutes,
    )
    with pytest.raises(ValueError, match="non-negative"):
        _ = SlurmSettingsGenerator(executor_settings=executor_settings).settings


@pytest.mark.parametrize("total_memory_in_gb", [-0.01, -1.0])
def test_format_slurm_memory_from_gb_rejects_negative_inputs(
    total_memory_in_gb: float,
) -> None:
    executor_settings = ExecutorSettings(
        number_of_processes=8,
        total_memory_in_gb=total_memory_in_gb,
        maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN_ROUND,
    )
    with pytest.raises(ValueError, match="non-negative"):
        _ = SlurmSettingsGenerator(executor_settings=executor_settings).settings
