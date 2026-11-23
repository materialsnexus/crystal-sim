"""Queueadapter settings helpers for mapping executor settings to SLURM overrides."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from math import ceil
from typing import TypeAlias

from crystal.constants.fireworks import (
    QUEUEADAPTER_NEGATIVE_MEMORY_ERROR,
    QUEUEADAPTER_NEGATIVE_WALLTIME_ERROR,
    QUEUEADAPTER_SPEC_KEY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    SLURM_MEMORY_DECIMAL_PLACES,
    SLURM_MEMORY_GB_SUFFIX,
    SLURM_WALLTIME_FORMAT,
)
from crystal.schemas.fireworks import PartitionType, SlurmSettings
from crystal.schemas.serialization import MSONableBaseModel
from crystal.schemas.settings import ExecutorSettings

QueueAdapterSettingsInput: TypeAlias = ExecutorSettings
QueueAdapterSettingsMapping: TypeAlias = Mapping[str, ExecutorSettings]


class SlurmSettingsGenerator(MSONableBaseModel):  # noqa: D101
    executor_settings: ExecutorSettings

    @property
    def settings(self) -> SlurmSettings:  # noqa: D102
        return SlurmSettings(
            ntasks=self.executor_settings.number_of_processes,
            walltime=self._walltime,
            mem=self._memory,
            gres=self._gres,
            queue=self._queue,
        )

    @property
    def _queue(self) -> str | None:
        if self.executor_settings.number_of_gpus > 0:
            return PartitionType.gpu
        return None

    @property
    def _gres(self) -> str | None:
        number_of_gpus = self.executor_settings.number_of_gpus
        if number_of_gpus == 0:
            return None
        return f"gpu:{number_of_gpus}"

    @property
    def _walltime(self) -> str:
        maximum_walltime_in_minutes = self.executor_settings.maximum_walltime_in_minutes
        if maximum_walltime_in_minutes < 0:
            raise ValueError(QUEUEADAPTER_NEGATIVE_WALLTIME_ERROR)
        total_seconds = ceil(maximum_walltime_in_minutes * SECONDS_PER_MINUTE)
        hours, remaining_seconds = divmod(total_seconds, SECONDS_PER_HOUR)
        minutes, seconds = divmod(remaining_seconds, SECONDS_PER_MINUTE)
        return SLURM_WALLTIME_FORMAT.format(
            hours=hours, minutes=minutes, seconds=seconds
        )

    @property
    def _memory(self) -> str:
        total_memory_in_gb = self.executor_settings.total_memory_in_gb
        if total_memory_in_gb < 0:
            raise ValueError(QUEUEADAPTER_NEGATIVE_MEMORY_ERROR)
        quantize_exp = Decimal(f"1.{'0' * SLURM_MEMORY_DECIMAL_PLACES}")
        memory_value = Decimal(str(total_memory_in_gb)).quantize(
            quantize_exp, rounding=ROUND_HALF_UP
        )
        memory_as_string = format(memory_value, "f").rstrip("0").rstrip(".")
        return f"{memory_as_string}{SLURM_MEMORY_GB_SUFFIX}"


def build_slurm_settings_from_executor_settings(  # noqa: D103
    executor_settings: ExecutorSettings,
) -> SlurmSettings:
    return SlurmSettingsGenerator(executor_settings=executor_settings).settings


def build_queueadapter_metadata_from_executor_settings(  # noqa: D103
    executor_settings: ExecutorSettings,
) -> dict[str, dict[str, int | str]]:
    queueadapter_spec = build_slurm_settings_from_executor_settings(
        executor_settings
    ).model_dump(exclude_none=True)
    return {QUEUEADAPTER_SPEC_KEY: queueadapter_spec}
