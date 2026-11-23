"""Shared MD data models."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots
from pydantic import NonNegativeFloat, NonNegativeInt, model_validator

from crystal.schemas.serialization import MSONableBaseModel

MINIMUM_MD_SERIES_SAMPLE_COUNT = 2


class MdSeriesStatistics(MSONableBaseModel):  # noqa: D101
    mean: float
    std: NonNegativeFloat
    drift_slope: float
    end_value: float


class MdTrajectorySummary(MSONableBaseModel):  # noqa: D101
    max_force_components: list[NonNegativeFloat]
    max_force_atom_indices: list[NonNegativeInt]
    fractional_z_of_max_force_atom: list[NonNegativeFloat]
    z_extent: list[NonNegativeFloat]
    trajectory_path: Path | None = None


class MdSeries(MSONableBaseModel):  # noqa: D101
    time_ps: list[float]
    values: list[float]

    @model_validator(mode="after")
    def _validate_lengths(self) -> Self:
        if len(self.time_ps) != len(self.values):
            error_msg = "time_ps and values must have the same length."
            raise ValueError(error_msg)
        if len(self.time_ps) < MINIMUM_MD_SERIES_SAMPLE_COUNT:
            error_msg = "time_ps and values must contain at least two entries."
            raise ValueError(error_msg)
        return self

    @property
    def statistics(self) -> MdSeriesStatistics:  # noqa: D102
        time_array = np.array(self.time_ps)
        values_array = np.array(self.values)

        mean = float(np.mean(values_array))
        std = float(np.std(values_array))
        drift_slope = float(np.polyfit(time_array, values_array, deg=1)[0])
        end_value = float(values_array[-1])

        return MdSeriesStatistics(
            mean=mean, std=std, drift_slope=drift_slope, end_value=end_value
        )

    def get_final_window_series(self, final_window_ps: float) -> Self:
        """Return samples from the final time window as a new series."""
        if final_window_ps <= 0.0:
            error_msg = "final_window_ps must be positive."
            raise ValueError(error_msg)

        final_time_ps = max(self.time_ps)
        window_start_time_ps = final_time_ps - final_window_ps
        window_points = [
            (time_ps, value)
            for time_ps, value in zip(self.time_ps, self.values, strict=True)
            if time_ps >= window_start_time_ps
        ]
        if len(window_points) < MINIMUM_MD_SERIES_SAMPLE_COUNT:
            window_points = list(
                zip(
                    self.time_ps[-MINIMUM_MD_SERIES_SAMPLE_COUNT:],
                    self.values[-MINIMUM_MD_SERIES_SAMPLE_COUNT:],
                    strict=True,
                )
            )
        window_time_ps, window_values = zip(*window_points, strict=True)
        return type(self)(time_ps=list(window_time_ps), values=list(window_values))

    def plot(
        self,
        *,
        y_label: str,
        axis: Axes | None = None,
        color: str | None = None,
        legend: str | None = None,
    ) -> Axes:
        """Plot the time series and return the matplotlib axes."""
        if axis is None:
            _, axis = subplots()
        axis.plot(self.time_ps, self.values, color=color, label=legend)
        axis.set_xlabel("Time [ps]")
        axis.set_ylabel(y_label)
        return axis


class DefaultMdLogData(MSONableBaseModel):
    """Parsed time-series values from the ASE default MD log."""

    time_ps: list[float]
    total_energy_ev: list[float]
    temperature_k: list[float]


class UserDefinedMdLogData(MSONableBaseModel):
    """Parsed time-series values from the custom MD log."""

    time_ps: list[float]
    density: list[float]
    max_force_component: list[float]
