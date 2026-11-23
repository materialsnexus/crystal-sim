"""Trajectory analysis helpers for MD stability workflows."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from numpy.typing import NDArray

from crystal.schemas.md.data import MdSeries, MdTrajectorySummary
from crystal.schemas.md.utils import compute_running_average_series
from crystal.schemas.serialization import MSONableBaseModel


def plot_time_series_with_distribution(
    *, series: MdSeries, output_path: Path, y_label: str, averaging_fraction: float
) -> None:
    """Write a time-series plot and histogram for the series values."""
    if not 0.0 < averaging_fraction <= 1.0:
        error_msg = "averaging_fraction must be in the interval (0, 1]."
        raise ValueError(error_msg)

    time_span_ps = series.time_ps[-1] - series.time_ps[0]
    time_averaging_window = max(time_span_ps * averaging_fraction, 1e-12)
    averaged_series = compute_running_average_series(
        series=series, time_averaging_window=time_averaging_window
    )

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    time_axis, histogram_axis = axes
    series.plot(y_label=y_label, axis=time_axis, color="tab:blue", legend="Series")
    averaged_series.plot(
        y_label=y_label, axis=time_axis, color="tab:orange", legend="Running average"
    )
    mean_value = series.statistics.mean
    time_axis.axhline(mean_value, color="tab:red", linestyle="--", linewidth=1.0)
    time_axis.legend()

    histogram_axis.hist(
        series.values,
        bins=min(20, len(series.values)),
        color="tab:blue",
        orientation="horizontal",
    )
    histogram_axis.axhline(mean_value, color="tab:red", linestyle="--", linewidth=1.0)
    histogram_axis.set_xlabel("Count")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


class TrajectoryAnalyzer(MSONableBaseModel):  # noqa: D101
    trajectory_path: Path

    @cached_property
    def frames(self) -> list[Atoms]:  # noqa: D102
        try:
            frames = read(self.trajectory_path, index=":")
        except Exception as error:
            error_msg = f"No frames were found in trajectory: {self.trajectory_path}"
            raise ValueError(error_msg) from error
        return frames if isinstance(frames, list) else [frames]

    @property
    def max_force_components(self) -> NDArray[np.float64]:  # noqa: D102
        return np.array(
            [float(np.abs(frame.get_forces()).max()) for frame in self.frames]  # type:ignore[no-untyped-call]
        )

    @property
    def max_force_atom_indices(self) -> NDArray[np.int_]:  # noqa: D102
        return np.array(
            [
                int(
                    np.unravel_index(
                        np.abs(frame.get_forces()).argmax(),  # type:ignore[no-untyped-call]
                        frame.get_forces().shape,  # type:ignore[no-untyped-call]
                    )[0]
                )
                for frame in self.frames
            ]
        )

    @property
    def fractional_z_positions(self) -> NDArray[np.float64]:  # noqa: D102
        return np.array(
            [
                float(frame.get_scaled_positions(wrap=False)[atom_index, 2])  # type:ignore[no-untyped-call]
                for frame, atom_index in zip(
                    self.frames, self.max_force_atom_indices, strict=True
                )
            ]
        )

    def z_extents(self) -> NDArray[np.float64]:
        """Return the z-extent of the atomic positions in each frame."""
        return np.array(
            [
                float(
                    frame.get_positions()[:, 2].max()  # type:ignore[no-untyped-call]
                    - frame.get_positions()[:, 2].min()  # type:ignore[no-untyped-call]
                )
                for frame in self.frames
            ]
        )

    def to_summary(self) -> MdTrajectorySummary:
        """Convert the analyzed trajectory into a serializable summary model."""
        return MdTrajectorySummary(
            max_force_components=self.max_force_components.tolist(),
            max_force_atom_indices=self.max_force_atom_indices.tolist(),
            fractional_z_of_max_force_atom=self.fractional_z_positions.tolist(),
            z_extent=self.z_extents().tolist(),
            trajectory_path=self.trajectory_path,
        )
