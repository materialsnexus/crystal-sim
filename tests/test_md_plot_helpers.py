"""Unit tests for MD trajectory analysis helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from crystal.helpers.md_plot_helpers import (
    TrajectoryAnalyzer,
    plot_time_series_with_distribution,
)
from crystal.schemas.md.data import MdSeries, MdTrajectorySummary
from crystal.schemas.md.utils import compute_running_average_series
from tests.helpers.constants.md import TEST_TRAJECTORY_PATH
from tests.helpers.constants.structure import FLOATING_POINT_TOLERANCE

TEST_ANALYZER = TrajectoryAnalyzer(trajectory_path=TEST_TRAJECTORY_PATH)
MOCK_MD_SERIES = MdSeries(
    time_ps=[0.0, 0.1, 0.2, 0.3], values=[300.0, 305.0, 299.0, 301.0]
)


def test_trajectory_analyzer_frames_returns_all_frames() -> None:
    assert len(TEST_ANALYZER.frames) == 15


def test_trajectory_analyzer_frames_raises_for_empty_trajectory(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "empty.traj"
    trajectory_path.touch()

    analyzer = TrajectoryAnalyzer(trajectory_path=trajectory_path)

    with pytest.raises(ValueError, match="No frames were found in trajectory"):
        _ = analyzer.frames


def test_trajectory_analyzer_max_force_components_returns_expected_values() -> None:
    assert TEST_ANALYZER.max_force_components.tolist()[-2:] == pytest.approx(
        [0.01075717, 0.00860948], abs=FLOATING_POINT_TOLERANCE
    )


def test_trajectory_analyzer_max_force_atom_indices_returns_expected_values() -> None:
    assert TEST_ANALYZER.max_force_atom_indices.tolist()[-2:] == [19, 19]


def test_trajectory_analyzer_fractional_z_positions_tracks_selected_atom() -> None:
    assert TEST_ANALYZER.fractional_z_positions.tolist()[-2:] == pytest.approx(
        [0.617, 0.617], abs=FLOATING_POINT_TOLERANCE
    )


def test_trajectory_analyzer_z_extents_returns_expected_cell_extent() -> None:
    assert TEST_ANALYZER.z_extents().tolist()[-2:] == pytest.approx(
        [7.997, 7.998], abs=FLOATING_POINT_TOLERANCE
    )


def test_trajectory_analyzer_to_summary_returns_serializable_summary() -> None:
    summary = TEST_ANALYZER.to_summary()

    assert isinstance(summary, MdTrajectorySummary)
    assert summary.trajectory_path == TEST_TRAJECTORY_PATH
    assert summary.max_force_components[-2:] == pytest.approx(
        [0.01075717, 0.00860948], abs=FLOATING_POINT_TOLERANCE
    )


def test_plot_time_series_with_distribution_writes_plot_file(tmp_path: Path) -> None:
    output_path = tmp_path / "temperature_vs_time.png"
    assert not output_path.is_file()
    plot_time_series_with_distribution(
        series=MOCK_MD_SERIES,
        output_path=output_path,
        y_label="Temperature [K]",
        averaging_fraction=0.5,
    )

    assert output_path.is_file()


def test_plot_time_series_with_distribution_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="averaging_fraction"):
        plot_time_series_with_distribution(
            series=MOCK_MD_SERIES,
            output_path=Path("ignored.png"),
            y_label="Energy [eV]",
            averaging_fraction=0.0,
        )


def test_compute_running_average_series_returns_expected_values() -> None:
    averaged_series = compute_running_average_series(
        series=MdSeries(time_ps=[0.0, 0.1, 0.2, 0.3], values=[1.0, 3.0, 5.0, 7.0]),
        time_averaging_window=0.1,
    )

    assert averaged_series.time_ps == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert averaged_series.values == pytest.approx([1.0, 2.0, 4.0, 6.0])


def test_compute_running_average_series_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="time_averaging_window"):
        compute_running_average_series(
            series=MdSeries(time_ps=[0.0, 0.1], values=[1.0, 2.0]),
            time_averaging_window=0.0,
        )
