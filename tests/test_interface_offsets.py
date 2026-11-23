import pytest

from crystal.structure.interfaces.utils import (
    build_in_plane_offset_grid,
    compute_interface_gaps,
)


def test_build_in_plane_offset_grid() -> None:
    in_plane_offsets = build_in_plane_offset_grid((2, 2))
    assert len(in_plane_offsets) == 4
    assert set(in_plane_offsets) == {(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)}

    in_plane_offsets = build_in_plane_offset_grid((1, 3))
    assert in_plane_offsets == [(0.0, 0.0), (0.0, 1 / 3), (0.0, 2 / 3)]

    assert len(in_plane_offsets) == len(set(in_plane_offsets))
    assert all(
        0.0 <= x_offset < 1.0 and 0.0 <= y_offset < 1.0
        for x_offset, y_offset in in_plane_offsets
    )


def test_compute_interface_gaps_defaults() -> None:
    assert compute_interface_gaps(gaps=None, multipliers=None, estimated_gap=2.5) == [
        2.5
    ]


def test_compute_interface_gaps_explicit() -> None:
    assert compute_interface_gaps(
        gaps=[1.0, 2.0], multipliers=None, estimated_gap=2.5
    ) == [1.0, 2.0]


def test_compute_interface_gaps_multipliers() -> None:
    assert compute_interface_gaps(
        gaps=None, multipliers=[1.0, 1.5], estimated_gap=2.0
    ) == [2.0, 3.0]


def test_compute_interface_gaps_conflict() -> None:
    with pytest.raises(ValueError, match="gaps and multipliers"):
        compute_interface_gaps(gaps=[1.0], multipliers=[1.0], estimated_gap=2.0)


def test_compute_interface_gaps_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_interface_gaps(gaps=[0.0], multipliers=None, estimated_gap=2.0)
    with pytest.raises(ValueError, match="positive"):
        compute_interface_gaps(gaps=None, multipliers=[-1.0], estimated_gap=2.0)
