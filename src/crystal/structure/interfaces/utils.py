"""Utilities for interface offset grids and gap expansion."""

from collections.abc import Sequence

from crystal.structure.slabs.builder import Slab

DEFAULT_ESTIMATED_GAP_SAFETY_MULTIPLIER = 1.1


def build_in_plane_offset_grid(
    grid_dimensions: tuple[int, int],
) -> list[tuple[float, float]]:
    """Return fractional in-plane offsets for an (nx, ny) grid in [0, 1)."""
    num_divisions_x, num_divisions_y = grid_dimensions
    if num_divisions_x <= 0 or num_divisions_y <= 0:
        msg = "Grid dimensions must be positive"
        raise ValueError(msg)
    in_plane_offsets: list[tuple[float, float]] = []
    for x_index in range(num_divisions_x):
        fractional_x_offset = x_index / num_divisions_x
        for y_index in range(num_divisions_y):
            fractional_y_offset = y_index / num_divisions_y
            in_plane_offsets.append((fractional_x_offset, fractional_y_offset))
    return in_plane_offsets


def compute_interface_gaps(
    *,
    gaps: Sequence[float] | None,
    multipliers: Sequence[float] | None,
    estimated_gap: float,
) -> list[float]:
    """Return concrete gap values from explicit gaps or estimated-gap multipliers."""
    if gaps is not None and multipliers is not None:
        msg = "Provide gaps or multipliers, not both (gaps and multipliers conflict)"
        raise ValueError(msg)
    if estimated_gap <= 0:
        msg = "Estimated gap must be positive"
        raise ValueError(msg)
    if gaps is not None:
        if len(gaps) == 0:
            msg = "Gap list must be non-empty"
            raise ValueError(msg)
        if any(gap <= 0 for gap in gaps):
            msg = "Gap values must be positive"
            raise ValueError(msg)
        return list(gaps)
    if multipliers is not None:
        if len(multipliers) == 0:
            msg = "Multiplier list must be non-empty"
            raise ValueError(msg)
        if any(multiplier <= 0 for multiplier in multipliers):
            msg = "Multiplier values must be positive"
            raise ValueError(msg)
        return [estimated_gap * multiplier for multiplier in multipliers]
    return [estimated_gap]


def compute_estimated_vacuum_gap_in_angstrom(
    *,
    top_slab: Slab,
    bottom_slab: Slab,
    safety_multiplier: float = DEFAULT_ESTIMATED_GAP_SAFETY_MULTIPLIER,
) -> float:
    """Estimate a safe inter-slab gap from nearest-neighbor distances."""
    if safety_multiplier <= 0:
        msg = "Safety multiplier must be positive"
        raise ValueError(msg)
    average_neighbor_distance = 0.5 * (
        top_slab.minimum_neighbor_distance_in_angstrom
        + bottom_slab.minimum_neighbor_distance_in_angstrom
    )
    return safety_multiplier * average_neighbor_distance
