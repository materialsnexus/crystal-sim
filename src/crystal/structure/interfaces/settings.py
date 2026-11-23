"""Pydantic settings for interface assembly."""

from __future__ import annotations

from pydantic import NonNegativeFloat, PositiveInt

from crystal.schemas.serialization import MSONableBaseModel

DEFAULT_INTERFACE_VACUUM_ANGSTROM = 15.0


class InterfaceAssemblySettings(MSONableBaseModel):  # noqa: D101
    offsets: list[tuple[float, float]]
    gaps: list[float] | None = None
    gap_multipliers: list[float] | None = None
    vacuum: NonNegativeFloat = DEFAULT_INTERFACE_VACUUM_ANGSTROM
    interface_jobs_per_batch: PositiveInt | None = None
