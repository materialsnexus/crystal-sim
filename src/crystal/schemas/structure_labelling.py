"""Settings for single-point structure-labelling workflows."""

from __future__ import annotations

from crystal.schemas.settings import SimulationSettings
from crystal.schemas.single_point import SinglePointMakerSettings


class StructureLabellingWorkflowSettings(SimulationSettings):
    """Settings for one single-point backend used by the workflow."""

    single_point_maker_settings: SinglePointMakerSettings
