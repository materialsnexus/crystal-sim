"""Schemas for MD adhesion energy workflows."""

from __future__ import annotations

from typing import Self

from pydantic import Field
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.schemas.md.mlip_md_equilibration import (
    MlipMdEquilibrationSimulationSettings,
)
from crystal.schemas.serialization import MSONableBaseModel
from crystal.schemas.settings import ExecutorSettings, SimulationSettings
from crystal.structure.interfaces.builder import Interface


class MDAdhesionEnergySimulationSettings(SimulationSettings):
    """Settings for MD adhesion calculations."""

    interface_md_settings: list[MlipMdEquilibrationSimulationSettings] = Field(
        default_factory=list
    )
    production_md_setting: MlipMdEquilibrationSimulationSettings
    md_executor_settings: ExecutorSettings = Field(default_factory=ExecutorSettings)


class MDAdhesionStructures(MSONableBaseModel):
    """Structures used by the MD branches."""

    top_slab_structure: PMG_Structure
    bottom_slab_structure: PMG_Structure
    interface_structure: PMG_Structure

    @classmethod
    def from_interface_structure(cls, interface_structure: PMG_Structure) -> Self:
        """Derive centered surface structures from an interface structure."""
        interface = Interface.from_structure(interface_structure)
        return cls(
            top_slab_structure=interface.get_centered_top_slab(),
            bottom_slab_structure=interface.get_centered_bottom_slab(),
            interface_structure=interface,
        )


class MDAdhesionEnergyResult(MSONableBaseModel):
    """Result of an MD adhesion calculation."""

    top_slab_energy: float
    bottom_slab_energy: float
    interface_energy: float
    interface_area: float
    work_of_adhesion_per_area: float
