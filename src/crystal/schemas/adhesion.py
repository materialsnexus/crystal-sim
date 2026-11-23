"""Schemas for adhesion workflow inputs and outputs."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, model_validator

from crystal.constants.adhesion import (
    INTERFACE_CHARGEMOL_DIR_NAMES_EMPTY_ERROR,
    INTERFACE_DIR_COUNT_MISMATCH_ERROR,
    INTERFACE_RECORD_COUNT_MISMATCH_ERROR,
    INTERFACE_RELAX_DIR_NAMES_EMPTY_ERROR,
    SLAB_CHARGEMOL_DIR_NAMES_EMPTY_ERROR,
    SLAB_DIR_COUNT_MISMATCH_ERROR,
    SLAB_RELAX_DIR_NAMES_EMPTY_ERROR,
)
from crystal.schemas.relaxation import RelaxMakerSettings
from crystal.schemas.serialization import MSONableBaseModel
from crystal.schemas.settings import SimulationSettings
from crystal.schemas.single_point import SinglePointWithChargesMakerSettings
from crystal.structure.interfaces.settings import InterfaceAssemblySettings
from crystal.structure.slabs.builder import Slab


class AdhesionWorkflowInputs(MSONableBaseModel):  # noqa: D101
    top_slab: Slab
    bottom_slab: Slab
    assembly_settings: InterfaceAssemblySettings


class AdhesionEnergyRecord(BaseModel):  # noqa: D101
    interface_index: int
    adhesion_energy: float
    interface_area_in_angstrom_square: float

    @property
    def adhesion_energy_per_area(self) -> float:  # noqa: D102
        return self.adhesion_energy / self.interface_area_in_angstrom_square


class AdhesionSimulationSettings(SimulationSettings):  # noqa: D101
    slab_relax_maker_settings: RelaxMakerSettings
    interface_relax_maker_settings: RelaxMakerSettings
    single_point_maker_settings: SinglePointWithChargesMakerSettings


class AdhesionWorkflowResult(BaseModel):  # noqa: D101
    slab_relax_dir_names: list[str]
    interface_relax_dir_names: list[str]
    slab_chargemol_dir_names: list[str]
    interface_chargemol_dir_names: list[str]
    adhesion_energy_records: list[AdhesionEnergyRecord]

    @model_validator(mode="after")
    def _validate_dir_names_and_counts(self) -> Self:
        if len(self.slab_relax_dir_names) != len(self.slab_chargemol_dir_names):
            error_msg = SLAB_DIR_COUNT_MISMATCH_ERROR
            raise ValueError(error_msg)
        if len(self.interface_relax_dir_names) != len(
            self.interface_chargemol_dir_names
        ):
            error_msg = INTERFACE_DIR_COUNT_MISMATCH_ERROR
            raise ValueError(error_msg)
        if len(self.interface_relax_dir_names) != len(self.adhesion_energy_records):
            error_msg = INTERFACE_RECORD_COUNT_MISMATCH_ERROR
            raise ValueError(error_msg)
        if not all(self.slab_relax_dir_names):
            error_msg = SLAB_RELAX_DIR_NAMES_EMPTY_ERROR
            raise ValueError(error_msg)
        if not all(self.interface_relax_dir_names):
            error_msg = INTERFACE_RELAX_DIR_NAMES_EMPTY_ERROR
            raise ValueError(error_msg)
        if not all(self.slab_chargemol_dir_names):
            error_msg = SLAB_CHARGEMOL_DIR_NAMES_EMPTY_ERROR
            raise ValueError(error_msg)
        if not all(self.interface_chargemol_dir_names):
            error_msg = INTERFACE_CHARGEMOL_DIR_NAMES_EMPTY_ERROR
            raise ValueError(error_msg)
        return self
