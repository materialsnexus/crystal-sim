"""Schemas for interface geometry optimisation workflows."""

from __future__ import annotations

from functools import cached_property
from typing import Self

from pydantic import Field, model_validator
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.helpers.file import count_args_not_none
from crystal.schemas.mlip import MlipRelaxMakerSettings
from crystal.schemas.serialization import MSONableBaseModel
from crystal.schemas.settings import SimulationSettings
from crystal.structure.interfaces.assembler import InterfaceAssembler
from crystal.structure.interfaces.builder import (
    Interface,
    InterfaceFactory,
    InterfaceMatchPolicy,
)
from crystal.structure.interfaces.settings import InterfaceAssemblySettings
from crystal.structure.slabs.builder import Slab

DEFAULT_INTERFACE_ASSEMBLY_SETTINGS = InterfaceAssemblySettings(
    offsets=[(0.0, 0.0)], gap_multipliers=[1.0]
)


class GeometryOptimisationSlabInputs(MSONableBaseModel):
    """Input state for slab-driven optimisation."""

    top_slab: Slab
    bottom_slab: Slab
    interface_match_policy: InterfaceMatchPolicy = Field(
        default_factory=InterfaceMatchPolicy
    )

    @cached_property
    def orthogonal_top_slab(self) -> Slab:  # noqa: D102
        return self.top_slab.orthogonalize_in_xy_plane

    @cached_property
    def orthogonal_bottom_slab(self) -> Slab:  # noqa: D102
        return self.bottom_slab.orthogonalize_in_xy_plane

    @cached_property
    def interface_factory(self) -> InterfaceFactory:  # noqa: D102
        return InterfaceFactory(
            top_slab=self.orthogonal_top_slab,
            bottom_slab=self.orthogonal_bottom_slab,
            interface_match_policy=self.interface_match_policy,
        )

    @property
    def reference_interface(self) -> Interface:  # noqa: D102
        return next(self.interface_factory.generate_interfaces(vacuum=self.gap))

    @cached_property
    def gap(self) -> float:
        """Return the minimum safe vacuum for both source slabs."""
        factory = self.interface_factory
        return max(
            factory.top_slab.minimum_neighbor_distance_in_angstrom,
            factory.bottom_slab.minimum_neighbor_distance_in_angstrom,
        )


class InterfaceGeometryOptimisationInputs(MSONableBaseModel):
    """Inputs for slab-driven or interface-driven geometry optimisation."""

    slab_inputs: GeometryOptimisationSlabInputs | None = None
    interface: Interface | None = None
    assembly_settings: InterfaceAssemblySettings = Field(
        default=DEFAULT_INTERFACE_ASSEMBLY_SETTINGS
    )

    @model_validator(mode="after")
    def _validate_source_mode(self) -> Self:
        """Require exactly one complete optimisation input mode."""
        if count_args_not_none(self.slab_inputs, self.interface) != 1:
            msg = "Provide exactly one of slab_inputs or interface."
            raise ValueError(msg)
        return self

    @cached_property
    def reference_interface(self) -> Interface:
        """Resolve the interface used as the source for candidate assembly."""
        return self.interface or self.slab_inputs.reference_interface  # type:ignore[union-attr]

    @property
    def gap(self) -> float:
        """Return the reference interface gap used to center source slabs."""
        return self.reference_interface.gap

    @cached_property
    def interface_candidates(self) -> list[Interface]:
        """Return assembled interface candidates from the selected input mode."""
        reference_interface = self.reference_interface
        top_slab = reference_interface.get_centered_top_slab(
            vacuum_in_angstrom=self.gap
        )
        bottom_slab = reference_interface.get_centered_bottom_slab(
            vacuum_in_angstrom=self.gap
        )
        interface_assembler = InterfaceAssembler(
            top_slab=top_slab, bottom_slab=bottom_slab
        )
        return list(
            interface_assembler.generate_interfaces(
                offsets=self.assembly_settings.offsets,
                gaps=self.assembly_settings.gaps,
                gap_multipliers=self.assembly_settings.gap_multipliers,
                vacuum=self.assembly_settings.vacuum,
            )
        )

    @property
    def interface_jobs_per_batch(self) -> int | None:
        """Return the requested relaxation batch size for interface candidates."""
        return self.assembly_settings.interface_jobs_per_batch


class GeometryOptimisationCandidateRecord(MSONableBaseModel):
    """Outcome for one geometry-optimisation candidate."""

    candidate_index: int
    final_relax_energy: float | None = None
    error_message: str = ""

    @property
    def is_success(self) -> bool:
        """Return whether the candidate relaxation produced final energy."""
        return self.final_relax_energy is not None


class GeometryOptimisationResult(MSONableBaseModel):
    """Generic selected structure and candidate records from relaxation."""

    selected_candidate_index: int
    selected_structure: PMG_Structure
    selected_final_energy: float
    candidate_records: list[GeometryOptimisationCandidateRecord]

    @property
    def selected_candidate_record(self) -> GeometryOptimisationCandidateRecord:
        """Return the record corresponding to the selected candidate index."""
        return self.candidate_records[self.selected_candidate_index]


class InterfaceGeometryOptimisationSimulationSettings(SimulationSettings):
    """Simulation settings for interface candidate relaxation."""

    relax_maker_settings: MlipRelaxMakerSettings
