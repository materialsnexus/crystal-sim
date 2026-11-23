"""Schemas for MLIP validation-dataset planning."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    model_validator,
)
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.structure_generators import (
    DEFAULT_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE,
)
from crystal.constants.validation_dataset import (
    DEFAULT_MAXIMUM_FRAMES_IN_SELECTION_POOL,
    VALIDATION_DATASET_TEMPERATURE_SAMPLING_REQUIRES_THERMOSTAT_ERROR,
)
from crystal.schemas.base_properties import Properties
from crystal.schemas.md.mlip_md_equilibration import MlipMdEquilibrationMakerSettings
from crystal.schemas.md.settings import MdEnsemble
from crystal.schemas.mlip import MlipRelaxMakerSettings
from crystal.schemas.properties import StructureList
from crystal.schemas.serialization import MSONableBaseModel
from crystal.schemas.settings import SimulationSettings
from crystal.schemas.structure_generators import DirectSamplingMakerSettings
from crystal.structure.interfaces.builder import InterfaceMatchPolicy
from crystal.structure.interfaces.settings import InterfaceAssemblySettings
from crystal.structure.slabs.builder import Slab

MillerIndex = tuple[int, int, int]
DEFAULT_MILLER_INDICES: tuple[MillerIndex, ...] = (
    (0, 0, 1),
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (-1, 0, 1),
    (0, -1, 1),
)
DEFAULT_MAXIMUM_STRUCTURE_ATOMS = 220


class ValidationDatasetBulkStructureInput(MSONableBaseModel):
    """Identify one user-provided bulk structure for validation generation."""

    structure_id: str = Field(
        min_length=1,
        description=(
            "Stable identifier for this input bulk structure; it is not an "
            "identifier for a generated slab, interface, or MD structure."
        ),
    )
    structure: PMG_Structure


class ValidationDatasetWorkflowInputs(MSONableBaseModel):
    """User-provided bulk structures that seed the validation workflow.

    The ordered pair is used in both top and bottom roles when generating
    interfaces.
    """

    bulk_structures: tuple[
        ValidationDatasetBulkStructureInput, ValidationDatasetBulkStructureInput
    ]

    @model_validator(mode="after")
    def _validate_source_ids(self) -> Self:
        source_ids = [source.structure_id for source in self.bulk_structures]
        if len(set(source_ids)) != len(source_ids):
            message = "Bulk structure IDs must be unique"
            raise ValueError(message)
        return self


class ValidationDatasetArtifactReference(MSONableBaseModel):
    """Reference to data stored outside a jobflow task document."""

    artifact_id: str = Field(min_length=1)
    store_name: str = Field(min_length=1)


class ValidationDatasetStructureBatchReference(MSONableBaseModel):
    """Reference to a bounded batch of structures stored outside MongoDB documents."""

    batch_id: str = Field(
        min_length=1, description="Identifier for the complete stored structure batch."
    )
    structure_artifact: ValidationDatasetArtifactReference
    item_count: NonNegativeInt


class ValidationDatasetStructureBatchJobReference(MSONableBaseModel):
    """Reference to a preparation job that produces one structure batch."""

    batch_id: str = Field(
        min_length=1,
        description=(
            "Identifier for the source batch: a bulk structure for slabs or a "
            "top/bottom source pair for interfaces."
        ),
    )
    source_kind: ValidationDatasetMdRunKind
    job_uuid: str = Field(min_length=1)


class ValidationDatasetSlabGenerationSettings(MSONableBaseModel):
    """Bounds and orientations used while generating slab candidates."""

    miller_indices: tuple[MillerIndex, ...] = Field(
        default=DEFAULT_MILLER_INDICES, min_length=1
    )
    maximum_atoms: PositiveInt = DEFAULT_MAXIMUM_STRUCTURE_ATOMS
    maximum_generation_attempts: PositiveInt = (
        DEFAULT_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE
    )
    supercell: tuple[PositiveInt, PositiveInt, PositiveInt] = (1, 1, 5)
    vacuum_thickness_in_layers: PositiveInt = 2
    orthogonalize_slabs: bool = True
    reduce_slabs_to_primitive: bool = True

    @model_validator(mode="after")
    def _validate_miller_indices(self) -> Self:
        if len(set(self.miller_indices)) != len(self.miller_indices):
            message = "Miller indices must be unique"
            raise ValueError(message)
        if (0, 0, 0) in self.miller_indices:
            message = "Miller index cannot be (0, 0, 0)"
            raise ValueError(message)
        return self


class ValidationDatasetSlabPreparationProperties(Properties):
    """Generated and successfully re-relaxed slabs for one bulk structure."""

    type: Literal["ValidationDatasetSlabPreparationProperties"] = (
        "ValidationDatasetSlabPreparationProperties"
    )
    source_structure_id: str = Field(
        min_length=1,
        description="Identifier of the input bulk structure that produced these slabs.",
    )
    slabs: list[Slab]
    failed_relaxation_count: NonNegativeInt = 0

    @computed_field
    @property
    def slab_count(self) -> int:
        """Return the number of generated slabs."""
        return len(self.slabs)


class ValidationDatasetInterfaceGenerationProperties(Properties):
    """Interfaces generated from bounded matched slab pairs."""

    type: Literal["ValidationDatasetInterfaceGenerationProperties"] = (
        "ValidationDatasetInterfaceGenerationProperties"
    )
    interfaces: StructureList

    @computed_field
    @property
    def interface_count(self) -> int:
        """Return the number of generated interfaces."""
        return len(self.interfaces)


class ValidationDatasetInterfaceMatchSettings(MSONableBaseModel):
    """Bounds applied to generated interface matches."""

    interface_match_policy: InterfaceMatchPolicy = Field(
        default_factory=lambda: InterfaceMatchPolicy(filter_by=None)
    )
    maximum_atoms: PositiveInt = DEFAULT_MAXIMUM_STRUCTURE_ATOMS
    maximum_matches_per_slab_pair: PositiveInt = 10
    maximum_generation_attempts: PositiveInt = (
        DEFAULT_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE
    )


class ValidationDatasetInterfaceGenerationSettings(MSONableBaseModel):
    """Matching and assembly settings for validation interfaces."""

    match_settings: ValidationDatasetInterfaceMatchSettings = Field(
        default_factory=ValidationDatasetInterfaceMatchSettings
    )
    assembly_settings: InterfaceAssemblySettings = Field(
        default_factory=lambda: InterfaceAssemblySettings(offsets=[(0.0, 0.0)])
    )


class ValidationDatasetMdRunKind(StrEnum):
    """Kinds of structures that can receive an MD run."""

    slab = "slab"
    interface = "interface"


class ValidationDatasetMdProperties(Properties):
    """Compact MD result metadata with the trajectory stored externally."""

    type: Literal["ValidationDatasetMdProperties"] = "ValidationDatasetMdProperties"
    run_id: str = Field(min_length=1)
    source_kind: ValidationDatasetMdRunKind
    source_id: str = Field(
        min_length=1,
        description=(
            "Identifier of the individual slab or interface structure used for "
            "this MD run."
        ),
    )
    temperature_k: PositiveFloat
    frame_count: NonNegativeInt
    number_of_atoms: PositiveInt
    trajectory_bytes: bytes


class ValidationDatasetMdTaskDocument(MSONableBaseModel):
    """Result document for one validation-dataset MD attempt."""

    run_id: str = Field(min_length=1)
    properties: ValidationDatasetMdProperties | None = None
    errors: str = ""

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        if (self.properties is None) == (self.errors == ""):
            message = (
                "Validation-dataset MD task documents must contain either properties "
                "or errors."
            )
            raise ValueError(message)
        if self.properties is not None and self.properties.run_id != self.run_id:
            message = (
                "Validation-dataset MD task document and properties must use the same "
                "run ID."
            )
            raise ValueError(message)
        return self


class ValidationDatasetMdSamplingSettings(MSONableBaseModel):
    """Temperature and submission bounds for MD planning."""

    temperatures_k: tuple[PositiveFloat, ...] = Field(default=(300.0,), min_length=1)
    maximum_md_runs: PositiveInt | None = 100

    @model_validator(mode="after")
    def _validate_temperatures(self) -> Self:
        if len(set(self.temperatures_k)) != len(self.temperatures_k):
            message = "MD temperatures must be unique"
            raise ValueError(message)
        return self


class ValidationDatasetMdRunSpec(MSONableBaseModel):
    """Small, serializable description of one planned MD attempt."""

    run_id: str = Field(min_length=1)
    source_kind: ValidationDatasetMdRunKind
    source_id: str = Field(
        min_length=1,
        description="Identifier of the individual slab or interface structure receiving the MD run.",
    )
    temperature_k: PositiveFloat


class ValidationDatasetDirectSelectionSettings(MSONableBaseModel):
    """Bounds for frame collection and final DIRECT selection."""

    target_number_of_structures: PositiveInt = 200
    maximum_frames_per_md_run: PositiveInt = 100
    maximum_frames_in_selection_pool: PositiveInt = (
        DEFAULT_MAXIMUM_FRAMES_IN_SELECTION_POOL
    )


class ValidationDatasetWorkflowSettings(SimulationSettings):
    """Top-level settings for the validation-dataset workflow."""

    mlip_relax_maker_settings: MlipRelaxMakerSettings
    md_equilibration_maker_settings: MlipMdEquilibrationMakerSettings
    direct_sampling_maker_settings: DirectSamplingMakerSettings
    slab_generation: ValidationDatasetSlabGenerationSettings = Field(
        default_factory=ValidationDatasetSlabGenerationSettings
    )
    interface_generation: ValidationDatasetInterfaceGenerationSettings = Field(
        default_factory=ValidationDatasetInterfaceGenerationSettings
    )
    md_sampling: ValidationDatasetMdSamplingSettings = Field(
        default_factory=ValidationDatasetMdSamplingSettings
    )
    direct_selection: ValidationDatasetDirectSelectionSettings = Field(
        default_factory=ValidationDatasetDirectSelectionSettings
    )

    @model_validator(mode="after")
    def _validate_temperature_sampling_settings(self) -> Self:
        dynamics = self.md_equilibration_maker_settings.simulation_settings.run_settings.dynamics
        if dynamics.ensemble is MdEnsemble.nve or dynamics.thermostat is None:
            raise ValueError(
                VALIDATION_DATASET_TEMPERATURE_SAMPLING_REQUIRES_THERMOSTAT_ERROR
            )
        return self
