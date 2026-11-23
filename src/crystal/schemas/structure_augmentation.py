"""Schemas for structure augmentation and geometric filtering."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator

from crystal.constants.structure_generators import STAGE_LABEL_PROPERTY_KEY
from crystal.schemas.base_properties import Properties
from crystal.schemas.filtering_statistics import CohesiveEnergyFilteringStatistics
from crystal.schemas.properties import StructureList


class StructureGenerationSource(StrEnum):
    """Stages used to label generated structures."""

    initial = "Structure Generation"
    mlip_relax = "MLIP Relax"
    rattle_after_mlip_relax = "Rattle after MLIP Relax"
    stretch_after_rattle = "Stretch after Rattle"
    stretch_after_mlip_relax = "Stretch after MLIP Relax"


class StructureAugmentationRootFailureSummary(BaseModel):
    """Bounded diagnostic summary for failed augmentation roots."""

    model_config = ConfigDict(extra="forbid")

    source: StructureGenerationSource
    exception_type: str
    count: PositiveInt
    first_error_message: str


class StructureGenerationSourceStatistics(BaseModel):
    """Counts for one structure-generation stage."""

    model_config = ConfigDict(extra="forbid")

    considered: NonNegativeInt = 0
    passed: NonNegativeInt = 0
    filtered: NonNegativeInt = 0

    @model_validator(mode="after")
    def _validate_count_partition(self) -> StructureGenerationSourceStatistics:
        if self.passed + self.filtered != self.considered:
            msg = "passed plus filtered must equal considered."
            raise ValueError(msg)
        return self


class StructureAugmentationAndFilteringStatistics(BaseModel):
    """Counts of candidates by structure-generation stage."""

    model_config = ConfigDict(extra="forbid")

    initial: StructureGenerationSourceStatistics
    mlip_relax: StructureGenerationSourceStatistics
    rattle_after_mlip_relax: StructureGenerationSourceStatistics
    stretch_after_rattle: StructureGenerationSourceStatistics
    stretch_after_mlip_relax: StructureGenerationSourceStatistics
    pre_augmentation_cohesive_energy: CohesiveEnergyFilteringStatistics | None = None
    root_failure_summaries: tuple[StructureAugmentationRootFailureSummary, ...] = ()

    @classmethod
    def from_source_statistics(
        cls,
        source_statistics: Mapping[
            StructureGenerationSource, StructureGenerationSourceStatistics
        ],
        root_failure_summaries: tuple[
            StructureAugmentationRootFailureSummary, ...
        ] = (),
        pre_augmentation_cohesive_energy: CohesiveEnergyFilteringStatistics
        | None = None,
    ) -> Self:
        """Build statistics from counts keyed by generation stage."""
        return cls(
            initial=source_statistics[StructureGenerationSource.initial],
            mlip_relax=source_statistics[StructureGenerationSource.mlip_relax],
            rattle_after_mlip_relax=source_statistics[
                StructureGenerationSource.rattle_after_mlip_relax
            ],
            stretch_after_rattle=source_statistics[
                StructureGenerationSource.stretch_after_rattle
            ],
            stretch_after_mlip_relax=source_statistics[
                StructureGenerationSource.stretch_after_mlip_relax
            ],
            pre_augmentation_cohesive_energy=pre_augmentation_cohesive_energy,
            root_failure_summaries=root_failure_summaries,
        )


class StructureAugmentationAndFilteringProperties(Properties):
    """Filtered structures and stage counts."""

    type: Literal["StructureAugmentationAndFilteringProperties"] = (
        "StructureAugmentationAndFilteringProperties"
    )
    structures: StructureList
    statistics: StructureAugmentationAndFilteringStatistics

    @model_validator(mode="after")
    def _validate_structure_generation_stage_labels(
        self,
    ) -> StructureAugmentationAndFilteringProperties:
        valid_stage_labels = {source.value for source in StructureGenerationSource}
        for structure in self.structures:
            stage_label = structure.properties.get(STAGE_LABEL_PROPERTY_KEY)
            if stage_label not in valid_stage_labels:
                msg = "each structure must have a valid generation-stage label."
                raise ValueError(msg)
        return self


class StructureAugmentationAndFilteringResult(BaseModel):
    """In-memory result of structure augmentation and geometric filtering."""

    properties: StructureAugmentationAndFilteringProperties
    warnings: tuple[str, ...] = ()
