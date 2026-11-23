"""Maker for structure augmentation and geometric filtering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from jobflow.core.job import Job, job

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.helpers.cohesive_energy import filter_structures_by_cohesive_energy
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.schemas.base import BaseMaker
from crystal.schemas.filtering import CohesiveEnergyFilteringSettings
from crystal.schemas.filtering_statistics import CohesiveEnergyFilteringStatistics
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationProperties,
)
from crystal.schemas.maker_names import MakerName
from crystal.schemas.mlip import MlipRelaxTaskDocument
from crystal.schemas.properties import RelaxProperties, StructureList
from crystal.schemas.structure_generators import (
    InitialStructureGenerationTaskDocument,
    StructureAugmentationAndFilteringSettings,
    StructureAugmentationAndFilteringTaskDocument,
)
from crystal.structure.generators.augmentation_filtering import (
    augment_and_filter_structures,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _AugmentationInputStructures:
    """Inputs and diagnostics extracted before structure augmentation."""

    initial_structures: StructureList
    relaxed_structures: StructureList
    relaxed_energies: list[float]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CohesiveEnergyFilteringResult:
    """Relaxed structures after cohesive-energy filtering and its statistics."""

    relaxed_structures: StructureList
    statistics: CohesiveEnergyFilteringStatistics | None


def _extract_augmentation_input_structures(
    *,
    initial_structure_generation_task_document: InitialStructureGenerationTaskDocument,
    mlip_relax_task_documents: list[MlipRelaxTaskDocument],
) -> _AugmentationInputStructures:
    initial_properties = initial_structure_generation_task_document.properties
    initial_structures = (
        initial_properties.structures
        if isinstance(initial_properties, InitialStructureGenerationProperties)
        else []
    )
    successful_relax_properties: list[RelaxProperties] = []
    failed_relax_documents: list[MlipRelaxTaskDocument] = []
    for task_document in mlip_relax_task_documents:
        properties = task_document.properties
        if isinstance(properties, RelaxProperties) and properties.relax_steps:
            successful_relax_properties.append(properties)
        else:
            failed_relax_documents.append(task_document)
    relaxed_structures = [
        properties.relaxed_structure for properties in successful_relax_properties
    ]
    relaxed_energies = [
        properties.final_energy for properties in successful_relax_properties
    ]
    warnings = tuple(
        (
            "MLIP relaxation failed: "
            f"{task_document.errors or 'no relaxation steps were recorded'}"
        )
        for task_document in failed_relax_documents
    )
    return _AugmentationInputStructures(
        initial_structures=initial_structures,
        relaxed_structures=relaxed_structures,
        relaxed_energies=relaxed_energies,
        warnings=warnings,
    )


def _filter_relaxed_structures_by_cohesive_energy(
    *,
    relaxed_structures: StructureList,
    relaxed_energies: list[float],
    total_relaxation_count: int,
    failed_relaxation_count: int,
    settings: CohesiveEnergyFilteringSettings | None,
) -> _CohesiveEnergyFilteringResult:
    """Filter relaxed structures and collect pre-augmentation statistics."""
    filtered_structures = (
        None
        if settings is None
        else filter_structures_by_cohesive_energy(
            structures=relaxed_structures,
            total_energies=relaxed_energies,
            settings=settings,
        )
    )
    if filtered_structures is None:
        return _CohesiveEnergyFilteringResult(
            relaxed_structures=relaxed_structures, statistics=None
        )
    return _CohesiveEnergyFilteringResult(
        relaxed_structures=filtered_structures,
        statistics=CohesiveEnergyFilteringStatistics(
            considered=total_relaxation_count,
            passed=len(filtered_structures),
            filtered=len(relaxed_structures) - len(filtered_structures),
            failed=failed_relaxation_count,
        ),
    )


@dataclass(kw_only=True)
class StructureAugmentationAndFilteringMaker(BaseMaker):
    """Augment MLIP-relaxed structures and filter the generated candidates."""

    name: MakerName = MakerName.structure_augmentation_and_filtering
    simulation_settings: StructureAugmentationAndFilteringSettings

    @job(
        output_schema=StructureAugmentationAndFilteringTaskDocument,
        **{STRUCTURE_BLOB_STORE_NAME: [["output", "properties", "structures"]]},
    )  # type: ignore[call-overload]
    def _run(
        self,
        initial_structure_generation_task_document: InitialStructureGenerationTaskDocument,
        mlip_relax_task_documents: list[MlipRelaxTaskDocument],
    ) -> StructureAugmentationAndFilteringTaskDocument:
        run_directory = Path.cwd()
        try:
            augmentation_inputs = _extract_augmentation_input_structures(
                initial_structure_generation_task_document=initial_structure_generation_task_document,
                mlip_relax_task_documents=mlip_relax_task_documents,
            )
            cohesive_energy_filtering_result = _filter_relaxed_structures_by_cohesive_energy(
                relaxed_structures=augmentation_inputs.relaxed_structures,
                relaxed_energies=augmentation_inputs.relaxed_energies,
                total_relaxation_count=len(mlip_relax_task_documents),
                failed_relaxation_count=(
                    len(mlip_relax_task_documents)
                    - len(augmentation_inputs.relaxed_structures)
                ),
                settings=self.simulation_settings.cohesive_energy_filtering_settings,
            )
            result = augment_and_filter_structures(
                initial_structures=augmentation_inputs.initial_structures,
                relaxed_structures=cohesive_energy_filtering_result.relaxed_structures,
                settings=self.simulation_settings,
            )
            statistics = result.properties.statistics.model_copy(
                update={
                    "pre_augmentation_cohesive_energy": (
                        cohesive_energy_filtering_result.statistics
                    )
                }
            )
            properties = result.properties.model_copy(update={"statistics": statistics})
            warnings = "\n".join((*augmentation_inputs.warnings, *result.warnings))
            errors = ""
            if warnings:
                logger.warning(
                    "Structure augmentation and filtering completed with warnings."
                )
        except Exception as exc:
            properties = None
            warnings = ""
            errors = get_diagnostic_exception_message(exc)
            logger.exception("Structure augmentation and filtering failed.")

        return StructureAugmentationAndFilteringTaskDocument(
            dir_name=run_directory.as_posix(),
            properties=properties,
            errors=errors,
            warnings=warnings,
        )

    def make(
        self,
        initial_structure_generation_task_document: InitialStructureGenerationTaskDocument,
        mlip_relax_task_documents: list[MlipRelaxTaskDocument],
    ) -> Job:
        """Create a structure augmentation and filtering job."""
        structure_job: Job = self._run(
            initial_structure_generation_task_document=initial_structure_generation_task_document,
            mlip_relax_task_documents=mlip_relax_task_documents,
        )
        structure_job.update_metadata(
            self.slurm_settings.to_queueadapter_spec(), dynamic=False
        )
        return structure_job
