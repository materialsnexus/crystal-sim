"""Evaluate augmented structures before cohesive-energy filtering."""

from __future__ import annotations

from pathlib import Path

from jobflow.core.flow import Flow
from jobflow.core.job import Job, Response, job
from jobflow.core.reference import OutputReference
from pydantic import BaseModel, ConfigDict, NonNegativeInt
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.single_point_maker import MlipSinglePointBatchMaker
from crystal.constants.structure_generators import (
    POST_AUGMENTATION_FAILURE_WARNING_MAX_LENGTH,
    POST_AUGMENTATION_FAILURE_WARNING_TRUNCATION_SUFFIX,
    POST_AUGMENTATION_MAX_FAILURE_WARNINGS,
    STRUCTURE_BLOB_STORE_NAME,
)
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.helpers.structure import split_items_into_batches
from crystal.schemas.filtering import (
    CohesiveEnergyFilteringSettings,
    CohesiveEnergyFilteringTaskDocument,
)
from crystal.schemas.filtering_statistics import CohesiveEnergyFilteringStatistics
from crystal.schemas.mlip import MlipSinglePointTaskDocument
from crystal.schemas.properties import (
    CohesiveEnergyFilteringProperties,
    SinglePointProperties,
    StructureList,
)
from crystal.schemas.structure_augmentation import (
    StructureAugmentationAndFilteringProperties,
)
from crystal.schemas.structure_generators import (
    StructureAugmentationAndFilteringTaskDocument,
)
from crystal.structure.handlers.cohesive_energy_filtering import (
    run_cohesive_energy_filtering,
)


class _PostAugmentationFilteringInputs(BaseModel):
    """Aligned source structures and successful evaluator energies."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structures: StructureList
    total_energies: list[float]
    failed_count: NonNegativeInt
    warnings: tuple[str, ...] = ()


def _append_bounded_failure_warning(warnings: list[str], warning: str) -> bool:
    """Append one bounded warning and report whether it was retained."""
    if len(warnings) >= POST_AUGMENTATION_MAX_FAILURE_WARNINGS:
        return False
    if len(warning) > POST_AUGMENTATION_FAILURE_WARNING_MAX_LENGTH:
        warning = (
            warning[
                : POST_AUGMENTATION_FAILURE_WARNING_MAX_LENGTH
                - len(POST_AUGMENTATION_FAILURE_WARNING_TRUNCATION_SUFFIX)
            ]
            + POST_AUGMENTATION_FAILURE_WARNING_TRUNCATION_SUFFIX
        )
    warnings.append(warning)
    return True


@job(**{STRUCTURE_BLOB_STORE_NAME: [["output", "structures"]]})  # type: ignore[call-overload]
def prepare_post_augmentation_cohesive_energy_filtering_inputs(
    *,
    source_structures: StructureList,
    evaluation_task_documents: list[MlipSinglePointTaskDocument],
) -> _PostAugmentationFilteringInputs:
    """Align evaluator results with source structures for generic filtering."""
    successful_structures: list[PMG_Structure] = []
    total_energies: list[float] = []
    warnings: list[str] = []
    omitted_warning_count = 0
    for index, source_structure in enumerate(source_structures):
        if index >= len(evaluation_task_documents):
            warning = (
                "MLIP single-point evaluation failed: no task document was "
                f"returned for augmented structure {index}."
            )
            if not _append_bounded_failure_warning(warnings, warning):
                omitted_warning_count += 1
            continue
        evaluation_task_document = evaluation_task_documents[index]
        evaluation_properties = evaluation_task_document.properties
        if not isinstance(evaluation_properties, SinglePointProperties):
            warning = (
                "MLIP single-point evaluation failed: "
                f"{evaluation_task_document.errors or 'no diagnostic message provided'}"
            )
            if not _append_bounded_failure_warning(warnings, warning):
                omitted_warning_count += 1
            continue
        successful_structures.append(source_structure)
        total_energies.append(evaluation_properties.energy)
    if omitted_warning_count:
        warnings.append(
            f"{omitted_warning_count} additional MLIP single-point failure "
            "diagnostics omitted."
        )
    return _PostAugmentationFilteringInputs(
        structures=successful_structures,
        total_energies=total_energies,
        failed_count=len(source_structures) - len(successful_structures),
        warnings=tuple(warnings),
    )


def _augmentation_failure_task_document(
    error: str,
) -> CohesiveEnergyFilteringTaskDocument:
    return CohesiveEnergyFilteringTaskDocument(
        dir_name=Path.cwd().as_posix(),
        properties=None,
        errors=get_diagnostic_exception_message(
            RuntimeError("Augmentation stage failed: " + error)
        ),
    )


def _build_passthrough_filtering_task_document(
    structures: StructureList,
) -> CohesiveEnergyFilteringTaskDocument:
    """Return a successful filtering document for a disabled or empty step."""
    return CohesiveEnergyFilteringTaskDocument(
        dir_name=Path.cwd().as_posix(),
        properties=CohesiveEnergyFilteringProperties(
            structures=structures,
            statistics=CohesiveEnergyFilteringStatistics(
                considered=len(structures), passed=len(structures)
            ),
        ),
    )


@job
def build_post_augmentation_cohesive_energy_filtering_flow(
    *,
    augmentation_and_filtering_task_document: StructureAugmentationAndFilteringTaskDocument,
    source_structures_job_uuid: str,
    single_point_batch_maker: MlipSinglePointBatchMaker,
    structures_per_job: int | None,
    cohesive_energy_filtering_settings: CohesiveEnergyFilteringSettings,
) -> Response[Flow | CohesiveEnergyFilteringTaskDocument]:
    """Dynamically evaluate augmented structures before DIRECT sampling."""
    if augmentation_and_filtering_task_document.errors:
        return Response(
            output=_augmentation_failure_task_document(
                augmentation_and_filtering_task_document.errors
            )
        )

    properties = augmentation_and_filtering_task_document.properties
    source_structures: StructureList = (
        properties.structures
        if isinstance(properties, StructureAugmentationAndFilteringProperties)
        else []
    )
    if (
        not source_structures
        or cohesive_energy_filtering_settings.maximum_cohesive_energy_per_atom is None
    ):
        return Response(
            output=_build_passthrough_filtering_task_document(source_structures)
        )

    source_structure_batches = split_items_into_batches(
        items=source_structures, items_per_batch=structures_per_job
    )
    evaluation_jobs: list[Job] = [
        single_point_batch_maker.make(structures=structure_batch)
        for structure_batch in source_structure_batches
        if structure_batch
    ]
    evaluation_task_documents = [
        evaluation_job.output[index]
        for evaluation_job, structure_batch in zip(
            evaluation_jobs, source_structure_batches, strict=True
        )
        for index in range(len(structure_batch))
    ]
    source_structures_reference = OutputReference(
        source_structures_job_uuid
    ).properties.structures
    preparation_job = prepare_post_augmentation_cohesive_energy_filtering_inputs(
        source_structures=source_structures_reference,
        evaluation_task_documents=evaluation_task_documents,
    )
    filtering_job = run_cohesive_energy_filtering(
        structures=preparation_job.output.structures,
        total_energies=preparation_job.output.total_energies,
        settings=cohesive_energy_filtering_settings,
        failed_count=preparation_job.output.failed_count,
        warnings=preparation_job.output.warnings,
    )
    return Response(
        replace=Flow(
            jobs=[*evaluation_jobs, preparation_job, filtering_job],
            output=filtering_job.output,
        )
    )
