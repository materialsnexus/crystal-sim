"""Workflow maker for scalable structure-dataset generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from jobflow.core.flow import Flow
from jobflow.core.job import Job, Response, job
from jobflow.core.reference import OutputReference

from crystal.calculators.mlip.relax_maker import MlipRelaxBatchMaker
from crystal.calculators.mlip.single_point_maker import MlipSinglePointBatchMaker
from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.helpers.structure import split_items_into_batches
from crystal.schemas.base import BaseMaker
from crystal.schemas.filtering import CohesiveEnergyFilteringTaskDocument
from crystal.schemas.filtering_statistics import CohesiveEnergyFilteringStatistics
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationProperties,
)
from crystal.schemas.maker_names import MakerName
from crystal.schemas.mlip import MlipRelaxTaskDocument
from crystal.schemas.properties import (
    CohesiveEnergyFilteringProperties,
    StructureGenerationProperties,
    StructureList,
)
from crystal.schemas.structure_augmentation import (
    StructureAugmentationAndFilteringProperties,
)
from crystal.schemas.structure_generators import (
    DatasetGenerationTaskDocUnion,
    DatasetGenerationWorkflowSettings,
    DatasetGenerationWorkflowTaskDocument,
    DirectSamplingTaskDocument,
    InitialStructureGenerationTaskDocument,
    PostAugmentationCohesiveEnergyFilteringSettings,
    StructureAugmentationAndFilteringTaskDocument,
)
from crystal.structure.generators.augmentation_filtering_maker import (
    StructureAugmentationAndFilteringMaker,
)
from crystal.structure.generators.post_augmentation_filtering import (
    build_post_augmentation_cohesive_energy_filtering_flow,
)
from crystal.structure.generators.pyxtal.initial_structure_generation_maker import (
    InitialStructureGenerationMaker,
)
from crystal.structure.generators.sampling.direct_maker import DirectSamplingMaker
from crystal.structure.labelling.structure_labelling_workflow_maker import (
    StructureLabellingWorkflowMaker,
)


class StageLabel(StrEnum):
    """Stages in the dataset-generation workflow."""

    structure_generation = "Structure Generation"
    mlip_relax = "MLIP Relax"
    structure_augmentation_and_filtering = "Structure Augmentation and Filtering"
    post_augmentation_cohesive_energy_filtering = (
        "Post-Augmentation Cohesive Energy Filtering"
    )
    direct_sampling = "DIRECT Sampling"


@dataclass(kw_only=True)
class DatasetGenerationWorkflowMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.dataset_generation_workflow
    simulation_settings: DatasetGenerationWorkflowSettings

    @property
    def _initial_structure_generation_maker(self) -> InitialStructureGenerationMaker:
        settings = self.simulation_settings.initial_structure_generation_maker_settings
        return InitialStructureGenerationMaker(
            simulation_settings=settings.simulation_settings,
            executor_settings=settings.executor_settings,
        )

    @property
    def _mlip_relax_batch_maker(self) -> MlipRelaxBatchMaker:
        settings = self.simulation_settings.mlip_relax_maker_settings
        return MlipRelaxBatchMaker(
            simulation_settings=settings.simulation_settings.model_copy(
                update={"store_final_step_properties_only": True}
            ),
            executor_settings=settings.executor_settings,
        )

    @property
    def _structure_augmentation_and_filtering_maker(
        self,
    ) -> StructureAugmentationAndFilteringMaker:
        settings = (
            self.simulation_settings.structure_augmentation_and_filtering_maker_settings
        )
        return StructureAugmentationAndFilteringMaker(
            simulation_settings=settings.simulation_settings,
            executor_settings=settings.executor_settings,
        )

    @property
    def _direct_sampling_maker(self) -> DirectSamplingMaker:
        settings = self.simulation_settings.direct_sampling_maker_settings
        return DirectSamplingMaker(
            simulation_settings=settings.simulation_settings,
            executor_settings=settings.executor_settings,
        )

    def _post_augmentation_mlip_single_point_batch_maker(
        self, settings: PostAugmentationCohesiveEnergyFilteringSettings
    ) -> MlipSinglePointBatchMaker:
        return MlipSinglePointBatchMaker(
            simulation_settings=settings.mlip_single_point_maker_settings.simulation_settings,
            executor_settings=settings.mlip_single_point_maker_settings.executor_settings,
        )

    @property
    def _structure_labelling_workflow_maker(
        self,
    ) -> StructureLabellingWorkflowMaker | None:
        settings = self.simulation_settings.structure_labelling_workflow_settings
        if settings is None:
            return None
        return StructureLabellingWorkflowMaker(simulation_settings=settings)

    def _apply_workflow_executor_settings(self, jobs: list[Job]) -> None:
        for selected_job in jobs:
            selected_job.update_metadata(
                self.slurm_settings.to_queueadapter_spec(), dynamic=False
            )

    def make(self) -> Flow:  # noqa: D102
        initial_job = self._initial_structure_generation_maker.make(
            number_of_structures=self.simulation_settings.number_of_initial_structures
        )
        mlip_flow = build_mlip_relaxation_flow_after_initial_structure_generation(
            initial_structure_generation_task_document=initial_job.output,
            mlip_relax_batch_maker=self._mlip_relax_batch_maker,
            structures_per_job=self.simulation_settings.mlip_relax_structures_per_job,
        )
        augmentation_job = self._structure_augmentation_and_filtering_maker.make(
            initial_structure_generation_task_document=initial_job.output,  # type: ignore[arg-type]
            mlip_relax_task_documents=mlip_flow.output,  # type: ignore[arg-type]
        )
        post_augmentation_settings = self.simulation_settings.post_augmentation_cohesive_energy_filtering_settings
        post_augmentation_filtering_flow: Job | None = None
        structure_source_properties = augmentation_job.output.properties
        structure_source_job_uuid = augmentation_job.uuid
        if post_augmentation_settings is not None:
            post_augmentation_filtering_flow = (
                build_post_augmentation_cohesive_energy_filtering_flow(
                    augmentation_and_filtering_task_document=augmentation_job.output,
                    source_structures_job_uuid=augmentation_job.uuid,
                    single_point_batch_maker=(
                        self._post_augmentation_mlip_single_point_batch_maker(
                            post_augmentation_settings
                        )
                    ),
                    structures_per_job=post_augmentation_settings.structures_per_job,
                    cohesive_energy_filtering_settings=(
                        post_augmentation_settings.cohesive_energy_filtering_settings
                    ),
                )
            )
            structure_source_properties = (
                post_augmentation_filtering_flow.output.properties
            )
            structure_source_job_uuid = post_augmentation_filtering_flow.uuid
        direct_flow = build_direct_sampling_flow_after_augmentation_and_filtering(
            structure_source_properties=structure_source_properties,
            source_structures_job_uuid=structure_source_job_uuid,
            direct_sampling_maker=self._direct_sampling_maker,
        )
        structure_labelling_job = build_structure_labelling_flow_after_direct_sampling(
            direct_sampling_task_document=direct_flow.output,
            structure_labelling_workflow_maker=self._structure_labelling_workflow_maker,
        )
        workflow_task_document = build_structure_generation_workflow_task_document(
            initial_structure_generation_task_document=initial_job.output,
            mlip_relax_task_documents=mlip_flow.output,
            direct_sampling_task_document=direct_flow.output,
            augmentation_and_filtering_task_document=augmentation_job.output,
            post_augmentation_cohesive_energy_filtering_task_document=(
                post_augmentation_filtering_flow.output
                if post_augmentation_filtering_flow is not None
                else None
            ),
        )
        static_jobs = [mlip_flow]
        jobs = [initial_job, mlip_flow, augmentation_job]
        if post_augmentation_filtering_flow is not None:
            static_jobs.append(post_augmentation_filtering_flow)
            jobs.append(post_augmentation_filtering_flow)
        static_jobs.extend(
            [direct_flow, structure_labelling_job, workflow_task_document]
        )
        jobs.extend([direct_flow, structure_labelling_job, workflow_task_document])
        self._apply_workflow_executor_settings(static_jobs)
        return Flow(jobs=jobs, output=workflow_task_document.output)


def _get_task_document_error(
    task_document: DatasetGenerationTaskDocUnion,
) -> str | None:
    """Return a task failure diagnostic, including early PyXtal termination."""
    if (
        isinstance(task_document.properties, InitialStructureGenerationProperties)
        and task_document.properties.terminated_early
    ):
        return (
            "PyXtal generation stopped after generating "
            f"{len(task_document.properties.structures)} of "
            f"{task_document.properties.generation_range.number_of_structures} "
            "requested structures."
        )
    if task_document.properties is None:
        return task_document.errors or None
    return None


def _get_task_document_warning(
    task_document: DatasetGenerationTaskDocUnion,
) -> str | None:
    """Return the recoverable warning recorded by one task document."""
    if isinstance(
        task_document,
        (
            StructureAugmentationAndFilteringTaskDocument,
            CohesiveEnergyFilteringTaskDocument,
            DirectSamplingTaskDocument,
        ),
    ):
        return task_document.warnings or None
    return None


def _format_stage_warnings(
    stage_label: StageLabel, task_documents: Sequence[DatasetGenerationTaskDocUnion]
) -> str:
    if not task_documents:
        return f"{stage_label}: 0 task documents received.\n"
    errors = [
        error
        for task_document in task_documents
        if (error := _get_task_document_error(task_document))
    ]
    warnings = [
        warning
        for task_document in task_documents
        if (warning := _get_task_document_warning(task_document))
    ]
    messages: list[str] = []
    if errors:
        failure = f"{stage_label}: {len(errors)}/{len(task_documents)} tasks failed."
        failure += f"\nExample error: {errors[0]}."
        messages.append(failure)
    if warnings:
        warning = f"{stage_label}: {len(warnings)} non-fatal warnings."
        warning += f"\nExample warning: {warnings[0]}."
        messages.append(warning)
    return "\n\n".join(messages)


def _build_workflow_warnings(
    *,
    initial_structure_generation_task_document: InitialStructureGenerationTaskDocument,
    mlip_relax_task_documents: list[MlipRelaxTaskDocument],
    augmentation_and_filtering_task_document: StructureAugmentationAndFilteringTaskDocument,
    direct_sampling_task_document: DirectSamplingTaskDocument,
    post_augmentation_cohesive_energy_filtering_task_document: CohesiveEnergyFilteringTaskDocument
    | None = None,
) -> str:
    """Collect warnings and failures from completed stages."""
    stage_documents: list[
        tuple[StageLabel, Sequence[DatasetGenerationTaskDocUnion]]
    ] = [
        (StageLabel.structure_generation, [initial_structure_generation_task_document]),
        (StageLabel.mlip_relax, mlip_relax_task_documents),
        (
            StageLabel.structure_augmentation_and_filtering,
            [augmentation_and_filtering_task_document],
        ),
    ]
    if post_augmentation_cohesive_energy_filtering_task_document is not None:
        stage_documents.append(
            (
                StageLabel.post_augmentation_cohesive_energy_filtering,
                [post_augmentation_cohesive_energy_filtering_task_document],
            )
        )
    stage_documents.append(
        (StageLabel.direct_sampling, [direct_sampling_task_document])
    )
    return "\n\n".join(
        warning
        for stage_label, task_documents in stage_documents
        if (warning := _format_stage_warnings(stage_label, task_documents))
    )


def _build_mlip_relaxation_flow(
    structures: StructureList,
    mlip_relax_batch_maker: MlipRelaxBatchMaker,
    structures_per_job: int | None = None,
) -> Flow:
    """Build MLIP batch jobs for resolved structures."""
    if not structures:
        return Flow(jobs=[], output=[])

    batches = split_items_into_batches(
        items=structures, items_per_batch=structures_per_job
    )
    mlip_jobs = [
        mlip_relax_batch_maker.make(structures=batch) for batch in batches if batch
    ]
    return Flow(
        jobs=mlip_jobs,
        output=[
            mlip_job.output[index]
            for mlip_job, batch in zip(mlip_jobs, batches, strict=True)
            for index in range(len(batch))
        ],
    )


@job
def build_mlip_relaxation_flow_after_initial_structure_generation(
    initial_structure_generation_task_document: InitialStructureGenerationTaskDocument,
    mlip_relax_batch_maker: MlipRelaxBatchMaker,
    structures_per_job: int | None = None,
) -> Response[Flow]:
    """Run MLIP batches that retain only final-step properties."""
    initial_properties = initial_structure_generation_task_document.properties
    if not isinstance(initial_properties, InitialStructureGenerationProperties):
        structures = []
    else:
        structures = initial_properties.structures
    return Response(
        replace=_build_mlip_relaxation_flow(
            structures=structures,
            mlip_relax_batch_maker=mlip_relax_batch_maker,
            structures_per_job=structures_per_job,
        )
    )


def _build_direct_sampling_flow(
    structures: StructureList | OutputReference,
    direct_sampling_maker: DirectSamplingMaker,
) -> Flow:
    """Build the global DIRECT sampling job for resolved structures."""
    direct_job = direct_sampling_maker.make(
        structures=structures  # type: ignore[arg-type]
    )
    return Flow(jobs=[direct_job], output=direct_job.output)


@job
def build_direct_sampling_flow_after_augmentation_and_filtering(
    structure_source_properties: (
        CohesiveEnergyFilteringProperties
        | StructureAugmentationAndFilteringProperties
        | None
    ),
    source_structures_job_uuid: str,
    direct_sampling_maker: DirectSamplingMaker,
) -> Response[Flow]:
    """Run DIRECT on augmented structures, optionally after filtering."""
    structures: StructureList | OutputReference = []
    if isinstance(
        structure_source_properties,
        (
            CohesiveEnergyFilteringProperties,
            StructureAugmentationAndFilteringProperties,
        ),
    ):
        structures = OutputReference(source_structures_job_uuid).properties.structures
    return Response(
        replace=_build_direct_sampling_flow(
            structures=structures, direct_sampling_maker=direct_sampling_maker
        )
    )


def _build_structure_labelling_flow(
    source_structures: StructureList,
    structure_labelling_workflow_maker: StructureLabellingWorkflowMaker,
) -> Flow:
    """Build the structure-labelling flow for resolved structures."""
    return structure_labelling_workflow_maker.make(source_structures=source_structures)


@job
def build_structure_labelling_flow_after_direct_sampling(
    direct_sampling_task_document: DirectSamplingTaskDocument,
    structure_labelling_workflow_maker: StructureLabellingWorkflowMaker | None,
) -> Response[Flow | None]:
    """Label structures only when DIRECT completed successfully."""
    properties = direct_sampling_task_document.properties
    if (
        not isinstance(properties, StructureGenerationProperties)
        or structure_labelling_workflow_maker is None
    ):
        return Response(output=None)

    return Response(
        replace=_build_structure_labelling_flow(
            source_structures=properties.structures,
            structure_labelling_workflow_maker=structure_labelling_workflow_maker,
        )
    )


@job(
    output_schema=DatasetGenerationWorkflowTaskDocument,
    **{STRUCTURE_BLOB_STORE_NAME: [["output", "properties", "structures"]]},
)  # type: ignore[call-overload]
def build_structure_generation_workflow_task_document(
    *,
    initial_structure_generation_task_document: InitialStructureGenerationTaskDocument,
    mlip_relax_task_documents: list[MlipRelaxTaskDocument],
    direct_sampling_task_document: DirectSamplingTaskDocument,
    augmentation_and_filtering_task_document: StructureAugmentationAndFilteringTaskDocument,
    post_augmentation_cohesive_energy_filtering_task_document: CohesiveEnergyFilteringTaskDocument
    | None = None,
) -> DatasetGenerationWorkflowTaskDocument:
    """Build the pre-labeling structure-generation document.

    The actual labels are stored in the individual task documents produced by
    the separate structure-labelling fan-out jobs.
    """
    warnings = _build_workflow_warnings(
        initial_structure_generation_task_document=initial_structure_generation_task_document,
        mlip_relax_task_documents=mlip_relax_task_documents,
        augmentation_and_filtering_task_document=augmentation_and_filtering_task_document,
        post_augmentation_cohesive_energy_filtering_task_document=(
            post_augmentation_cohesive_energy_filtering_task_document
        ),
        direct_sampling_task_document=direct_sampling_task_document,
    )
    augmentation_properties = augmentation_and_filtering_task_document.properties
    if not isinstance(
        augmentation_properties, StructureAugmentationAndFilteringProperties
    ):
        return DatasetGenerationWorkflowTaskDocument(
            dir_name=augmentation_and_filtering_task_document.dir_name,
            errors=get_diagnostic_exception_message(
                RuntimeError(augmentation_and_filtering_task_document.errors)
            ),
            properties=None,
            warnings=warnings,
        )
    post_augmentation_statistics: CohesiveEnergyFilteringStatistics | None = None
    if post_augmentation_cohesive_energy_filtering_task_document is not None:
        post_augmentation_properties = (
            post_augmentation_cohesive_energy_filtering_task_document.properties
        )
        if not isinstance(
            post_augmentation_properties, CohesiveEnergyFilteringProperties
        ):
            return DatasetGenerationWorkflowTaskDocument(
                dir_name=post_augmentation_cohesive_energy_filtering_task_document.dir_name,
                errors=post_augmentation_cohesive_energy_filtering_task_document.errors,
                properties=None,
                warnings=warnings,
                augmentation_and_filtering_statistics=augmentation_properties.statistics,
            )
        post_augmentation_statistics = post_augmentation_properties.statistics
    return DatasetGenerationWorkflowTaskDocument(
        dir_name=direct_sampling_task_document.dir_name,
        errors=direct_sampling_task_document.errors,
        properties=direct_sampling_task_document.properties,
        warnings=warnings,
        augmentation_and_filtering_statistics=augmentation_properties.statistics,
        post_augmentation_cohesive_energy_filtering_statistics=post_augmentation_statistics,
    )
