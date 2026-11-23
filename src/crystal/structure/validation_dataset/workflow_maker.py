"""Build the workflow that creates, simulates, and selects validation structures."""

from __future__ import annotations

from dataclasses import dataclass

from jobflow.core.flow import Flow, JobOrder
from jobflow.core.job import Job, JobConfig, Response, job
from jobflow.core.state import CURRENT_JOB
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.relax_maker import MlipRelaxBatchMaker, MlipRelaxMaker
from crystal.helpers.fireworks.slurm_settings import (
    build_queueadapter_metadata_from_executor_settings,
)
from crystal.schemas.base import BaseMaker
from crystal.schemas.maker_names import MakerName
from crystal.schemas.md.mlip_md_equilibration import MlipMdEquilibrationMakerSettings
from crystal.schemas.mlip import MlipRelaxTaskDocument
from crystal.schemas.properties import RelaxProperties
from crystal.schemas.relaxation_constraints import RelaxationConstraints
from crystal.schemas.structure_generators import DirectSamplingMakerSettings
from crystal.schemas.validation_dataset import (
    ValidationDatasetDirectSelectionSettings,
    ValidationDatasetMdRunKind,
    ValidationDatasetMdSamplingSettings,
    ValidationDatasetStructureBatchJobReference,
    ValidationDatasetStructureBatchReference,
    ValidationDatasetWorkflowInputs,
    ValidationDatasetWorkflowSettings,
)
from crystal.structure.validation_dataset.artifact_manifest import (
    collect_md_trajectory_manifest,
)
from crystal.structure.validation_dataset.direct_selection import (
    build_validation_dataset_direct_selection_flow,
)
from crystal.structure.validation_dataset.interface_generation import (
    generate_validation_dataset_interface_properties,
)
from crystal.structure.validation_dataset.md_execution import run_validation_dataset_md
from crystal.structure.validation_dataset.md_sampling import (
    plan_validation_dataset_md_runs,
)
from crystal.structure.validation_dataset.slab_generation import (
    generate_validation_dataset_slab_preparation_properties,
)
from crystal.structure.validation_dataset.slab_relaxation import (
    build_validation_dataset_slab_relaxation_flow,
)
from crystal.structure.validation_dataset.structure_artifacts import (
    collect_structure_batch_references,
    load_validation_dataset_structure,
)


@job
def extract_validation_dataset_relaxed_bulk_structure(
    task_document: MlipRelaxTaskDocument,
) -> PMG_Structure:
    """Take the relaxed bulk structure needed by slab generation.

    This is a workflow step because it connects the bulk relaxation to slab
    generation.
    """
    if not isinstance(task_document.properties, RelaxProperties):
        message = "Bulk MLIP relaxation did not produce relaxed structure properties."
        raise TypeError(message)
    return task_document.properties.relaxed_structure


def _build_validation_dataset_structure_source_mapping(
    slab_sources: list[ValidationDatasetStructureBatchReference],
    interface_sources: list[ValidationDatasetStructureBatchReference],
) -> tuple[
    list[str],
    list[str],
    dict[str, tuple[ValidationDatasetStructureBatchReference, int]],
]:
    """Build deterministic source IDs and compact structure references."""
    slab_ids: list[str] = []
    interface_ids: list[str] = []
    structure_references: dict[
        str, tuple[ValidationDatasetStructureBatchReference, int]
    ] = {}
    for slab_structure_batch in slab_sources:
        for index in range(slab_structure_batch.item_count):
            source_id = f"{slab_structure_batch.batch_id}:slab:{index}"
            slab_ids.append(source_id)
            structure_references[source_id] = (slab_structure_batch, index)
    for interface_structure_batch in interface_sources:
        for index in range(interface_structure_batch.item_count):
            source_id = f"{interface_structure_batch.batch_id}:interface:{index}"
            interface_ids.append(source_id)
            structure_references[source_id] = (interface_structure_batch, index)
    return slab_ids, interface_ids, structure_references


def build_validation_dataset_md_and_selection_flow(
    *,
    slab_sources: list[ValidationDatasetStructureBatchReference],
    interface_sources: list[ValidationDatasetStructureBatchReference],
    md_sampling_settings: ValidationDatasetMdSamplingSettings,
    md_maker_settings: MlipMdEquilibrationMakerSettings,
    direct_sampling_maker_settings: DirectSamplingMakerSettings,
    direct_selection_settings: ValidationDatasetDirectSelectionSettings,
) -> Response[Flow]:
    """Build bounded MD fanout and the manifest-to-DIRECT tail at runtime."""
    slab_ids, interface_ids, structure_references = (
        _build_validation_dataset_structure_source_mapping(
            slab_sources=slab_sources, interface_sources=interface_sources
        )
    )
    run_specs = plan_validation_dataset_md_runs(
        slab_ids=slab_ids,
        interface_ids=interface_ids,
        md_sampling_settings=md_sampling_settings,
    )
    md_jobs: list[Job] = []
    structure_loader_jobs: list[Job] = []
    structure_loader_jobs_by_source_id: dict[str, Job] = {}
    md_job_uuids: list[str] = []
    for run_spec in run_specs:
        structure_batch, structure_index = structure_references[run_spec.source_id]
        structure_loader_job = structure_loader_jobs_by_source_id.get(
            run_spec.source_id
        )
        if structure_loader_job is None:
            structure_loader_job = load_validation_dataset_structure(
                structure_batch=structure_batch, structure_index=structure_index
            )
            structure_loader_jobs_by_source_id[run_spec.source_id] = (
                structure_loader_job
            )
            structure_loader_jobs.append(structure_loader_job)
        md_job = run_validation_dataset_md(
            run_spec=run_spec,
            structure=structure_loader_job.output,
            simulation_settings=md_maker_settings.simulation_settings,
            requested_gpu_count=md_maker_settings.executor_settings.number_of_gpus,
        )
        md_job.update_metadata(
            build_queueadapter_metadata_from_executor_settings(
                md_maker_settings.executor_settings
            ),
            dynamic=False,
        )
        md_jobs.append(md_job)
        md_job_uuids.append(md_job.uuid)

    manifest_job = collect_md_trajectory_manifest(md_job_uuids=md_job_uuids)
    direct_flow = build_validation_dataset_direct_selection_flow(
        direct_sampling_maker_settings=direct_sampling_maker_settings,
        direct_selection_settings=direct_selection_settings,
        trajectory_artifacts=manifest_job.output,  # type: ignore[arg-type]
    )
    md_execution_flow = Flow(jobs=[*structure_loader_jobs, *md_jobs])
    md_manifest_frame_flow = Flow(
        jobs=[md_execution_flow, manifest_job], order=JobOrder.LINEAR
    )
    return Response(
        replace=Flow(
            jobs=[md_manifest_frame_flow, direct_flow], output=direct_flow.output
        )
    )


@job(config=JobConfig(expose_store=True))
def fanout_validation_dataset_md_and_selection(
    *,
    slab_batch_job_references: list[ValidationDatasetStructureBatchJobReference],
    interface_batch_job_references: list[ValidationDatasetStructureBatchJobReference],
    md_sampling_settings: ValidationDatasetMdSamplingSettings,
    md_maker_settings: MlipMdEquilibrationMakerSettings,
    direct_sampling_maker_settings: DirectSamplingMakerSettings,
    direct_selection_settings: ValidationDatasetDirectSelectionSettings,
) -> Response[Flow]:
    """Start the MD simulations and final structure selection.

    This is a workflow step because the number of MD jobs is known only after the
    preparation jobs finish.
    """
    return build_validation_dataset_md_and_selection_flow(
        slab_sources=collect_structure_batch_references(
            slab_batch_job_references, CURRENT_JOB.store
        ),
        interface_sources=collect_structure_batch_references(
            interface_batch_job_references, CURRENT_JOB.store
        ),
        md_sampling_settings=md_sampling_settings,
        md_maker_settings=md_maker_settings,
        direct_sampling_maker_settings=direct_sampling_maker_settings,
        direct_selection_settings=direct_selection_settings,
    )


@dataclass(kw_only=True)
class ValidationDatasetWorkflowMaker(BaseMaker):
    """Build the static preparation and dynamic MD/selection workflow."""

    name: MakerName = MakerName.validation_dataset_workflow
    simulation_settings: ValidationDatasetWorkflowSettings

    @property
    def _bulk_relax_maker(self) -> MlipRelaxMaker:
        relax_maker_settings = self.simulation_settings.mlip_relax_maker_settings
        bulk_relaxation_constraints = (
            relax_maker_settings.simulation_settings.constraints
            or RelaxationConstraints()
        )
        bulk_relaxation_settings = relax_maker_settings.simulation_settings.model_copy(
            update={
                "constraints": bulk_relaxation_constraints.model_copy(
                    update={"fix_cell": False}
                )
            }
        )
        return MlipRelaxMaker(
            simulation_settings=bulk_relaxation_settings,
            executor_settings=relax_maker_settings.executor_settings,
        )

    def make(self, inputs: ValidationDatasetWorkflowInputs) -> Flow:
        """Build the complete reference-based validation-dataset workflow."""
        workflow_settings = self.simulation_settings
        preparation_executor_settings = (
            workflow_settings.mlip_relax_maker_settings.executor_settings
        )
        preparation_queueadapter_metadata = (
            build_queueadapter_metadata_from_executor_settings(
                preparation_executor_settings
            )
        )
        jobs: list[Job | Flow] = []
        slab_output_sources = []
        slab_batch_job_references: list[
            ValidationDatasetStructureBatchJobReference
        ] = []
        interface_batch_job_references: list[
            ValidationDatasetStructureBatchJobReference
        ] = []
        bulk_relax_maker = self._bulk_relax_maker
        for structure_input in inputs.bulk_structures:
            bulk_relax_job = bulk_relax_maker.make(structure=structure_input.structure)
            jobs.append(bulk_relax_job)
            relaxed_structure_job = extract_validation_dataset_relaxed_bulk_structure(
                task_document=bulk_relax_job.output
            )
            jobs.append(relaxed_structure_job)
            slab_generation_job = (
                generate_validation_dataset_slab_preparation_properties(
                    source_structure_id=structure_input.structure_id,
                    structure=relaxed_structure_job.output,
                    slab_generation_settings=workflow_settings.slab_generation,
                )
            )
            slab_generation_job.update_metadata(
                preparation_queueadapter_metadata, dynamic=False
            )
            jobs.append(slab_generation_job)
            slab_relaxation_flow = build_validation_dataset_slab_relaxation_flow(
                source_structure_id=structure_input.structure_id,
                slabs=slab_generation_job.output.slabs,
                relax_maker=MlipRelaxBatchMaker(
                    simulation_settings=(
                        workflow_settings.mlip_relax_maker_settings.simulation_settings
                    ),
                    executor_settings=preparation_executor_settings,
                ),
            )
            jobs.append(slab_relaxation_flow)
            collection_job: Job = slab_relaxation_flow.jobs[-1]  # type: ignore[assignment]
            slab_batch_job_references.append(
                ValidationDatasetStructureBatchJobReference(
                    batch_id=structure_input.structure_id,
                    source_kind=ValidationDatasetMdRunKind.slab,
                    job_uuid=collection_job.uuid,
                )
            )
            slab_output_sources.append(
                (structure_input.structure_id, slab_relaxation_flow.output.slabs)
            )

        for top_source_id, top_slabs in slab_output_sources:
            for bottom_source_id, bottom_slabs in slab_output_sources:
                interface_job: Job = generate_validation_dataset_interface_properties(
                    top_slabs=top_slabs,
                    bottom_slabs=bottom_slabs,
                    interface_generation_settings=workflow_settings.interface_generation,
                )
                interface_job.update_metadata(
                    preparation_queueadapter_metadata, dynamic=False
                )
                jobs.append(interface_job)
                interface_batch_job_references.append(
                    ValidationDatasetStructureBatchJobReference(
                        batch_id=f"{top_source_id}:{bottom_source_id}",
                        source_kind=ValidationDatasetMdRunKind.interface,
                        job_uuid=interface_job.uuid,
                    )
                )

        preparation_flow = Flow(jobs=jobs)
        fanout_job = fanout_validation_dataset_md_and_selection(
            slab_batch_job_references=slab_batch_job_references,
            interface_batch_job_references=interface_batch_job_references,
            md_sampling_settings=workflow_settings.md_sampling,
            md_maker_settings=workflow_settings.md_equilibration_maker_settings,
            direct_sampling_maker_settings=(
                workflow_settings.direct_sampling_maker_settings
            ),
            direct_selection_settings=workflow_settings.direct_selection,
        )
        fanout_job.update_metadata(
            self.slurm_settings.to_queueadapter_spec(), dynamic=False
        )
        return Flow(
            jobs=[preparation_flow, fanout_job],
            order=JobOrder.LINEAR,
            output=fanout_job.output,
        )
