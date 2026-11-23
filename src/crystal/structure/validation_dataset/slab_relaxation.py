"""Re-relax generated slabs and keep the ones that succeed."""

from __future__ import annotations

from collections.abc import Sequence

from jobflow.core.flow import Flow
from jobflow.core.job import Job, job

from crystal.calculators.mlip.relax_maker import MlipRelaxBatchMaker
from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.flows.adhesion_workflow_helpers import (
    build_slab_from_structure_with_slab_metadata,
)
from crystal.schemas.mlip import MlipRelaxTaskDocument
from crystal.schemas.properties import RelaxProperties
from crystal.schemas.validation_dataset import (
    ValidationDatasetSlabPreparationProperties,
)
from crystal.structure.slabs.builder import Slab


def collect_successfully_relaxed_slabs(
    *,
    source_structure_id: str,
    input_slabs: Sequence[Slab],
    relax_task_documents: Sequence[MlipRelaxTaskDocument],
) -> ValidationDatasetSlabPreparationProperties:
    """Keep successful relaxed slabs and restore their slab metadata."""
    if len(input_slabs) != len(relax_task_documents):
        message = "Input slab and relaxation document counts must match."
        raise ValueError(message)

    successful_slabs: list[Slab] = []
    failed_relaxation_count = 0
    for slab, task_document in zip(input_slabs, relax_task_documents, strict=True):
        if not isinstance(task_document.properties, RelaxProperties):
            failed_relaxation_count += 1
            continue
        relaxed_slab = build_slab_from_structure_with_slab_metadata(
            slab=slab, structure=task_document.properties.relaxed_structure
        )
        successful_slabs.append(relaxed_slab)

    return ValidationDatasetSlabPreparationProperties(
        source_structure_id=source_structure_id,
        slabs=successful_slabs,
        failed_relaxation_count=failed_relaxation_count,
    )


@job(**{STRUCTURE_BLOB_STORE_NAME: [["output", "slabs"]]})  # type: ignore[call-overload]
def collect_successfully_relaxed_slab_properties(
    *,
    source_structure_id: str,
    input_slabs: list[Slab],
    relax_task_documents: list[MlipRelaxTaskDocument],
) -> ValidationDatasetSlabPreparationProperties:
    """Keep the slabs whose MLIP relaxations succeeded.

    This is a workflow step because the resulting structures are passed to later
    stages through the configured data store.
    """
    return collect_successfully_relaxed_slabs(
        source_structure_id=source_structure_id,
        input_slabs=input_slabs,
        relax_task_documents=relax_task_documents,
    )


def build_validation_dataset_slab_relaxation_flow(
    *, source_structure_id: str, slabs: list[Slab], relax_maker: MlipRelaxBatchMaker
) -> Flow:
    """Build the MLIP batch-relaxation and successful-collection flow."""
    relax_job: Job = relax_maker.make(structures=slabs)  # type: ignore[arg-type]
    collection_job: Job = collect_successfully_relaxed_slab_properties(
        source_structure_id=source_structure_id,
        input_slabs=slabs,
        relax_task_documents=relax_job.output,
    )
    return Flow(jobs=[relax_job, collection_job], output=collection_job.output)
