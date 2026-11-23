"""Store generated structures and load them when later jobs need them."""

from __future__ import annotations

from collections.abc import Iterable

from jobflow.core.job import JobConfig, job
from jobflow.core.state import CURRENT_JOB
from jobflow.core.store import JobStore
from monty.json import MontyDecoder
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.schemas.validation_dataset import (
    ValidationDatasetArtifactReference,
    ValidationDatasetStructureBatchJobReference,
    ValidationDatasetStructureBatchReference,
)
from crystal.structure.validation_dataset.artifact_storage import (
    get_artifact_document,
    get_latest_execution_output_for_job_uuid,
)

STRUCTURE_BATCH_COUNT_FIELDS = {"slabs": "slab_count", "interfaces": "interface_count"}


def _batch_reference_from_output(
    preparation_job_output: dict[str, object],
    job_reference: ValidationDatasetStructureBatchJobReference,
) -> ValidationDatasetStructureBatchReference:
    pointer_error = "Structure batch output does not contain a blob pointer."
    batch_properties = preparation_job_output.get("properties")
    if not isinstance(batch_properties, dict):
        raise TypeError(pointer_error)

    output_field = (
        "slabs" if job_reference.source_kind.value == "slab" else "interfaces"
    )
    structure_batch_pointer = batch_properties.get(output_field)
    count_field = STRUCTURE_BATCH_COUNT_FIELDS[output_field]
    item_count = batch_properties.get(count_field)
    if not isinstance(structure_batch_pointer, dict) or not isinstance(item_count, int):
        raise TypeError(pointer_error)

    artifact_id = structure_batch_pointer.get("blob_uuid")
    store_name = structure_batch_pointer.get("store")
    if not isinstance(artifact_id, str) or not isinstance(store_name, str):
        raise TypeError(pointer_error)

    return ValidationDatasetStructureBatchReference(
        batch_id=job_reference.batch_id,
        structure_artifact=ValidationDatasetArtifactReference(
            artifact_id=artifact_id, store_name=store_name
        ),
        item_count=item_count,
    )


def collect_structure_batch_references(
    job_references: Iterable[ValidationDatasetStructureBatchJobReference],
    jobstore: JobStore,
) -> list[ValidationDatasetStructureBatchReference]:
    """Resolve preparation-job documents into compact structure-batch pointers."""
    batch_references: list[ValidationDatasetStructureBatchReference] = []
    for job_reference in job_references:
        output = get_latest_execution_output_for_job_uuid(
            jobstore, job_reference.job_uuid
        )
        if output is None:
            message = f"Preparation job output is unavailable: {job_reference.job_uuid}"
            raise ValueError(message)
        batch_references.append(_batch_reference_from_output(output, job_reference))
    return batch_references


def _load_structure_batch_data(
    structure_batch: ValidationDatasetStructureBatchReference, jobstore: JobStore
) -> list[PMG_Structure]:
    structure_store = jobstore.additional_stores[  # type: ignore[has-type]
        structure_batch.structure_artifact.store_name
    ]

    document = get_artifact_document(
        store=structure_store, artifact=structure_batch.structure_artifact
    )
    if document is None:
        message = "Validation-dataset structure batch blob was not found."
        raise KeyError(message)

    decoded_data = MontyDecoder().process_decoded(document.get("data"))
    if not isinstance(decoded_data, list) or not all(
        isinstance(structure, PMG_Structure) for structure in decoded_data
    ):
        message = "Validation-dataset structure batch blob is not a structure list."
        raise TypeError(message)
    if len(decoded_data) != structure_batch.item_count:
        message = (
            "Expected structure batch count does not match retrieved data. "
            f"Expected {structure_batch.item_count}; retrieved {len(decoded_data)}."
        )
        raise ValueError(message)
    return decoded_data


@job(config=JobConfig(expose_store=True), **{STRUCTURE_BLOB_STORE_NAME: [["output"]]})  # type: ignore[call-overload]
def load_validation_dataset_structure(
    *, structure_batch: ValidationDatasetStructureBatchReference, structure_index: int
) -> PMG_Structure:
    """Load one prepared structure for an MD simulation.

    This is a workflow step because prepared structures are kept in the data store
    instead of being copied through the workflow record.
    """
    if structure_index < 0 or structure_index >= structure_batch.item_count:
        message = "Validation-dataset structure index is outside its batch."
        raise IndexError(message)

    structures = _load_structure_batch_data(
        structure_batch=structure_batch, jobstore=CURRENT_JOB.store
    )
    return structures[structure_index]
