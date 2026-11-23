"""Read job results and stored data without loading large files into job data."""

from __future__ import annotations

from typing import Any

from jobflow.core.store import JobStore
from maggma.core.store import Store

from crystal.schemas.validation_dataset import ValidationDatasetArtifactReference


def get_latest_execution_output_for_job_uuid(
    jobstore: JobStore, job_uuid: str
) -> dict[str, Any] | None:
    """Return the highest-index execution output without resolving artifacts."""
    documents = jobstore.docs_store.query(
        criteria={"uuid": job_uuid}, properties=["output"], sort={"index": -1}, limit=1
    )
    document = next(iter(documents), None)
    if document is None:
        return None
    output = document.get("output")
    return output if isinstance(output, dict) else None


def get_artifact_document(
    *, store: Store, artifact: ValidationDatasetArtifactReference
) -> dict[str, Any] | None:
    """Return one raw artifact document without interpreting its data payload."""
    return next(
        iter(
            store.query(
                criteria={"blob_uuid": artifact.artifact_id}, properties=["data"]
            )
        ),
        None,
    )
