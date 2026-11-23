"""JobStore access for persisted QET fine-tuning labels."""

from __future__ import annotations

from typing import Any

from jobflow.core.store import JobStore
from monty.json import MontyDecoder

from crystal.schemas.gpaw import GpawScfChargemolTaskDocument
from crystal.schemas.qet_finetuning import QetFineTuningStoredJobQuery


def load_labelled_documents_for_qet_fine_tuning(
    *, jobstore: JobStore, stored_job_query: QetFineTuningStoredJobQuery
) -> list[GpawScfChargemolTaskDocument]:
    """Load and validate QET label documents selected by a JobStore query."""
    stored_job_documents = jobstore.query(
        criteria=stored_job_query.criteria,
        properties=["output"],
        sort={"uuid": 1, "index": 1},
        load=True,
    )
    label_task_documents: list[GpawScfChargemolTaskDocument] = []
    for stored_job_document in stored_job_documents:
        stored_output: Any = stored_job_document.get("output")
        decoded_output = MontyDecoder().process_decoded(stored_output)
        if not isinstance(decoded_output, GpawScfChargemolTaskDocument):
            msg = (
                "QET fine-tuning stored-job query returned an unsupported output: "
                f"{type(decoded_output).__name__}."
            )
            raise TypeError(msg)
        label_task_documents.append(decoded_output)

    return label_task_documents
