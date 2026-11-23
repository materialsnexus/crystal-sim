"""Tests for loading QET labels from stored Jobflow outputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from jobflow.core.job import job
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import MemoryStore

from crystal.repositories.qet_finetuning_label_repository import (
    load_labelled_documents_for_qet_fine_tuning,
)
from crystal.schemas.qet_finetuning import QetFineTuningStoredJobQuery
from tests.helpers.qet_finetuning import (
    create_gpaw_label_task_document,
    store_gpaw_label_task_document_for_test,
)


@job
def _store_unsupported_label_output() -> str:
    return "not a GPAW label"


def test_loads_and_orders_stored_label_documents_by_job_identity(
    tmp_path: Path,
) -> None:
    jobstore = JobStore(docs_store=MemoryStore())
    jobstore.connect()
    label_jobs = []
    for index in (1, 2):
        label_job = store_gpaw_label_task_document_for_test(
            create_gpaw_label_task_document(index=index)
        )
        label_job.update_metadata(
            {"label_query_group": "qet", "label_index": index}, dynamic=False
        )
        run_locally(label_job, root_dir=tmp_path, create_folders=True, store=jobstore)
        label_jobs.append(label_job)

    loaded_documents = load_labelled_documents_for_qet_fine_tuning(
        jobstore=jobstore,
        stored_job_query=QetFineTuningStoredJobQuery(
            criteria={"metadata.label_query_group": "qet"}
        ),
    )

    expected_documents_by_uuid = {
        label_job.uuid: f"label-{index}"
        for label_job, index in zip(label_jobs, (1, 2), strict=True)
    }
    expected_document_names = [
        expected_documents_by_uuid[label_job.uuid]
        for label_job in sorted(label_jobs, key=lambda item: item.uuid)
    ]
    assert [document.dir_name for document in loaded_documents] == (
        expected_document_names
    )


def test_empty_stored_job_query_returns_no_label_documents() -> None:
    jobstore = JobStore(docs_store=MemoryStore())
    jobstore.connect()

    loaded_documents = load_labelled_documents_for_qet_fine_tuning(
        jobstore=jobstore,
        stored_job_query=QetFineTuningStoredJobQuery(
            criteria={"metadata.label_query_group": "missing"}
        ),
    )

    assert loaded_documents == []


def test_rejects_unsupported_stored_job_output(tmp_path: Path) -> None:
    jobstore = JobStore(docs_store=MemoryStore())
    jobstore.connect()
    unsupported_job = _store_unsupported_label_output()
    unsupported_job.update_metadata({"label_query_group": "qet"}, dynamic=False)
    run_locally(unsupported_job, root_dir=tmp_path, create_folders=True, store=jobstore)

    with pytest.raises(TypeError, match="unsupported output: str"):
        load_labelled_documents_for_qet_fine_tuning(
            jobstore=jobstore,
            stored_job_query=QetFineTuningStoredJobQuery(
                criteria={"metadata.label_query_group": "qet"}
            ),
        )
