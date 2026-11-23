"""Find completed MD trajectories that can be used for validation-data selection."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jobflow.core.job import JobConfig, job
from jobflow.core.state import CURRENT_JOB
from jobflow.core.store import JobStore

from crystal.schemas.validation_dataset import ValidationDatasetArtifactReference
from crystal.structure.validation_dataset.artifact_storage import (
    get_artifact_document,
    get_latest_execution_output_for_job_uuid,
)


def _trajectory_reference_from_output(
    md_job_output: dict[str, Any],
) -> ValidationDatasetArtifactReference | None:
    """Return the blob pointer for a valid completed MD run."""
    if md_job_output["errors"]:
        return None
    md_properties = md_job_output["properties"]
    trajectory_blob_pointer = md_properties["trajectory_bytes"]
    return ValidationDatasetArtifactReference(
        artifact_id=trajectory_blob_pointer["blob_uuid"],
        store_name=trajectory_blob_pointer["store"],
    )


def build_md_trajectory_manifest(
    md_job_uuids: Iterable[str], jobstore: JobStore
) -> list[ValidationDatasetArtifactReference]:
    """Return valid trajectory pointers for the frame-selection stage."""
    trajectory_artifact_references: list[ValidationDatasetArtifactReference] = []
    for job_uuid in md_job_uuids:
        output = get_latest_execution_output_for_job_uuid(jobstore, job_uuid)
        if output is None:
            continue
        trajectory_artifact = _trajectory_reference_from_output(output)
        if trajectory_artifact is not None:
            trajectory_store = jobstore.additional_stores[  # type: ignore[has-type]
                trajectory_artifact.store_name
            ]
            if (
                get_artifact_document(
                    store=trajectory_store, artifact=trajectory_artifact
                )
                is None
            ):
                continue
            trajectory_artifact_references.append(trajectory_artifact)
    return trajectory_artifact_references


@job(config=JobConfig(expose_store=True))
def collect_md_trajectory_manifest(
    md_job_uuids: list[str],
) -> list[ValidationDatasetArtifactReference]:
    """Find completed MD trajectories for frame selection.

    This is a workflow step because trajectory data stays in the data store instead
    of being copied between jobs.
    """
    return build_md_trajectory_manifest(md_job_uuids, CURRENT_JOB.store)
