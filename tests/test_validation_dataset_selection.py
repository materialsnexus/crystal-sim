"""Tests for scalable validation-dataset artifact collection and selection."""

from __future__ import annotations

from pathlib import Path

from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import MemoryStore
from monty.json import MontyDecoder

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.constants.validation_dataset import (
    VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME,
)
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.properties import StructureGenerationProperties
from crystal.schemas.structure_generators import DirectSamplingTaskDocument
from crystal.schemas.validation_dataset import (
    ValidationDatasetArtifactReference,
    ValidationDatasetDirectSelectionSettings,
)
from crystal.structure.validation_dataset.artifact_manifest import (
    collect_md_trajectory_manifest,
)
from crystal.structure.validation_dataset.direct_selection import (
    cap_direct_selection_properties,
    sample_validation_dataset_structures,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.validation_dataset import (
    build_validation_dataset_md_trajectory_bytes_for_test,
)


def _job_store_with_trajectories(trajectories: dict[str, bytes]) -> JobStore:
    documents_store = MemoryStore()
    trajectory_store = MemoryStore()
    structures_store = MemoryStore()
    documents_store.connect()
    trajectory_store.connect()
    structures_store.connect()
    trajectory_store.update(
        [
            {"task_id": artifact_id, "blob_uuid": artifact_id, "data": trajectory_bytes}
            for artifact_id, trajectory_bytes in trajectories.items()
        ]
    )
    jobstore = JobStore(
        docs_store=documents_store,
        additional_stores={
            STRUCTURE_BLOB_STORE_NAME: structures_store,
            VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME: trajectory_store,
        },
    )
    jobstore.connect()
    return jobstore


@run_in_temp_dir
def test_manifest_collects_available_compact_artifact_pointers() -> None:
    trajectory_bytes = build_validation_dataset_md_trajectory_bytes_for_test(
        number_of_frames=3
    )
    jobstore = _job_store_with_trajectories({"trajectory-1": trajectory_bytes})
    jobstore.docs_store.update(
        [
            {
                "uuid": "md-job-1",
                "index": 1,
                "output": {
                    "errors": "",
                    "properties": {
                        "run_id": "slab:sio2:900K",
                        "frame_count": 3,
                        "number_of_atoms": len(TEST_SI_STRUCTURE),
                        "trajectory_bytes": {
                            "store": VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME,
                            "blob_uuid": "trajectory-1",
                        },
                    },
                },
            },
            {
                "uuid": "md-job-failed",
                "index": 1,
                "output": {"errors": "MLIP-MD failed", "properties": None},
            },
            {
                "uuid": "md-job-missing-artifact",
                "index": 1,
                "output": {
                    "errors": "",
                    "properties": {
                        "run_id": "slab:sio2:1500K",
                        "frame_count": 3,
                        "number_of_atoms": len(TEST_SI_STRUCTURE),
                        "trajectory_bytes": {
                            "store": VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME,
                            "blob_uuid": "trajectory-missing",
                        },
                    },
                },
            },
        ]
    )

    manifest_job = collect_md_trajectory_manifest(
        md_job_uuids=[
            "md-job-1",
            "missing-md-job",
            "md-job-failed",
            "md-job-missing-artifact",
        ]
    )
    result = run_locally(manifest_job, root_dir=Path.cwd(), store=jobstore)
    trajectory_artifacts = result[manifest_job.uuid][1].output

    assert trajectory_artifacts == [
        ValidationDatasetArtifactReference(
            artifact_id="trajectory-1",
            store_name=VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME,
        )
    ]


@run_in_temp_dir
def test_structure_sampling_applies_per_run_and_global_frame_bounds() -> None:
    trajectory_bytes = {
        "trajectory-1": build_validation_dataset_md_trajectory_bytes_for_test(
            number_of_frames=5
        ),
        "trajectory-2": build_validation_dataset_md_trajectory_bytes_for_test(
            number_of_frames=4
        ),
    }
    jobstore = _job_store_with_trajectories(trajectory_bytes)
    sampling_job = sample_validation_dataset_structures(
        trajectory_artifacts=[
            ValidationDatasetArtifactReference(
                artifact_id="trajectory-1",
                store_name=VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME,
            ),
            ValidationDatasetArtifactReference(
                artifact_id="trajectory-2",
                store_name=VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME,
            ),
        ],
        direct_selection_settings=ValidationDatasetDirectSelectionSettings(
            maximum_frames_per_md_run=2, maximum_frames_in_selection_pool=3
        ),
    )
    run_locally(sampling_job, root_dir=Path.cwd(), store=jobstore)

    stored_job_document = next(jobstore.docs_store.query({"uuid": sampling_job.uuid}))
    stored_structures = stored_job_document["output"]["structures"]
    assert stored_structures["store"] == STRUCTURE_BLOB_STORE_NAME

    loaded_output = MontyDecoder().process_decoded(
        jobstore.get_output(sampling_job.uuid, load=True)
    )
    assert len(loaded_output.structures) == 3


def test_direct_selection_cap_limits_successful_output() -> None:
    document = DirectSamplingTaskDocument(
        properties=StructureGenerationProperties(
            structures=[TEST_SI_STRUCTURE, TEST_SI_STRUCTURE, TEST_SI_STRUCTURE]
        )
    )

    result = cap_direct_selection_properties(
        direct_sampling_task_document=document, maximum_selected_structures=2
    )

    assert len(result.structures) == 2
