"""Turn selected MD frames into structures and choose the final validation set."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from ase.io.trajectory import Trajectory
from jobflow.core.flow import Flow
from jobflow.core.job import Job, JobConfig, job
from jobflow.core.state import CURRENT_JOB
from jobflow.core.store import JobStore
from pydantic import PositiveInt
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.schemas.properties import StructureGenerationProperties
from crystal.schemas.structure_generators import (
    DirectSamplingMakerSettings,
    DirectSamplingTaskDocument,
)
from crystal.schemas.validation_dataset import (
    ValidationDatasetArtifactReference,
    ValidationDatasetDirectSelectionSettings,
)
from crystal.structure.generators.sampling.direct_maker import DirectSamplingMaker
from crystal.structure.validation_dataset.artifact_storage import get_artifact_document


def _select_evenly_spaced_frame_indices(
    frame_count: int, maximum_frames: PositiveInt
) -> list[int]:
    """Select deterministic endpoints and intermediate trajectory frames."""
    if frame_count <= maximum_frames:
        return list(range(frame_count))
    if maximum_frames == 1:
        return [0]
    return [
        (index * (frame_count - 1)) // (maximum_frames - 1)
        for index in range(maximum_frames)
    ]


def _get_trajectory_bytes(
    trajectory_artifact: ValidationDatasetArtifactReference, jobstore: JobStore
) -> bytes:
    """Load one trajectory blob from its configured additional store."""
    trajectory_store = jobstore.additional_stores[  # type: ignore[has-type]
        trajectory_artifact.store_name
    ]
    document = get_artifact_document(
        store=trajectory_store, artifact=trajectory_artifact
    )
    data = None if document is None else document.get("data")
    if not isinstance(data, bytes):
        message = f"Trajectory blob is not available: {trajectory_artifact.artifact_id}"
        raise KeyError(message)
    return data


def _append_selected_trajectory_structures(
    source_path: Path, structures: list[PMG_Structure], maximum_frames: PositiveInt
) -> int:
    with Trajectory(  # type: ignore[no-untyped-call]
        source_path, mode="r"
    ) as source_trajectory:
        selected_indices = _select_evenly_spaced_frame_indices(
            len(source_trajectory), maximum_frames
        )
        structures.extend(
            AseAtomsAdaptor.get_structure(source_trajectory[frame_index])
            for frame_index in selected_indices
        )
    return len(selected_indices)


def build_validation_dataset_selection_structures(
    trajectory_artifacts: list[ValidationDatasetArtifactReference],
    direct_selection_settings: ValidationDatasetDirectSelectionSettings,
    jobstore: JobStore,
) -> StructureGenerationProperties:
    """Sample bounded structures from the available MD trajectories."""
    structures: list[PMG_Structure] = []
    frame_count = 0
    maximum_pool_frames = direct_selection_settings.maximum_frames_in_selection_pool
    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        for trajectory_index, trajectory_artifact in enumerate(trajectory_artifacts):
            remaining_frames = maximum_pool_frames - frame_count
            if remaining_frames <= 0:
                break
            source_path = temporary_path / f"source-{trajectory_index}.traj"
            source_path.write_bytes(
                _get_trajectory_bytes(trajectory_artifact, jobstore)
            )
            try:
                frame_count += _append_selected_trajectory_structures(
                    source_path=source_path,
                    structures=structures,
                    maximum_frames=min(
                        direct_selection_settings.maximum_frames_per_md_run,
                        remaining_frames,
                    ),
                )
            finally:
                source_path.unlink(missing_ok=True)
    return StructureGenerationProperties(structures=structures)


@job(
    config=JobConfig(expose_store=True),
    **{STRUCTURE_BLOB_STORE_NAME: [["output", "structures"]]},
)  # type: ignore[call-overload]
def sample_validation_dataset_structures(
    trajectory_artifacts: list[ValidationDatasetArtifactReference],
    direct_selection_settings: ValidationDatasetDirectSelectionSettings,
) -> StructureGenerationProperties:
    """Sample trajectory frames and persist them in the structure store."""
    return build_validation_dataset_selection_structures(
        trajectory_artifacts=trajectory_artifacts,
        direct_selection_settings=direct_selection_settings,
        jobstore=CURRENT_JOB.store,
    )


def cap_direct_selection_properties(
    direct_sampling_task_document: DirectSamplingTaskDocument,
    maximum_selected_structures: int,
) -> StructureGenerationProperties:
    """Enforce the user-facing DIRECT target after sampler selection."""
    if direct_sampling_task_document.errors:
        message = f"DIRECT sampling failed: {direct_sampling_task_document.errors}"
        raise RuntimeError(message)
    direct_sampling_properties = direct_sampling_task_document.properties
    if not isinstance(direct_sampling_properties, StructureGenerationProperties):
        message = "DIRECT sampling returned no structure-generation properties."
        raise TypeError(message)
    return StructureGenerationProperties(
        structures=direct_sampling_properties.structures[:maximum_selected_structures]
    )


@job(**{STRUCTURE_BLOB_STORE_NAME: [["output", "structures"]]})  # type: ignore[call-overload]
def cap_direct_selection(
    direct_sampling_task_document: DirectSamplingTaskDocument,
    maximum_selected_structures: int,
) -> StructureGenerationProperties:
    """Keep only the requested structures from the DIRECT result.

    This is a workflow step because the selected structures must be saved for the
    workflow's final output.
    """
    return cap_direct_selection_properties(
        direct_sampling_task_document=direct_sampling_task_document,
        maximum_selected_structures=maximum_selected_structures,
    )


def build_validation_dataset_direct_selection_flow(
    *,
    trajectory_artifacts: list[ValidationDatasetArtifactReference],
    direct_sampling_maker_settings: DirectSamplingMakerSettings,
    direct_selection_settings: ValidationDatasetDirectSelectionSettings,
) -> Flow:
    """Build the bounded trajectory sampling and DIRECT selection flow."""
    sampling_job: Job = sample_validation_dataset_structures(
        trajectory_artifacts=trajectory_artifacts,
        direct_selection_settings=direct_selection_settings,
    )
    direct_job = DirectSamplingMaker(
        simulation_settings=direct_sampling_maker_settings.simulation_settings,
        executor_settings=direct_sampling_maker_settings.executor_settings,
    ).make(structures=sampling_job.output.structures)  # type: ignore[arg-type]
    cap_job = cap_direct_selection(
        direct_sampling_task_document=direct_job.output,
        maximum_selected_structures=direct_selection_settings.target_number_of_structures,
    )
    return Flow(jobs=[sampling_job, direct_job, cap_job], output=cap_job.output)
