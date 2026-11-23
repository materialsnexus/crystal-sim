from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from crystal.helpers.fireworks.runtime_config import FireWorksRuntimePaths
from crystal.schemas.fireworks_runtime import FireWorksRuntimeProfile
from tests.helpers import slurm_runtime


def test_successful_slurm_cleanup_deletes_mongo_before_directory(
    tmp_path: Path,
) -> None:
    shared_execution_directory = tmp_path / "shared-execution"
    cleanup_events: list[str] = []
    temporary_runtime = MagicMock()
    temporary_runtime.delete_mongo_database.side_effect = lambda: cleanup_events.append(
        "delete-mongo"
    )
    temporary_runtime_execution = object()

    def record_directory_removal(path: Path) -> None:
        assert path == shared_execution_directory
        cleanup_events.append("remove-directory")

    with (
        patch.object(
            slurm_runtime,
            "_shared_execution_directory",
            return_value=shared_execution_directory,
        ),
        patch.object(
            slurm_runtime,
            "build_temporary_slurm_runtime",
            return_value=temporary_runtime,
        ),
        patch.object(
            slurm_runtime,
            "_execute_workflow_through_runtime",
            return_value=temporary_runtime_execution,
        ),
        patch.object(slurm_runtime, "rmtree", side_effect=record_directory_removal),
        slurm_runtime.execute_slurm_workflow(job_or_flow=MagicMock()) as execution,
    ):
        assert execution is temporary_runtime_execution

    assert cleanup_events == ["delete-mongo", "remove-directory"]


def test_temporary_slurm_runtime_uses_slurm_profile(tmp_path: Path) -> None:
    with patch.object(
        FireWorksRuntimePaths, "from_profile", wraps=FireWorksRuntimePaths.from_profile
    ) as from_profile:
        runtime = slurm_runtime.build_temporary_slurm_runtime(
            temporary_runtime_parent_directory=tmp_path
        )

    from_profile.assert_called_once_with(profile=FireWorksRuntimeProfile.SLURM)
    assert runtime.runtime_paths.is_queue_launch


def test_temporary_slurm_runtime_isolates_all_store_databases(tmp_path: Path) -> None:
    runtime = slurm_runtime.build_temporary_slurm_runtime(
        temporary_runtime_parent_directory=tmp_path
    )
    paths = runtime.runtime_paths.profile_paths
    launchpad = yaml.safe_load(paths.launchpad_path.read_text())
    jobstore = yaml.safe_load(paths.jobstore_path.read_text())

    assert launchpad["name"] == runtime.mongo_database_name
    assert jobstore["docs_store"]["database"] == runtime.mongo_database_name
    assert all(
        store["database"] == runtime.mongo_database_name
        for store in jobstore["additional_stores"].values()
    )
