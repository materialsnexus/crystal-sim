"""Reusable isolated SLURM runtime orchestration for integration tests."""

from __future__ import annotations

import shlex
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Any
from uuid import uuid4

import yaml
from fireworks import Workflow
from fireworks.core.fworker import FWorker
from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from fireworks.utilities.fw_serializers import load_object_from_file
from jobflow.core.flow import Flow
from jobflow.core.job import Job

from crystal import CRYSTAL_ROOT
from crystal.constants.fireworks_runtime import FireWorksRuntimeCommandFlag
from crystal.firework_helpers import workflow_reached_terminal_state
from crystal.helpers.fireworks.connections import run_launchpad
from crystal.helpers.fireworks.manager import JobManager
from crystal.helpers.fireworks.runtime_config import FireWorksRuntimePaths
from crystal.helpers.fireworks.runtime_execution import FireWorksRuntimeExecutionManager
from crystal.schemas.fireworks_runtime import (
    FireWorksRuntimeProfile,
    FireWorksRuntimeProfilePaths,
)

SLURM_INTEGRATION_WORKFLOW_TIMEOUT_SECONDS = 300
SLURM_INTEGRATION_WORKFLOW_POLL_INTERVAL_SECONDS = 5
FIREWORKS_ID_ASSIGNER_DOCUMENT_ID = "fw_id_assigner"
TEMPORARY_SLURM_DATABASE_PREFIX = "crystal_slurm_test_"
SLURM_INTEGRATION_RUN_DIRECTORY_NAME = "slurm-integration"
TEMPORARY_SLURM_RUNTIME_DIRECTORY_NAME = "runtime"
TEMPORARY_SLURM_LAUNCHPAD_FILENAME = "launchpad.yaml"
TEMPORARY_SLURM_JOBSTORE_FILENAME = "jobstore.yaml"
TEMPORARY_SLURM_FWORKER_FILENAME = "fworker.yaml"
TEMPORARY_SLURM_QADAPTER_FILENAME = "qadapter.yaml"
TEMPORARY_SLURM_LOG_DIRECTORY_NAME = "queue-logs"


class SlurmRuntimeConfigurationError(TypeError):
    """Raised when a SLURM runtime configuration cannot be isolated."""


class SlurmWorkflowTimeoutError(TimeoutError):
    """Raised when a SLURM workflow does not reach a terminal state."""


@dataclass(frozen=True, slots=True)
class SlurmWorkflowExecution:
    """A terminal workflow execution submitted through the SLURM profile."""

    manager: JobManager
    anchor_firework_id: int

    @property
    def workflow(self) -> Workflow:
        """Return the complete FireWorks workflow containing the submitted anchor."""
        return self.manager.launchpad.get_wf_by_fw_id(self.anchor_firework_id)


@dataclass(frozen=True, slots=True)
class TemporarySlurmRuntime:
    """SLURM runtime paths isolated by a unique temporary Mongo database."""

    runtime_paths: FireWorksRuntimePaths
    mongo_database_name: str

    def delete_mongo_database(self) -> None:
        """Delete only this runtime's isolated Mongo database."""
        temporary_launchpad = run_launchpad(
            config_file=self.runtime_paths.profile_paths.launchpad_path
        )
        temporary_launchpad.connection.drop_database(self.mongo_database_name)


@contextmanager
def execute_slurm_workflow(
    job_or_flow: Job | Flow,
    *,
    timeout_seconds: float = SLURM_INTEGRATION_WORKFLOW_TIMEOUT_SECONDS,
) -> Generator[SlurmWorkflowExecution, None, None]:
    """Run a workflow through an isolated overlay of the SLURM profile."""
    shared_execution_directory = _shared_execution_directory()
    temporary_runtime = build_temporary_slurm_runtime(
        temporary_runtime_parent_directory=shared_execution_directory
    )
    test_completed_successfully = False
    try:
        yield _execute_workflow_through_runtime(
            job_or_flow=job_or_flow,
            runtime_paths=temporary_runtime.runtime_paths,
            run_directory=shared_execution_directory,
            timeout_seconds=timeout_seconds,
        )
        test_completed_successfully = True
    finally:
        temporary_runtime.delete_mongo_database()
        if test_completed_successfully:
            rmtree(shared_execution_directory)


def _shared_execution_directory() -> Path:
    integration_run_identifier = uuid4().hex
    return (
        CRYSTAL_ROOT
        / "tmp"
        / SLURM_INTEGRATION_RUN_DIRECTORY_NAME
        / integration_run_identifier
    )


def build_temporary_slurm_runtime(
    *, temporary_runtime_parent_directory: Path
) -> TemporarySlurmRuntime:
    """Derive an isolated runtime from the SLURM configuration."""
    slurm_runtime_paths = FireWorksRuntimePaths.from_profile(
        profile=FireWorksRuntimeProfile.SLURM
    )
    temporary_runtime_config_directory = (
        temporary_runtime_parent_directory / TEMPORARY_SLURM_RUNTIME_DIRECTORY_NAME
    )
    temporary_runtime_config_directory.mkdir(parents=True, exist_ok=True)
    mongo_database_name = f"{TEMPORARY_SLURM_DATABASE_PREFIX}{uuid4().hex}"
    temporary_launchpad_path = (
        temporary_runtime_config_directory / TEMPORARY_SLURM_LAUNCHPAD_FILENAME
    )
    temporary_jobstore_path = (
        temporary_runtime_config_directory / TEMPORARY_SLURM_JOBSTORE_FILENAME
    )
    temporary_fworker_path = (
        temporary_runtime_config_directory / TEMPORARY_SLURM_FWORKER_FILENAME
    )
    temporary_qadapter_path = (
        temporary_runtime_config_directory / TEMPORARY_SLURM_QADAPTER_FILENAME
    )
    _write_temporary_launchpad_configuration(
        source_path=slurm_runtime_paths.profile_paths.launchpad_path,
        output_path=temporary_launchpad_path,
        mongo_database_name=mongo_database_name,
    )
    _write_temporary_jobstore_configuration(
        source_path=slurm_runtime_paths.profile_paths.jobstore_path,
        output_path=temporary_jobstore_path,
        mongo_database_name=mongo_database_name,
    )
    _write_temporary_fworker_configuration(
        source_path=slurm_runtime_paths.profile_paths.fworker_path,
        output_path=temporary_fworker_path,
    )
    _write_temporary_qadapter_configuration(
        source_path=slurm_runtime_paths.profile_paths.qadapter_template_path,
        output_path=temporary_qadapter_path,
        fworker_path=temporary_fworker_path,
        launchpad_path=temporary_launchpad_path,
        log_directory=(
            temporary_runtime_config_directory / TEMPORARY_SLURM_LOG_DIRECTORY_NAME
        ),
    )
    temporary_runtime_paths = FireWorksRuntimePaths.from_paths(
        profile=FireWorksRuntimeProfile.SLURM,
        profile_paths=FireWorksRuntimeProfilePaths(
            launchpad_path=temporary_launchpad_path,
            jobstore_path=temporary_jobstore_path,
            fworker_path=temporary_fworker_path,
            qadapter_template_path=temporary_qadapter_path,
        ),
        fw_config_path=slurm_runtime_paths.fw_config_path,
        mpi_environment_setup_script_path=(
            slurm_runtime_paths.mpi_environment_setup_script_path
        ),
    )
    return TemporarySlurmRuntime(
        runtime_paths=temporary_runtime_paths, mongo_database_name=mongo_database_name
    )


def _write_temporary_launchpad_configuration(
    *, source_path: Path, output_path: Path, mongo_database_name: str
) -> None:
    launchpad_configuration = _load_yaml_mapping(source_path)
    launchpad_configuration["name"] = mongo_database_name
    _write_yaml_mapping(output_path, launchpad_configuration)


def _write_temporary_jobstore_configuration(
    *, source_path: Path, output_path: Path, mongo_database_name: str
) -> None:
    jobstore_configuration = _load_yaml_mapping(source_path)
    docs_store_configuration = jobstore_configuration["docs_store"]
    docs_store_configuration["database"] = mongo_database_name
    additional_store_configurations = jobstore_configuration.get(
        "additional_stores", {}
    )
    for additional_store_configuration in additional_store_configurations.values():
        additional_store_configuration["database"] = mongo_database_name
    _write_yaml_mapping(output_path, jobstore_configuration)


def _write_temporary_fworker_configuration(
    *, source_path: Path, output_path: Path
) -> None:
    output_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def _write_temporary_qadapter_configuration(
    *,
    source_path: Path | None,
    output_path: Path,
    fworker_path: Path,
    launchpad_path: Path,
    log_directory: Path,
) -> None:
    if source_path is None:
        raise SlurmRuntimeConfigurationError
    queue_adapter: CommonAdapter = load_object_from_file(source_path.as_posix())
    queue_adapter["rocket_launch"] = (
        _build_rlaunch_command_with_temporary_configuration_paths(
            rocket_launch_command=str(queue_adapter["rocket_launch"]),
            fworker_path=fworker_path,
            launchpad_path=launchpad_path,
        )
    )
    queue_adapter["logdir"] = log_directory.as_posix()
    _write_yaml_mapping(output_path, queue_adapter.to_dict())


def _build_rlaunch_command_with_temporary_configuration_paths(
    *, rocket_launch_command: str, fworker_path: Path, launchpad_path: Path
) -> str:
    rocket_launch_arguments = shlex.split(rocket_launch_command)
    _replace_rlaunch_configuration_argument(
        rocket_launch_arguments=rocket_launch_arguments,
        configuration_argument=FireWorksRuntimeCommandFlag.FWORKER,
        configuration_path=fworker_path,
    )
    _replace_rlaunch_configuration_argument(
        rocket_launch_arguments=rocket_launch_arguments,
        configuration_argument=FireWorksRuntimeCommandFlag.LAUNCHPAD,
        configuration_path=launchpad_path,
    )
    return shlex.join(rocket_launch_arguments)


def _replace_rlaunch_configuration_argument(
    *,
    rocket_launch_arguments: list[str],
    configuration_argument: str,
    configuration_path: Path,
) -> None:
    configuration_argument_index = rocket_launch_arguments.index(configuration_argument)
    configuration_path_index = configuration_argument_index + 1
    rocket_launch_arguments[configuration_path_index] = configuration_path.as_posix()


def _load_yaml_mapping(configuration_path: Path) -> dict[str, Any]:
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise SlurmRuntimeConfigurationError
    return configuration


def _write_yaml_mapping(output_path: Path, configuration: dict[str, Any]) -> None:
    output_path.write_text(
        yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8"
    )


def _execute_workflow_through_runtime(
    *,
    job_or_flow: Job | Flow,
    runtime_paths: FireWorksRuntimePaths,
    run_directory: Path,
    timeout_seconds: float,
) -> SlurmWorkflowExecution:
    manager = JobManager.from_configs(
        launchpad_config=runtime_paths.profile_paths.launchpad_path,
        jobstore_config=runtime_paths.profile_paths.jobstore_path,
        fireworker=FWorker.from_file(
            runtime_paths.profile_paths.fworker_path.as_posix()
        ),
    )
    _ensure_fireworks_id_assigner(manager)
    firework_ids = manager.add_to_launchpad(job_or_flow, dupe_finder=DupeFinderExact())
    anchor_firework_id = firework_ids[0]
    execution_manager = FireWorksRuntimeExecutionManager(
        runtime_paths=runtime_paths, run_directory=run_directory
    )
    launch_handle = execution_manager.start_launch(
        mode="rapidfire", nlaunches="infinite"
    )
    try:
        _wait_for_workflow_terminal_state(
            manager=manager,
            firework_id=anchor_firework_id,
            timeout_seconds=timeout_seconds,
        )
    finally:
        launch_handle.stop()
    return SlurmWorkflowExecution(
        manager=manager, anchor_firework_id=anchor_firework_id
    )


def _ensure_fireworks_id_assigner(manager: JobManager) -> None:
    manager.launchpad.fw_id_assigner.update_one(
        {"_id": FIREWORKS_ID_ASSIGNER_DOCUMENT_ID},
        {"$setOnInsert": {"next_fw_id": 1, "next_launch_id": 1}},
        upsert=True,
    )
    for counter_name in ("next_fw_id", "next_launch_id"):
        manager.launchpad.fw_id_assigner.update_one(
            {
                "_id": FIREWORKS_ID_ASSIGNER_DOCUMENT_ID,
                counter_name: {"$exists": False},
            },
            {"$set": {counter_name: 1}},
        )


def _wait_for_workflow_terminal_state(
    *, manager: JobManager, firework_id: int, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if workflow_reached_terminal_state(firework_id, manager.launchpad):
            return
        time.sleep(SLURM_INTEGRATION_WORKFLOW_POLL_INTERVAL_SECONDS)
    raise SlurmWorkflowTimeoutError
