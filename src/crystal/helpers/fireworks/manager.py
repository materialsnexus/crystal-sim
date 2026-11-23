# noqa: D100
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fireworks import Firework, LaunchPad
from fireworks.core.fworker import FWorker
from fireworks.features.dupefinder import DupeFinderBase
from jobflow.core.flow import Flow
from jobflow.core.job import Job
from jobflow.core.store import JobStore
from jobflow.managers.fireworks import flow_to_workflow
from pydantic import BaseModel, ConfigDict

from crystal.constants.fireworks import QUEUEADAPTER_SPEC_KEY
from crystal.helpers.fireworks.connections import connect_to_jobstore, run_launchpad
from crystal.helpers.fireworks.dupe_finder import DEDUPE_KEY, DupeFinderBySpecKey
from crystal.helpers.fireworks.slurm_settings import (
    QueueAdapterSettingsInput,
    QueueAdapterSettingsMapping,
    build_slurm_settings_from_executor_settings,
)
from crystal.helpers.job import compute_dedupe

logger = logging.getLogger(__name__)


class JobManager(BaseModel):
    """Wrapper to add jobs and flows to launchpad.

    During adding jobs/flows to launchpad, the manager also creates a dedupe_key to each firework specification.
    This `dedupe_key` is used to determine the duplicated runs.

    NOTE: If a graph of jobs only contains list of connected jobs without any jobs returning Response, the duplication setup works fine.

    NOTE: This applies to only the submission time jobs, i.e. Any jobs which are spawned with Response(replace, addition, detour)
    will not be assigned a dedupe key. These jobs will be run each time and will be duplicated in the database.

    Deduplication Key Semantics:

    A dedupe key is computed using

    1. Function identity
    2. Function arguments and keyword arguments
    3. For bound methods (e.g. Maker methods):
       - If the bound object defines a ``dedupe_property``, that value is incorporated into the hash.
       - Otherwise, the full bound method signature (including implicit self) is used.

    For downstream (child) jobs whose arguments include `OutputReference` objects,
    the dedupe computation replaces upstream job UUIDs with their corresponding
    dedupe keys to ensure all jobs in identical flows dedupe correctly.


    Supported Use Cases:

    - Identical standalone jobs
    - Identical flows submitted multiple times
    - Flows containing dependent jobs via OutputReference

    Limitations:

    1. Jobs reconstructed outside of a flow will not dedupe against identical jobs in a flow because flow-level
       dedupe keys incorporate upstream lineage.

    2. Jobs dynamically generated at runtime via Response(replace=...), Response(addition=...), Response(detour=...) are not supported due to submission-time deduplication key generation.

    NOTE: Since dedupe keys are computed before conversion to FireWorks, runtime-spawned jobs will always execute unless additional runtime dedupe handling is implemented.

    NOTE: Ensure a qlaunch or rlaunch is running and setup for the correct fireworker and launchpad and qadapter
    to assign correct jobs.
    If not, move to the directory where you want to store the output directories and launch
    `path/to/qlaunch -r -w /path/to/fworker.yaml -l /path/to/launchpad/yaml -q /path/to/default/qadapter.yaml rapidfire --nlaunches infinite > output.log 2>&1 & echo $! > nohup.pid

    Use
    - `add_to_launchpad`: to add jobs/flows to launchpad
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    launchpad: LaunchPad
    jobstore: JobStore
    fireworker: FWorker

    @classmethod
    def from_configs(  # noqa: D102
        cls, launchpad_config: Path, jobstore_config: Path, fireworker: FWorker
    ) -> JobManager:
        launchpad = run_launchpad(config_file=launchpad_config)
        jobstore = connect_to_jobstore(config_file=jobstore_config)
        return cls(launchpad=launchpad, jobstore=jobstore, fireworker=fireworker)

    def _apply_nondefault_queueadapter_settings_to_firework_spec(
        self,
        firework: Firework,
        queueadapter_settings: QueueAdapterSettingsInput | None,
    ) -> None:
        if queueadapter_settings is None:
            return
        slurm_settings = build_slurm_settings_from_executor_settings(
            queueadapter_settings
        )
        queueadapter_overrides_dict = slurm_settings.model_dump(exclude_none=True)
        firework.spec.setdefault(QUEUEADAPTER_SPEC_KEY, {}).update(
            queueadapter_overrides_dict
        )

    def _apply_duplicate_finder_setting_to_firework_spec(
        self, firework: Firework, dupe_finder: DupeFinderBase
    ) -> None:
        firework.spec.setdefault("_dupefinder", dupe_finder)

    def _get_correct_queueadapter_settings(
        self,
        settings_or_settings_map: QueueAdapterSettingsInput
        | QueueAdapterSettingsMapping,
        fw_name: str,
    ) -> QueueAdapterSettingsInput | None:
        if isinstance(settings_or_settings_map, QueueAdapterSettingsInput):
            return settings_or_settings_map
        return settings_or_settings_map.get(fw_name, None)

    @property
    def default_dupe_finder(self) -> DupeFinderBase:  # noqa: D102
        return DupeFinderBySpecKey()

    def add_to_launchpad(  # noqa: D102
        self,
        job_or_flow: Job | Flow,
        queueadapter_settings: QueueAdapterSettingsInput
        | QueueAdapterSettingsMapping
        | None = None,
        dupe_finder: DupeFinderBase | None = None,
    ) -> list[int]:
        dupe_finder = dupe_finder or self.default_dupe_finder
        flow = Flow(jobs=[job_or_flow]) if isinstance(job_or_flow, Job) else job_or_flow
        assign_dedupe_values(flow)
        workflow = flow_to_workflow(flow, store=self.jobstore)
        for firework in workflow.fws:
            self._apply_duplicate_finder_setting_to_firework_spec(
                firework=firework, dupe_finder=dupe_finder
            )
            if queueadapter_settings is not None:
                firework_name = firework.name
                correct_queueadapter_settings = self._get_correct_queueadapter_settings(
                    queueadapter_settings, firework_name
                )
                self._apply_nondefault_queueadapter_settings_to_firework_spec(
                    firework=firework,
                    queueadapter_settings=correct_queueadapter_settings,
                )
        old_firework_id_to_new_id_mapping = self.launchpad.add_wf(workflow)
        return list(old_firework_id_to_new_id_mapping.values())


def attach_dedupe_value_to_job(job: Job, dedupe_value: str) -> None:  # noqa: D103
    job.config.manager_config[DEDUPE_KEY] = dedupe_value
    # Ensures that the manager config is not passed onto Response job for now
    job.config.pass_manager_config = False


def assign_dedupe_values(flow: Flow) -> dict[str, Any]:  # noqa: D103
    job_uuid_to_dedupe_mapping: dict[str, Any] = {}
    for job, _ in flow.iterflow():  # type:ignore[no-untyped-call]
        dedupe_value = compute_dedupe(job, job_uuid_to_dedupe_mapping)
        attach_dedupe_value_to_job(job, dedupe_value)
        job_uuid_to_dedupe_mapping[job.uuid] = dedupe_value
    return job_uuid_to_dedupe_mapping
