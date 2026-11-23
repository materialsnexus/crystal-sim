"""Maker for static GPAW isolated-atom reference calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jobflow.core.flow import Flow
from jobflow.core.job import job

from crystal.calculators.gpaw.isolated_atom_registry_jobflow import (
    register_gpaw_isolated_atom_task_document,
)
from crystal.calculators.gpaw.isolated_atom_runner import run_gpaw_isolated_atom
from crystal.schemas.base import BaseMaker
from crystal.schemas.elements import ElementStr
from crystal.schemas.gpaw import GpawIsolatedAtomSimulationSettings
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawIsolatedAtomReferenceCollection,
    GpawIsolatedAtomTaskDocument,
)
from crystal.schemas.maker_names import MakerName
from crystal.schemas.scratch_execution import LocalScratchPolicy

GPAW_ISOLATED_ATOM_PRIMARY_FAILURE_DIAGNOSTIC = "Primary task failed"
GPAW_ISOLATED_ATOM_FALLBACK_FAILURE_DIAGNOSTIC = "PPCG fallback failed"


@job(output_schema=GpawIsolatedAtomReferenceCollection)
def collect_gpaw_isolated_atom_task_documents(
    task_documents: list[GpawIsolatedAtomTaskDocument],
    simulation_settings: GpawIsolatedAtomSimulationSettings,
) -> GpawIsolatedAtomReferenceCollection:
    """Collect static isolated-atom task results into one reference result."""
    return GpawIsolatedAtomReferenceCollection.from_task_documents(
        task_documents=tuple(task_documents), simulation_settings=simulation_settings
    )


@job(output_schema=GpawIsolatedAtomTaskDocument)
def recover_gpaw_isolated_atom_task_document(
    primary_task_document: GpawIsolatedAtomTaskDocument,
    simulation_settings: GpawIsolatedAtomSimulationSettings,
    number_of_processes: int,
    local_scratch_policy: LocalScratchPolicy,
) -> GpawIsolatedAtomTaskDocument:
    """Recover one failed primary task with a separate PPCG Jobflow job."""
    if primary_task_document.is_success:
        return primary_task_document

    run_directory = Path.cwd().resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    try:
        run_result = run_gpaw_isolated_atom(
            element=primary_task_document.element,
            simulation_settings=simulation_settings,
            run_directory=run_directory,
            number_of_processes=number_of_processes,
            local_scratch_policy=local_scratch_policy,
            use_ppcg_fallback=True,
        )
    except RuntimeError as fallback_error:
        error = (
            f"{GPAW_ISOLATED_ATOM_PRIMARY_FAILURE_DIAGNOSTIC}: "
            f"{primary_task_document.errors}; "
            f"{GPAW_ISOLATED_ATOM_FALLBACK_FAILURE_DIAGNOSTIC}: {fallback_error}"
        )
        return GpawIsolatedAtomTaskDocument.from_directory_and_error(
            dir_path=run_directory, error=error, element=primary_task_document.element
        )
    return GpawIsolatedAtomTaskDocument.from_directory_and_run_result(
        dir_path=run_directory, run_result=run_result
    )


@dataclass(kw_only=True)
class GpawIsolatedAtomMaker(BaseMaker):
    """Create one static GPAW atom calculation job for each required element."""

    name: MakerName = MakerName.gpaw_isolated_atom
    simulation_settings: GpawIsolatedAtomSimulationSettings

    @job(output_schema=GpawIsolatedAtomTaskDocument)
    def _run(self, element: ElementStr) -> GpawIsolatedAtomTaskDocument:
        run_directory = Path.cwd().resolve()
        run_directory.mkdir(parents=True, exist_ok=True)
        try:
            run_result = run_gpaw_isolated_atom(
                element=element,
                simulation_settings=self.simulation_settings,
                run_directory=run_directory,
                number_of_processes=self.executor_settings.number_of_processes,
                local_scratch_policy=self.executor_settings.local_scratch_policy,
            )
        except RuntimeError as runtime_error:
            return GpawIsolatedAtomTaskDocument.from_directory_and_error(
                dir_path=run_directory, error=str(runtime_error), element=element
            )
        return GpawIsolatedAtomTaskDocument.from_directory_and_run_result(
            dir_path=run_directory, run_result=run_result
        )

    def make(self, elements: frozenset[ElementStr]) -> Flow:
        """Build static atom jobs, recovery jobs, registry jobs, and collector."""
        if not elements:
            msg = "GPAW isolated-atom references require at least one element."
            raise ValueError(msg)
        atom_jobs = [self._run(element=element) for element in sorted(elements)]
        recovery_jobs = [
            recover_gpaw_isolated_atom_task_document(
                primary_task_document=atom_job.output,
                simulation_settings=self.simulation_settings,
                number_of_processes=self.executor_settings.number_of_processes,
                local_scratch_policy=self.executor_settings.local_scratch_policy,
            )
            for atom_job in atom_jobs
        ]
        registry_jobs = [
            register_gpaw_isolated_atom_task_document(
                task_document=recovery_job.output,
                simulation_settings=self.simulation_settings,
                source_job_uuid=recovery_job.uuid,
            )
            for recovery_job in recovery_jobs
        ]
        collector_job = collect_gpaw_isolated_atom_task_documents(
            task_documents=[registry_job.output for registry_job in registry_jobs],
            simulation_settings=self.simulation_settings,
        )
        workflow_metadata = self.slurm_settings.to_queueadapter_spec()
        for workflow_job in [*atom_jobs, *recovery_jobs, *registry_jobs, collector_job]:
            workflow_job.update_metadata(workflow_metadata, dynamic=False)
        return Flow(
            jobs=[*atom_jobs, *recovery_jobs, *registry_jobs, collector_job],
            output=collector_job.output,
        )
