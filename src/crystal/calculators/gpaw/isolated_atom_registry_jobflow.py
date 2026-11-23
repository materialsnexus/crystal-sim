"""Jobflow access to the GPAW isolated-atom registry."""

from __future__ import annotations

from jobflow.core.job import JobConfig, job
from jobflow.core.state import CURRENT_JOB
from jobflow.core.store import JobStore
from maggma.stores import MongoStore

from crystal.constants.gpaw import GPAW_ISOLATED_ATOM_REGISTRY_STORE_NAME
from crystal.repositories.gpaw_isolated_atom_registry import (
    GpawIsolatedAtomRegistryRepository,
)
from crystal.schemas.gpaw import GpawIsolatedAtomSimulationSettings
from crystal.schemas.gpaw_isolated_atom_reference import GpawIsolatedAtomTaskDocument
from crystal.schemas.gpaw_isolated_atom_registry import (
    GpawIsolatedAtomRegistryEntry,
    GpawIsolatedAtomRegistryIdentity,
)


@job(config=JobConfig(expose_store=True), output_schema=GpawIsolatedAtomTaskDocument)
def register_gpaw_isolated_atom_task_document(
    task_document: GpawIsolatedAtomTaskDocument,
    simulation_settings: GpawIsolatedAtomSimulationSettings,
    source_job_uuid: str,
) -> GpawIsolatedAtomTaskDocument:
    """Persist one successful isolated-atom task document for reuse."""
    if not task_document.is_success:
        return task_document

    isolated_atom_energy = task_document.isolated_atom_energy
    eigensolver = task_document.eigensolver
    if eigensolver is None:
        msg = (
            "A successful GPAW isolated-atom task requires an eigensolver "
            "for registry registration."
        )
        raise ValueError(msg)
    registry_repository = get_gpaw_isolated_atom_registry_repository(CURRENT_JOB.store)
    registry_repository.upsert_success(
        GpawIsolatedAtomRegistryEntry(
            identity=GpawIsolatedAtomRegistryIdentity.from_simulation_settings_and_runtime_provenance(
                element=task_document.element,
                simulation_settings=simulation_settings,
                runtime_provenance=isolated_atom_energy.runtime_provenance,
                eigensolver=eigensolver,
            ),
            simulation_settings=simulation_settings,
            energy=isolated_atom_energy.energy,
            magnetic_moment=task_document.magnetic_moment,
            eigensolver=eigensolver,
            runtime_provenance=isolated_atom_energy.runtime_provenance,
            source_job_uuid=source_job_uuid,
        )
    )
    return task_document


def get_gpaw_isolated_atom_registry_repository(
    jobflow_store: object,
) -> GpawIsolatedAtomRegistryRepository:
    """Build the configured GPAW registry repository from a Jobflow store."""
    if not isinstance(jobflow_store, JobStore):
        msg = "GPAW isolated-atom registry access requires the Jobflow store."
        raise TypeError(msg)
    additional_stores = getattr(jobflow_store, "additional_stores", {})
    registry_store = additional_stores.get(GPAW_ISOLATED_ATOM_REGISTRY_STORE_NAME)
    if not isinstance(registry_store, MongoStore):
        msg = "GPAW isolated-atom registry access requires its configured Mongo store."
        raise TypeError(msg)
    return GpawIsolatedAtomRegistryRepository(registry_store)
