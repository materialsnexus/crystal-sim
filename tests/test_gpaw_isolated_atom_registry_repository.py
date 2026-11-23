"""MongoDB persistence tests for reusable GPAW isolated-atom results."""

from __future__ import annotations

from uuid import uuid4

import pytest
from maggma.stores import MongoStore

from crystal.repositories.gpaw_isolated_atom_registry import (
    GpawIsolatedAtomRegistryConflictError,
    GpawIsolatedAtomRegistryRepository,
)
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.gpaw import (
    GpawEigensolverName,
    GpawIsolatedAtomCalculatorSettings,
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
)
from crystal.schemas.gpaw_isolated_atom_registry import (
    GpawIsolatedAtomRegistryEntry,
    GpawIsolatedAtomRegistryIdentity,
)

TEST_GPAW_VERSION = "25.7.0"
TEST_ENERGY = -3.0
TEST_MAGNETIC_MOMENT = 2.0
TEST_ELEMENT = "O"
TEST_ATOMIC_DATA_DIGEST = "a" * 64


def _registry_entry(
    *,
    source_job_uuid: str | None = None,
    energy: float = TEST_ENERGY,
    magnetic_moment: float = TEST_MAGNETIC_MOMENT,
    gpaw_version: str = TEST_GPAW_VERSION,
    atomic_data_digest: str = TEST_ATOMIC_DATA_DIGEST,
    element: str = TEST_ELEMENT,
    cutoff_ev: float | None = None,
    eigensolver: GpawEigensolverName = GpawEigensolverName.etdm_fdpw,
) -> GpawIsolatedAtomRegistryEntry:
    simulation_settings = (
        GpawIsolatedAtomSimulationSettings()
        if cutoff_ev is None
        else GpawIsolatedAtomSimulationSettings(
            calculator_settings=GpawIsolatedAtomCalculatorSettings(cutoff_ev=cutoff_ev)
        )
    )
    runtime_provenance = GpawRuntimeProvenance(
        gpaw_version=gpaw_version,
        atomic_data=(
            GpawAtomicDataInfo(
                element_symbol=element, paw_setup_sha256_digest=atomic_data_digest
            ),
        ),
    )
    identity = GpawIsolatedAtomRegistryIdentity.from_simulation_settings_and_runtime_provenance(
        element=element,
        simulation_settings=simulation_settings,
        runtime_provenance=runtime_provenance,
        eigensolver=eigensolver,
    )
    return GpawIsolatedAtomRegistryEntry(
        identity=identity,
        simulation_settings=simulation_settings,
        energy=energy,
        magnetic_moment=magnetic_moment,
        eigensolver=eigensolver,
        runtime_provenance=runtime_provenance,
        source_job_uuid=source_job_uuid or str(uuid4()),
    )


@pytest.fixture
def registry_store(gpaw_isolated_atom_registry_store: MongoStore) -> MongoStore:
    return gpaw_isolated_atom_registry_store


@pytest.mark.needs_mongodb
def test_registry_upsert_inserts_one_entry_for_its_exact_identity(
    registry_store: MongoStore,
) -> None:
    entry = _registry_entry()
    repository = GpawIsolatedAtomRegistryRepository(registry_store)

    stored_entry = repository.upsert_success(entry)

    assert stored_entry == entry
    assert repository.get_entries((entry.identity,)) == {
        entry.identity.entry_fingerprint: entry
    }


@pytest.mark.needs_mongodb
def test_registry_retry_with_identical_scientific_result_keeps_first_source_job_uuid(
    registry_store: MongoStore,
) -> None:
    original_entry = _registry_entry(source_job_uuid=str(uuid4()))
    retry_entry = _registry_entry(source_job_uuid=str(uuid4()))
    repository = GpawIsolatedAtomRegistryRepository(registry_store)

    first_stored_entry = repository.upsert_success(original_entry)
    retry_stored_entry = repository.upsert_success(retry_entry)

    assert retry_entry.source_job_uuid != original_entry.source_job_uuid
    assert first_stored_entry == original_entry
    assert retry_stored_entry == original_entry
    assert retry_stored_entry.source_job_uuid == original_entry.source_job_uuid


@pytest.mark.needs_mongodb
def test_registry_stores_results_for_different_gpaw_versions(
    registry_store: MongoStore,
) -> None:
    original_entry = _registry_entry(source_job_uuid=str(uuid4()))
    updated_version_entry = _registry_entry(
        source_job_uuid=str(uuid4()), gpaw_version="25.8.0"
    )
    repository = GpawIsolatedAtomRegistryRepository(registry_store)

    repository.upsert_success(original_entry)
    stored_entry = repository.upsert_success(updated_version_entry)

    assert updated_version_entry.identity != original_entry.identity
    assert stored_entry == updated_version_entry


@pytest.mark.needs_mongodb
def test_registry_rejects_conflicting_result_for_existing_identity(
    registry_store: MongoStore,
) -> None:
    original_entry = _registry_entry()
    conflicting_entry = _registry_entry(energy=TEST_ENERGY - 1.0)
    repository = GpawIsolatedAtomRegistryRepository(registry_store)
    repository.upsert_success(original_entry)

    with pytest.raises(GpawIsolatedAtomRegistryConflictError):
        repository.upsert_success(conflicting_entry)

    assert repository.get_entries((original_entry.identity,)) == {
        original_entry.identity.entry_fingerprint: original_entry
    }


@pytest.mark.needs_mongodb
def test_registry_lookup_requires_exact_runtime_and_recipe_identity(
    registry_store: MongoStore,
) -> None:
    original_runtime_entry = _registry_entry()
    different_runtime_entry = _registry_entry(atomic_data_digest="b" * 64)
    different_recipe_entry = _registry_entry(cutoff_ev=600.0)
    repository = GpawIsolatedAtomRegistryRepository(registry_store)
    repository.upsert_success(original_runtime_entry)
    repository.upsert_success(different_runtime_entry)
    repository.upsert_success(different_recipe_entry)

    matching_entries = repository.get_entries((original_runtime_entry.identity,))

    assert matching_entries == {
        original_runtime_entry.identity.entry_fingerprint: original_runtime_entry
    }


@pytest.mark.needs_mongodb
def test_registry_lookup_distinguishes_selected_eigensolver(
    registry_store: MongoStore,
) -> None:
    etdm_entry = _registry_entry(eigensolver=GpawEigensolverName.etdm_fdpw)
    ppcg_entry = _registry_entry(eigensolver=GpawEigensolverName.ppcg)
    repository = GpawIsolatedAtomRegistryRepository(registry_store)
    repository.upsert_success(etdm_entry)
    repository.upsert_success(ppcg_entry)

    assert repository.get_entries((ppcg_entry.identity,)) == {
        ppcg_entry.identity.entry_fingerprint: ppcg_entry
    }


@pytest.mark.needs_mongodb
def test_registry_lookup_of_no_identities_returns_no_entries(
    registry_store: MongoStore,
) -> None:
    repository = GpawIsolatedAtomRegistryRepository(registry_store)

    assert repository.get_entries(()) == {}
