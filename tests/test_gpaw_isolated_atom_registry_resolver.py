"""Tests for resolving GPAW atom references from explicit source choices."""

from __future__ import annotations

from collections.abc import Collection

import pytest
from maggma.stores import MongoStore

from crystal.calculators.gpaw.isolated_atom_registry_resolver import (
    resolve_gpaw_isolated_atom_reference,
)
from crystal.repositories.gpaw_isolated_atom_registry import (
    GpawIsolatedAtomRegistryRepository,
)
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.gpaw import (
    GpawEigensolverName,
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
    GpawScfChargemolTaskDocument,
    GpawScfSimulationSettings,
)
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawManualIsolatedAtomReferenceSource,
    GpawRegistryIsolatedAtomReferenceSource,
    IsolatedAtomEnergy,
    IsolatedAtomEnergyReference,
)
from crystal.schemas.gpaw_isolated_atom_registry import (
    GpawIsolatedAtomRegistryEntry,
    GpawIsolatedAtomRegistryIdentity,
)
from crystal.schemas.properties import SinglePointProperties
from tests.helpers.builders.process_run_info import (
    build_successful_process_run_info_for_test,
)
from tests.helpers.constants.qet_finetuning import TEST_QET_FINE_TUNING_CHARGES
from tests.helpers.constants.structure import (
    TEST_RELAX_ENERGY_PLACEHOLDER,
    TEST_RELAX_STEP_INDEX,
    TEST_RELAX_STRESS_VECTOR_ZERO,
    TEST_RELAX_ZERO_FORCE_VECTOR,
    TEST_SI_STRUCTURE,
)

TEST_GPAW_VERSION = "25.7.0"
TEST_SILICON_SETUP_DIGEST = "a" * 64
TEST_SILICON_ISOLATED_ATOM_ENERGY = -5.0
TEST_SILICON_MAGNETIC_MOMENT = 2.0
TEST_SOURCE_JOB_UUID = "2dc60bb5-b177-4cd0-93f5-d15ea0834741"
TEST_AMBIGUOUS_SOLVER_ENTRIES_PATTERN = r"multiple solver entries.*Si"


def _runtime_provenance(
    gpaw_version: str = TEST_GPAW_VERSION, element: str = "Si"
) -> GpawRuntimeProvenance:
    return GpawRuntimeProvenance(
        gpaw_version=gpaw_version,
        atomic_data=(
            GpawAtomicDataInfo(
                element_symbol=element,
                paw_setup_sha256_digest=TEST_SILICON_SETUP_DIGEST,
            ),
        ),
    )


def _atom_settings() -> GpawIsolatedAtomSimulationSettings:
    return GpawIsolatedAtomSimulationSettings()


def _label_task_document(
    runtime_provenance: GpawRuntimeProvenance | None = None,
) -> GpawScfChargemolTaskDocument:
    return GpawScfChargemolTaskDocument(
        dir_name="gpaw-label",
        errors="",
        properties=SinglePointProperties(
            step=TEST_RELAX_STEP_INDEX,
            structure=TEST_SI_STRUCTURE,
            energy=TEST_RELAX_ENERGY_PLACEHOLDER,
            forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in TEST_SI_STRUCTURE],
            stress=TEST_RELAX_STRESS_VECTOR_ZERO,
            charges=TEST_QET_FINE_TUNING_CHARGES,
        ),
        structure=TEST_SI_STRUCTURE,
        chargemol_run_info=build_successful_process_run_info_for_test(),
        gpaw_scf_simulation_settings=GpawScfSimulationSettings(),
        runtime_provenance=runtime_provenance or _runtime_provenance(),
    )


def _label_task_documents() -> list[GpawScfChargemolTaskDocument]:
    return [_label_task_document()]


def _isolated_atom_reference(
    gpaw_version: str = TEST_GPAW_VERSION,
    eigensolver: GpawEigensolverName = GpawEigensolverName.etdm_fdpw,
) -> IsolatedAtomEnergyReference:
    return IsolatedAtomEnergyReference(
        simulation_settings=_atom_settings(),
        isolated_atom_energies=(
            IsolatedAtomEnergy(
                element="Si",
                energy=TEST_SILICON_ISOLATED_ATOM_ENERGY,
                eigensolver=eigensolver,
                runtime_provenance=_runtime_provenance(gpaw_version),
            ),
        ),
    )


def _registry_entry(
    eigensolver: GpawEigensolverName = GpawEigensolverName.etdm_fdpw,
) -> GpawIsolatedAtomRegistryEntry:
    simulation_settings = _atom_settings()
    runtime_provenance = _runtime_provenance()
    return GpawIsolatedAtomRegistryEntry(
        identity=GpawIsolatedAtomRegistryIdentity.from_simulation_settings_and_runtime_provenance(
            element="Si",
            simulation_settings=simulation_settings,
            runtime_provenance=runtime_provenance,
            eigensolver=eigensolver,
        ),
        simulation_settings=simulation_settings,
        energy=TEST_SILICON_ISOLATED_ATOM_ENERGY,
        magnetic_moment=TEST_SILICON_MAGNETIC_MOMENT,
        eigensolver=eigensolver,
        runtime_provenance=runtime_provenance,
        source_job_uuid=TEST_SOURCE_JOB_UUID,
    )


class _InMemoryRegistryRepository(GpawIsolatedAtomRegistryRepository):
    def __init__(self, entries: tuple[GpawIsolatedAtomRegistryEntry, ...]) -> None:
        self._entries_by_fingerprint = {
            entry.identity.entry_fingerprint: entry for entry in entries
        }

    def get_entries(
        self, registry_identities: Collection[GpawIsolatedAtomRegistryIdentity]
    ) -> dict[str, GpawIsolatedAtomRegistryEntry]:
        requested_fingerprints = {
            identity.entry_fingerprint for identity in registry_identities
        }
        return {
            fingerprint: entry
            for fingerprint, entry in self._entries_by_fingerprint.items()
            if fingerprint in requested_fingerprints
        }


def test_label_task_document_rejects_missing_runtime_paw_data() -> None:
    with pytest.raises(ValueError, match="runtime PAW data must cover"):
        _label_task_document(runtime_provenance=_runtime_provenance(element="O"))


def test_manual_reference_resolution_never_requires_a_registry() -> None:
    atom_reference = _isolated_atom_reference()

    resolved_reference = resolve_gpaw_isolated_atom_reference(
        label_task_documents=_label_task_documents(),
        reference_source=GpawManualIsolatedAtomReferenceSource(
            isolated_atom_reference=atom_reference
        ),
    )

    assert resolved_reference == atom_reference


def test_manual_reference_resolution_rejects_a_different_gpaw_version() -> None:
    with pytest.raises(ValueError, match="version must match"):
        resolve_gpaw_isolated_atom_reference(
            label_task_documents=_label_task_documents(),
            reference_source=GpawManualIsolatedAtomReferenceSource(
                isolated_atom_reference=_isolated_atom_reference("25.8.0")
            ),
        )


@pytest.mark.needs_mongodb
def test_registry_reference_resolution_uses_label_paw_data(
    gpaw_isolated_atom_registry_store: MongoStore,
) -> None:
    repository = GpawIsolatedAtomRegistryRepository(gpaw_isolated_atom_registry_store)
    entry = _registry_entry()
    repository.upsert_success(entry)

    resolved_reference = resolve_gpaw_isolated_atom_reference(
        label_task_documents=_label_task_documents(),
        reference_source=GpawRegistryIsolatedAtomReferenceSource(
            simulation_settings=_atom_settings()
        ),
        registry_repository=repository,
    )

    assert resolved_reference == _isolated_atom_reference()


@pytest.mark.needs_mongodb
def test_registry_reference_resolution_can_select_ppcg_fallback_result(
    gpaw_isolated_atom_registry_store: MongoStore,
) -> None:
    repository = GpawIsolatedAtomRegistryRepository(gpaw_isolated_atom_registry_store)
    repository.upsert_success(_registry_entry(GpawEigensolverName.ppcg))

    resolved_reference = resolve_gpaw_isolated_atom_reference(
        label_task_documents=_label_task_documents(),
        reference_source=GpawRegistryIsolatedAtomReferenceSource(
            simulation_settings=_atom_settings()
        ),
        registry_repository=repository,
    )

    assert resolved_reference == _isolated_atom_reference(
        eigensolver=GpawEigensolverName.ppcg
    )


def test_registry_reference_resolution_rejects_ambiguous_solver_entries_without_selection() -> (
    None
):
    repository = _InMemoryRegistryRepository(
        (
            _registry_entry(GpawEigensolverName.etdm_fdpw),
            _registry_entry(GpawEigensolverName.ppcg),
        )
    )

    with pytest.raises(ValueError, match=TEST_AMBIGUOUS_SOLVER_ENTRIES_PATTERN):
        resolve_gpaw_isolated_atom_reference(
            label_task_documents=_label_task_documents(),
            reference_source=GpawRegistryIsolatedAtomReferenceSource(
                simulation_settings=_atom_settings()
            ),
            registry_repository=repository,
        )


def test_registry_reference_resolution_requires_a_repository() -> None:
    with pytest.raises(ValueError, match="repository"):
        resolve_gpaw_isolated_atom_reference(
            label_task_documents=_label_task_documents(),
            reference_source=GpawRegistryIsolatedAtomReferenceSource(
                simulation_settings=_atom_settings()
            ),
        )


def test_manual_reference_resolution_rejects_datasets_without_successful_labels() -> (
    None
):
    label_task_documents = [
        GpawScfChargemolTaskDocument(
            dir_name="gpaw-label",
            errors="GPAW failed",
            properties=None,
            structure=None,
            chargemol_run_info=None,
        )
    ]

    with pytest.raises(ValueError, match="successful label"):
        resolve_gpaw_isolated_atom_reference(
            label_task_documents=label_task_documents,
            reference_source=GpawManualIsolatedAtomReferenceSource(
                isolated_atom_reference=_isolated_atom_reference()
            ),
        )


@pytest.mark.needs_mongodb
def test_registry_reference_resolution_rejects_missing_atom_entries(
    gpaw_isolated_atom_registry_store: MongoStore,
) -> None:

    with pytest.raises(ValueError, match="missing"):
        resolve_gpaw_isolated_atom_reference(
            label_task_documents=_label_task_documents(),
            reference_source=GpawRegistryIsolatedAtomReferenceSource(
                simulation_settings=_atom_settings()
            ),
            registry_repository=GpawIsolatedAtomRegistryRepository(
                gpaw_isolated_atom_registry_store
            ),
        )
