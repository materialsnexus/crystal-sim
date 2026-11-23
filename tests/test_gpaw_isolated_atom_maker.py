"""Tests for static GPAW isolated-atom Jobflow orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import MongoStore

from crystal.calculators.gpaw.isolated_atom_maker import GpawIsolatedAtomMaker
from crystal.calculators.gpaw.isolated_atom_runner import GpawIsolatedAtomRunResult
from crystal.repositories.gpaw_isolated_atom_registry import (
    GpawIsolatedAtomRegistryConflictError,
)
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.gpaw import (
    GpawEigensolverName,
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
)
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawIsolatedAtomReferenceCollection,
)
from crystal.schemas.settings import ExecutorSettings

TEST_GPAW_VERSION = "26.7.0"
TEST_SETUP_DIGESTS = {"O": "a" * 64, "Si": "b" * 64}
TEST_NUMBER_OF_PROCESSES = 3
TEST_SILICON_FAILURE_MESSAGE = "Si atom failed"
TEST_SILICON_FALLBACK_FAILURE_MESSAGE = "Si PPCG fallback failed"


def _run_result(element: str, **_kwargs: object) -> GpawIsolatedAtomRunResult:
    return GpawIsolatedAtomRunResult(
        element=element,
        energy={"O": -3.0, "Si": -5.0}[element],
        magnetic_moment={"O": 2.0, "Si": 2.0}[element],
        eigensolver=GpawEigensolverName.etdm_fdpw,
        runtime_provenance=GpawRuntimeProvenance(
            gpaw_version=TEST_GPAW_VERSION,
            atomic_data=(
                GpawAtomicDataInfo(
                    element_symbol=element,
                    paw_setup_sha256_digest=TEST_SETUP_DIGESTS[element],
                ),
            ),
        ),
    )


def _fallback_run_result(element: str, **_kwargs: object) -> GpawIsolatedAtomRunResult:
    result = _run_result(element)
    return GpawIsolatedAtomRunResult(
        element=result.element,
        energy=result.energy - 1.0,
        magnetic_moment=result.magnetic_moment,
        eigensolver=GpawEigensolverName.ppcg,
        runtime_provenance=result.runtime_provenance,
    )


@patch("crystal.calculators.gpaw.isolated_atom_maker.run_gpaw_isolated_atom")
@pytest.mark.needs_mongodb
def test_gpaw_isolated_atom_maker_builds_and_runs_static_sorted_flow(
    mock_run_gpaw_isolated_atom: MagicMock,
    tmp_path: Path,
    mock_jobstore: JobStore,
    gpaw_isolated_atom_registry_store: MongoStore,
) -> None:
    mock_run_gpaw_isolated_atom.side_effect = _run_result
    simulation_settings = GpawIsolatedAtomSimulationSettings()
    executor_settings = ExecutorSettings(number_of_processes=TEST_NUMBER_OF_PROCESSES)
    maker = GpawIsolatedAtomMaker(
        simulation_settings=simulation_settings, executor_settings=executor_settings
    )

    flow = maker.make(elements=frozenset({"Si", "O"}))
    registry_store = gpaw_isolated_atom_registry_store

    collector_job = flow.jobs[-1]
    run = run_locally(flow, root_dir=tmp_path, create_folders=True, store=mock_jobstore)

    collection = run[collector_job.uuid][1].output
    assert isinstance(collection, GpawIsolatedAtomReferenceCollection)
    assert collection.is_success
    assert collection.reference is not None
    assert collection.reference.simulation_settings == simulation_settings
    assert collection.reference.energies_by_element == {"O": -3.0, "Si": -5.0}
    assert registry_store.count({}) == 2


@patch("crystal.calculators.gpaw.isolated_atom_maker.run_gpaw_isolated_atom")
@pytest.mark.needs_mongodb
def test_gpaw_isolated_atom_maker_returns_diagnostic_collection_after_atom_failure(
    mock_run_gpaw_isolated_atom: MagicMock,
    tmp_path: Path,
    mock_jobstore: JobStore,
    gpaw_isolated_atom_registry_store: MongoStore,
) -> None:
    def _run_or_fail(element: str, **kwargs: object) -> GpawIsolatedAtomRunResult:
        if element == "Si":
            if kwargs.get("use_ppcg_fallback", False):
                raise RuntimeError(TEST_SILICON_FALLBACK_FAILURE_MESSAGE)
            raise RuntimeError(TEST_SILICON_FAILURE_MESSAGE)
        return _run_result(element)

    mock_run_gpaw_isolated_atom.side_effect = _run_or_fail
    flow = GpawIsolatedAtomMaker(
        simulation_settings=GpawIsolatedAtomSimulationSettings()
    ).make(elements=frozenset({"Si", "O"}))
    registry_store = gpaw_isolated_atom_registry_store

    collector_job = flow.jobs[-1]
    run = run_locally(flow, root_dir=tmp_path, create_folders=True, store=mock_jobstore)

    collection = run[collector_job.uuid][1].output
    assert isinstance(collection, GpawIsolatedAtomReferenceCollection)
    assert not collection.is_success
    assert collection.reference is None
    assert [document.element for document in collection.task_documents] == ["O", "Si"]
    assert collection.task_documents[0].is_success
    assert TEST_SILICON_FAILURE_MESSAGE in collection.task_documents[1].errors
    assert TEST_SILICON_FALLBACK_FAILURE_MESSAGE in collection.task_documents[1].errors
    assert registry_store.count({}) == 1


@patch("crystal.calculators.gpaw.isolated_atom_maker.run_gpaw_isolated_atom")
@pytest.mark.needs_mongodb
def test_gpaw_isolated_atom_static_recovery_runs_ppcg_only_after_primary_failure(
    mock_run_gpaw_isolated_atom: MagicMock,
    tmp_path: Path,
    mock_jobstore: JobStore,
    gpaw_isolated_atom_registry_store: MongoStore,
) -> None:
    def _primary_or_fallback(
        element: str, **kwargs: object
    ) -> GpawIsolatedAtomRunResult:
        if kwargs.get("use_ppcg_fallback", False):
            return _fallback_run_result(element)
        raise RuntimeError(TEST_SILICON_FAILURE_MESSAGE)

    mock_run_gpaw_isolated_atom.side_effect = _primary_or_fallback
    flow = GpawIsolatedAtomMaker(
        simulation_settings=GpawIsolatedAtomSimulationSettings()
    ).make(elements=frozenset({"Si"}))
    collector_job = flow.jobs[-1]

    run = run_locally(flow, root_dir=tmp_path, create_folders=True, store=mock_jobstore)

    collection = run[collector_job.uuid][1].output
    assert isinstance(collection, GpawIsolatedAtomReferenceCollection)
    assert collection.is_success
    assert collection.task_documents[0].eigensolver is GpawEigensolverName.ppcg
    assert collection.reference is not None
    assert collection.reference.energies_by_element == {"Si": -6.0}
    assert [
        call.kwargs.get("use_ppcg_fallback", False)
        for call in mock_run_gpaw_isolated_atom.call_args_list
    ] == [False, True]
    assert gpaw_isolated_atom_registry_store.count({}) == 1


@patch("crystal.calculators.gpaw.isolated_atom_maker.run_gpaw_isolated_atom")
@pytest.mark.needs_mongodb
def test_gpaw_isolated_atom_static_recovery_preserves_both_failure_diagnostics(
    mock_run_gpaw_isolated_atom: MagicMock,
    tmp_path: Path,
    mock_jobstore: JobStore,
    gpaw_isolated_atom_registry_store: MongoStore,
) -> None:
    def _always_fail(**kwargs: object) -> GpawIsolatedAtomRunResult:
        if kwargs.get("use_ppcg_fallback", False):
            raise RuntimeError(TEST_SILICON_FALLBACK_FAILURE_MESSAGE)
        raise RuntimeError(TEST_SILICON_FAILURE_MESSAGE)

    mock_run_gpaw_isolated_atom.side_effect = _always_fail
    flow = GpawIsolatedAtomMaker(
        simulation_settings=GpawIsolatedAtomSimulationSettings()
    ).make(elements=frozenset({"Si"}))
    collector_job = flow.jobs[-1]

    run = run_locally(flow, root_dir=tmp_path, create_folders=True, store=mock_jobstore)

    collection = run[collector_job.uuid][1].output
    assert isinstance(collection, GpawIsolatedAtomReferenceCollection)
    assert not collection.is_success
    failed_document = collection.task_documents[0]
    assert TEST_SILICON_FAILURE_MESSAGE in failed_document.errors
    assert TEST_SILICON_FALLBACK_FAILURE_MESSAGE in failed_document.errors
    assert gpaw_isolated_atom_registry_store.count({}) == 0


@patch(
    "crystal.calculators.gpaw.isolated_atom_registry_jobflow."
    "GpawIsolatedAtomRegistryRepository.upsert_success"
)
@patch("crystal.calculators.gpaw.isolated_atom_maker.run_gpaw_isolated_atom")
@pytest.mark.needs_mongodb
def test_gpaw_isolated_atom_maker_does_not_hide_registry_conflicts(
    mock_run_gpaw_isolated_atom: MagicMock,
    mock_upsert_success: MagicMock,
    tmp_path: Path,
    mock_jobstore: JobStore,
) -> None:
    mock_run_gpaw_isolated_atom.side_effect = _run_result
    mock_upsert_success.side_effect = GpawIsolatedAtomRegistryConflictError(
        "conflicting atom result"
    )
    atom_flow = GpawIsolatedAtomMaker(
        simulation_settings=GpawIsolatedAtomSimulationSettings()
    ).make(elements=frozenset({"O"}))

    with pytest.raises(GpawIsolatedAtomRegistryConflictError, match="conflicting"):
        run_locally(
            atom_flow,
            root_dir=tmp_path,
            create_folders=True,
            store=mock_jobstore,
            raise_immediately=True,
        )


def test_gpaw_isolated_atom_maker_rejects_an_empty_element_set() -> None:
    maker = GpawIsolatedAtomMaker(
        simulation_settings=GpawIsolatedAtomSimulationSettings()
    )

    with pytest.raises(ValueError, match="at least one element"):
        maker.make(elements=frozenset())
