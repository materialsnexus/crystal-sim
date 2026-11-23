"""Tests for isolated-atom energy reference domain contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from crystal.calculators.gpaw.isolated_atom_runner import GpawIsolatedAtomRunResult
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.gpaw import (
    GpawEigensolverName,
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
)
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawIsolatedAtomReferenceCollection,
    GpawIsolatedAtomTaskDocument,
    IsolatedAtomEnergy,
    IsolatedAtomEnergyReference,
)

SILICON_SETUP_DIGEST = "a" * 64
OXYGEN_SETUP_DIGEST = "b" * 64
GPAW_VERSION = "25.1.0"
OTHER_GPAW_VERSION = "25.2.0"


def _atomic_data_info(
    element_symbol: str, paw_setup_sha256_digest: str
) -> GpawAtomicDataInfo:
    return GpawAtomicDataInfo(
        element_symbol=element_symbol, paw_setup_sha256_digest=paw_setup_sha256_digest
    )


def _runtime_provenance(
    *atomic_data: GpawAtomicDataInfo, gpaw_version: str = GPAW_VERSION
) -> GpawRuntimeProvenance:
    return GpawRuntimeProvenance(gpaw_version=gpaw_version, atomic_data=atomic_data)


def _isolated_atom_energy(
    element: str,
    energy: float,
    paw_setup_sha256_digest: str,
    *,
    gpaw_version: str = GPAW_VERSION,
) -> IsolatedAtomEnergy:
    return IsolatedAtomEnergy(
        element=element,
        energy=energy,
        runtime_provenance=_runtime_provenance(
            _atomic_data_info(element, paw_setup_sha256_digest),
            gpaw_version=gpaw_version,
        ),
    )


def _isolated_atom_task_document(
    element: str,
    energy: float,
    paw_setup_sha256_digest: str,
    *,
    magnetic_moment: float = 2.0,
) -> GpawIsolatedAtomTaskDocument:
    return GpawIsolatedAtomTaskDocument(
        dir_name=f"/calculation/{element}",
        element=element,
        energy=energy,
        magnetic_moment=magnetic_moment,
        eigensolver=GpawEigensolverName.etdm_fdpw,
        runtime_provenance=_runtime_provenance(
            _atomic_data_info(element, paw_setup_sha256_digest)
        ),
    )


def test_isolated_atom_reference_accepts_complete_finite_entries() -> None:
    reference = IsolatedAtomEnergyReference(
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
        isolated_atom_energies=(
            _isolated_atom_energy("Si", -5.0, SILICON_SETUP_DIGEST),
            _isolated_atom_energy("O", -3.0, OXYGEN_SETUP_DIGEST),
        ),
    )

    assert reference.energies_by_element == {"Si": -5.0, "O": -3.0}
    assert reference.elements == frozenset({"Si", "O"})
    assert (
        reference.atomic_data_by_element["Si"].paw_setup_sha256_digest
        == SILICON_SETUP_DIGEST
    )


def test_isolated_atom_reference_rejects_duplicate_entries() -> None:
    duplicate_entries = (
        _isolated_atom_energy("Si", -5.0, SILICON_SETUP_DIGEST),
        _isolated_atom_energy("si", -4.0, SILICON_SETUP_DIGEST),
    )

    with pytest.raises(ValueError, match="Duplicate isolated-atom energy"):
        IsolatedAtomEnergyReference(
            simulation_settings=GpawIsolatedAtomSimulationSettings(),
            isolated_atom_energies=duplicate_entries,
        )


def test_isolated_atom_energy_requires_one_matching_atomic_data_entry() -> None:
    with pytest.raises(ValueError, match="its element only"):
        IsolatedAtomEnergy(
            element="Si",
            energy=-5.0,
            runtime_provenance=_runtime_provenance(
                _atomic_data_info("Si", SILICON_SETUP_DIGEST),
                _atomic_data_info("O", OXYGEN_SETUP_DIGEST),
            ),
        )


def test_isolated_atom_reference_rejects_different_gpaw_versions() -> None:
    with pytest.raises(ValueError, match="same GPAW version"):
        IsolatedAtomEnergyReference(
            simulation_settings=GpawIsolatedAtomSimulationSettings(),
            isolated_atom_energies=(
                _isolated_atom_energy("Si", -5.0, SILICON_SETUP_DIGEST),
                _isolated_atom_energy(
                    "O", -3.0, OXYGEN_SETUP_DIGEST, gpaw_version=OTHER_GPAW_VERSION
                ),
            ),
        )


def test_isolated_atom_task_document_records_energy_only_success() -> None:
    task_document = _isolated_atom_task_document(
        "Si", -5.0, SILICON_SETUP_DIGEST, magnetic_moment=2.0
    )

    assert task_document.is_success
    assert task_document.isolated_atom_energy == _isolated_atom_energy(
        "Si", -5.0, SILICON_SETUP_DIGEST
    )
    assert task_document.magnetic_moment == 2.0


def test_failed_isolated_atom_task_exposes_error_when_energy_is_requested() -> None:
    task_document = GpawIsolatedAtomTaskDocument.from_directory_and_error(
        dir_path=Path("calculation"), error="GPAW failed to converge.", element="Si"
    )

    with pytest.raises(ValueError, match="GPAW failed to converge"):
        _ = task_document.isolated_atom_energy


def test_isolated_atom_task_document_creates_success_from_runner_result() -> None:
    run_result = GpawIsolatedAtomRunResult(
        element="Si",
        energy=-5.0,
        magnetic_moment=2.0,
        eigensolver=GpawEigensolverName.ppcg,
        runtime_provenance=_runtime_provenance(
            _atomic_data_info("Si", SILICON_SETUP_DIGEST)
        ),
    )

    task_document = GpawIsolatedAtomTaskDocument.from_directory_and_run_result(
        dir_path=Path("calculation"), run_result=run_result
    )

    assert task_document.is_success
    assert task_document.dir_name == "calculation"
    assert task_document.energy == -5.0
    assert task_document.eigensolver is GpawEigensolverName.ppcg


def test_isolated_atom_task_document_creates_failure_from_error() -> None:
    task_document = GpawIsolatedAtomTaskDocument.from_directory_and_error(
        dir_path=Path("calculation"), error="GPAW failed", element="Si"
    )

    assert not task_document.is_success
    assert task_document.errors == "GPAW failed"
    assert task_document.energy is None
    assert task_document.magnetic_moment is None
    assert task_document.runtime_provenance is None


@pytest.mark.parametrize(
    (
        "errors",
        "energy",
        "magnetic_moment",
        "eigensolver",
        "runtime_provenance",
        "expected_message",
    ),
    [
        (
            "",
            None,
            None,
            None,
            None,
            "missing required outputs: energy, magnetic_moment, eigensolver, runtime_provenance",
        ),
        (
            "",
            -5.0,
            2.0,
            None,
            _runtime_provenance(_atomic_data_info("Si", SILICON_SETUP_DIGEST)),
            "missing required outputs: eigensolver",
        ),
        ("GPAW failed", -5.0, None, None, None, "cannot include computed outputs"),
        ("GPAW failed", None, 2.0, None, None, "cannot include computed outputs"),
        (
            "GPAW failed",
            None,
            None,
            None,
            _runtime_provenance(_atomic_data_info("Si", SILICON_SETUP_DIGEST)),
            "cannot include computed outputs",
        ),
    ],
)
def test_isolated_atom_task_document_requires_consistent_outputs(
    errors: str,
    energy: float | None,
    magnetic_moment: float | None,
    eigensolver: GpawEigensolverName | None,
    runtime_provenance: GpawRuntimeProvenance | None,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        GpawIsolatedAtomTaskDocument(
            dir_name="calculation",
            element="Si",
            errors=errors,
            energy=energy,
            magnetic_moment=magnetic_moment,
            eigensolver=eigensolver,
            runtime_provenance=runtime_provenance,
        )


def test_isolated_atom_collection_builds_reference_after_all_tasks_succeed() -> None:
    silicon_task_document = _isolated_atom_task_document(
        "Si", -5.0, SILICON_SETUP_DIGEST
    )
    oxygen_task_document = _isolated_atom_task_document("O", -3.0, OXYGEN_SETUP_DIGEST)

    collection = GpawIsolatedAtomReferenceCollection.from_task_documents(
        task_documents=(silicon_task_document, oxygen_task_document),
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
    )

    assert collection.is_success
    assert collection.reference is not None
    assert collection.reference.energies_by_element == {"Si": -5.0, "O": -3.0}
    assert collection.task_documents == (silicon_task_document, oxygen_task_document)


def test_isolated_atom_collection_preserves_partial_failure_without_reference() -> None:
    silicon_task_document = _isolated_atom_task_document(
        "Si", -5.0, SILICON_SETUP_DIGEST
    )
    oxygen_failure = GpawIsolatedAtomTaskDocument.from_directory_and_error(
        dir_path=Path("calculation/O"), error="GPAW failed", element="O"
    )

    collection = GpawIsolatedAtomReferenceCollection.from_task_documents(
        task_documents=(silicon_task_document, oxygen_failure),
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
    )

    assert not collection.is_success
    assert collection.reference is None
    assert collection.task_documents == (silicon_task_document, oxygen_failure)


def test_isolated_atom_collection_rejects_duplicate_task_document_elements() -> None:
    silicon_task_document = _isolated_atom_task_document(
        "Si", -5.0, SILICON_SETUP_DIGEST
    )
    silicon_failure = GpawIsolatedAtomTaskDocument.from_directory_and_error(
        dir_path=Path("calculation/Si-retry"), error="GPAW failed", element="Si"
    )

    with pytest.raises(ValueError, match="Duplicate GPAW isolated-atom task"):
        GpawIsolatedAtomReferenceCollection.from_task_documents(
            task_documents=(silicon_task_document, silicon_failure),
            simulation_settings=GpawIsolatedAtomSimulationSettings(),
        )
