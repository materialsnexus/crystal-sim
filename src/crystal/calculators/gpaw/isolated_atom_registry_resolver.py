"""Resolve reusable GPAW isolated-atom references for direct GPAW labels."""

from __future__ import annotations

from crystal.repositories.gpaw_isolated_atom_registry import (
    GpawIsolatedAtomRegistryRepository,
)
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.elements import ElementStr
from crystal.schemas.gpaw import (
    GpawEigensolverName,
    GpawIsolatedAtomSimulationSettings,
    GpawScfChargemolTaskDocument,
)
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawIsolatedAtomReferenceSource,
    GpawManualIsolatedAtomReferenceSource,
    IsolatedAtomEnergy,
    IsolatedAtomEnergyReference,
)
from crystal.schemas.gpaw_isolated_atom_registry import GpawIsolatedAtomRegistryIdentity


def resolve_gpaw_isolated_atom_reference(
    *,
    label_task_documents: list[GpawScfChargemolTaskDocument],
    reference_source: GpawIsolatedAtomReferenceSource,
    registry_repository: GpawIsolatedAtomRegistryRepository | None = None,
) -> IsolatedAtomEnergyReference:
    """Resolve and validate the selected isolated-atom reference source."""
    successful_label_task_documents = _successful_gpaw_label_task_documents(
        label_task_documents
    )
    required_paw_data_by_element = _required_paw_data_by_element(
        successful_label_task_documents
    )
    gpaw_version = _gpaw_version_from_successful_labels(successful_label_task_documents)
    if isinstance(reference_source, GpawManualIsolatedAtomReferenceSource):
        _validate_isolated_atom_reference(
            successful_label_task_documents=successful_label_task_documents,
            required_paw_data_by_element=required_paw_data_by_element,
            isolated_atom_reference=reference_source.isolated_atom_reference,
        )
        return reference_source.isolated_atom_reference
    if registry_repository is None:
        msg = "GPAW registry reference resolution requires a registry repository."
        raise ValueError(msg)
    return _resolve_registry_isolated_atom_reference(
        successful_label_task_documents=successful_label_task_documents,
        required_paw_data_by_element=required_paw_data_by_element,
        gpaw_version=gpaw_version,
        simulation_settings=reference_source.simulation_settings,
        eigensolver=reference_source.eigensolver,
        registry_repository=registry_repository,
    )


def _resolve_registry_isolated_atom_reference(
    *,
    successful_label_task_documents: tuple[GpawScfChargemolTaskDocument, ...],
    required_paw_data_by_element: dict[ElementStr, GpawAtomicDataInfo],
    gpaw_version: str,
    simulation_settings: GpawIsolatedAtomSimulationSettings,
    eigensolver: GpawEigensolverName | None,
    registry_repository: GpawIsolatedAtomRegistryRepository,
) -> IsolatedAtomEnergyReference:
    solver_names = (
        (eigensolver,) if eigensolver is not None else tuple(GpawEigensolverName)
    )
    requested_registry_identities = tuple(
        GpawIsolatedAtomRegistryIdentity.from_simulation_settings_and_paw_data(
            simulation_settings=simulation_settings,
            paw_data=paw_data,
            gpaw_version=gpaw_version,
            eigensolver=solver_name,
        )
        for _, paw_data in sorted(required_paw_data_by_element.items())
        for solver_name in solver_names
    )
    registry_entries_by_fingerprint = registry_repository.get_entries(
        requested_registry_identities
    )
    selected_entries = {
        element: tuple(
            registry_entries_by_fingerprint[identity.entry_fingerprint]
            for identity in requested_registry_identities
            if identity.element == element
            and identity.entry_fingerprint in registry_entries_by_fingerprint
        )
        for element in required_paw_data_by_element
    }
    ambiguous_elements = tuple(
        element for element, entries in selected_entries.items() if len(entries) > 1
    )
    if ambiguous_elements:
        msg = (
            "The GPAW isolated-atom registry has multiple solver entries for "
            f"{', '.join(sorted(ambiguous_elements))}; specify an eigensolver."
        )
        raise ValueError(msg)
    missing_elements = tuple(
        element for element, entries in selected_entries.items() if not entries
    )
    if missing_elements:
        msg = (
            "The GPAW isolated-atom registry is missing entries for "
            f"{', '.join(missing_elements)}."
        )
        raise ValueError(msg)
    isolated_atom_reference = IsolatedAtomEnergyReference(
        simulation_settings=simulation_settings,
        isolated_atom_energies=tuple(
            IsolatedAtomEnergy(
                element=element,
                energy=entries[0].energy,
                eigensolver=entries[0].eigensolver,
                runtime_provenance=entries[0].runtime_provenance,
            )
            for element, entries in sorted(selected_entries.items())
        ),
    )
    _validate_isolated_atom_reference(
        successful_label_task_documents=successful_label_task_documents,
        required_paw_data_by_element=required_paw_data_by_element,
        isolated_atom_reference=isolated_atom_reference,
    )
    return isolated_atom_reference


def _successful_gpaw_label_task_documents(
    label_task_documents: object,
) -> tuple[GpawScfChargemolTaskDocument, ...]:
    if not isinstance(label_task_documents, list):
        msg = "GPAW isolated-atom reference resolution requires a list of GPAW labels."
        raise TypeError(msg)
    if not all(
        isinstance(task_document, GpawScfChargemolTaskDocument)
        for task_document in label_task_documents
    ):
        msg = "GPAW isolated-atom reference resolution received an unsupported label."
        raise TypeError(msg)
    successful_label_task_documents = tuple(
        task_document
        for task_document in label_task_documents
        if task_document.is_success
    )
    if not successful_label_task_documents:
        msg = "GPAW isolated-atom reference resolution requires at least one successful label."
        raise ValueError(msg)
    return successful_label_task_documents


def _required_paw_data_by_element(
    successful_label_task_documents: tuple[GpawScfChargemolTaskDocument, ...],
) -> dict[ElementStr, GpawAtomicDataInfo]:
    required_paw_data_by_element: dict[ElementStr, GpawAtomicDataInfo] = {}
    for task_document in successful_label_task_documents:
        runtime_provenance = task_document.runtime_provenance
        structure = task_document.structure
        assert runtime_provenance is not None
        assert structure is not None
        label_paw_data_by_element = runtime_provenance.atomic_data_by_element
        for element in structure.symbol_set:
            label_paw_data = label_paw_data_by_element[element]
            existing_paw_data = required_paw_data_by_element.get(element)
            if (
                existing_paw_data is not None
                and existing_paw_data.paw_setup_sha256_digest
                != label_paw_data.paw_setup_sha256_digest
            ):
                msg = "Successful GPAW labels must use consistent PAW data per element."
                raise ValueError(msg)
            required_paw_data_by_element[element] = label_paw_data
    return required_paw_data_by_element


def _gpaw_version_from_successful_labels(
    successful_label_task_documents: tuple[GpawScfChargemolTaskDocument, ...],
) -> str:
    """Require all successful labels to use one GPAW version."""
    gpaw_versions = {
        task_document.runtime_provenance.gpaw_version
        for task_document in successful_label_task_documents
        if task_document.runtime_provenance is not None
    }
    if len(gpaw_versions) != 1:
        msg = "Successful GPAW labels must use the same GPAW version."
        raise ValueError(msg)
    return next(iter(gpaw_versions))


def _validate_isolated_atom_reference(
    *,
    successful_label_task_documents: tuple[GpawScfChargemolTaskDocument, ...],
    required_paw_data_by_element: dict[ElementStr, GpawAtomicDataInfo],
    isolated_atom_reference: IsolatedAtomEnergyReference,
) -> None:
    reference_settings = isolated_atom_reference.simulation_settings
    reference_calculator_settings = reference_settings.calculator_settings
    reference_gpaw_version = isolated_atom_reference.isolated_atom_energies[
        0
    ].runtime_provenance.gpaw_version
    for task_document in successful_label_task_documents:
        label_settings = task_document.gpaw_scf_simulation_settings
        runtime_provenance = task_document.runtime_provenance
        if label_settings is None or runtime_provenance is None:
            msg = "Successful GPAW labels require SCF settings and runtime provenance."
            raise ValueError(msg)
        gpaw_settings = label_settings.gpaw_settings
        if gpaw_settings.xc != reference_calculator_settings.xc:
            msg = "GPAW label XC must match the isolated-atom reference."
            raise ValueError(msg)
        if gpaw_settings.cutoff_ev != reference_calculator_settings.cutoff_ev:
            msg = "GPAW label cutoff must match the isolated-atom reference."
            raise ValueError(msg)
        if runtime_provenance.gpaw_version != reference_gpaw_version:
            msg = "GPAW label version must match the isolated-atom reference."
            raise ValueError(msg)
    reference_paw_data_by_element = isolated_atom_reference.atomic_data_by_element
    for element, label_paw_data in required_paw_data_by_element.items():
        reference_paw_data = reference_paw_data_by_element.get(element)
        if reference_paw_data is None:
            msg = "The isolated-atom reference does not cover every GPAW label element."
            raise ValueError(msg)
        if (
            reference_paw_data.paw_setup_sha256_digest
            != label_paw_data.paw_setup_sha256_digest
        ):
            msg = "GPAW label PAW data must match the isolated-atom reference."
            raise ValueError(msg)
