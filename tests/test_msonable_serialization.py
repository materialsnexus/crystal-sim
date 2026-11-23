"""Tests for MSONable serialization roundtrips."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from crystal.schemas.adhesion import AdhesionWorkflowInputs
from crystal.schemas.interface_geometry_optimisation import (
    GeometryOptimisationCandidateRecord,
    GeometryOptimisationResult,
    GeometryOptimisationSlabInputs,
    InterfaceGeometryOptimisationInputs,
)
from crystal.schemas.mlip import MlipRelaxTaskDocument
from crystal.schemas.properties import RelaxProperties
from crystal.schemas.qe import QeScfSimulationSettings
from crystal.structure.interfaces.settings import InterfaceAssemblySettings
from tests.helpers.constants.properties import MOCK_MLIP_RELAX_PROPERTIES
from tests.helpers.constants.structure import (
    DEFAULT_INTERFACE_GAPS,
    DEFAULT_INTERFACE_OFFSETS,
    DEFAULT_INTERFACE_VACUUM,
    TEST_INTERFACE,
    TEST_INTERFACE_FACTORY,
    TEST_LINBO3_SLAB,
)


def test_interface_assembly_settings_msonable_roundtrip() -> None:
    settings = InterfaceAssemblySettings(
        offsets=DEFAULT_INTERFACE_OFFSETS,
        gaps=DEFAULT_INTERFACE_GAPS,
        vacuum=DEFAULT_INTERFACE_VACUUM,
    )
    settings_dict = settings.as_dict()
    settings_roundtrip = InterfaceAssemblySettings.from_dict(settings_dict)
    assert settings_roundtrip.as_dict() == settings_dict


def test_adhesion_workflow_inputs_msonable_roundtrip() -> None:
    settings = InterfaceAssemblySettings(
        offsets=DEFAULT_INTERFACE_OFFSETS,
        gaps=DEFAULT_INTERFACE_GAPS,
        vacuum=DEFAULT_INTERFACE_VACUUM,
    )
    inputs = AdhesionWorkflowInputs(
        top_slab=TEST_LINBO3_SLAB,
        bottom_slab=TEST_LINBO3_SLAB,
        assembly_settings=settings,
    )
    inputs_dict = inputs.as_dict()
    inputs_roundtrip = AdhesionWorkflowInputs.from_dict(inputs_dict)
    assert inputs_roundtrip.as_dict() == inputs_dict


def test_interface_geometry_optimisation_nested_slab_inputs_msonable_roundtrip() -> (
    None
):
    inputs = InterfaceGeometryOptimisationInputs(
        slab_inputs=GeometryOptimisationSlabInputs(
            top_slab=deepcopy(TEST_INTERFACE_FACTORY.top_slab),
            bottom_slab=deepcopy(TEST_INTERFACE_FACTORY.bottom_slab),
        )
    )

    restored = InterfaceGeometryOptimisationInputs.from_dict(inputs.as_dict())

    assert restored.as_dict() == inputs.as_dict()


def test_interface_geometry_optimisation_interface_inputs_msonable_roundtrip() -> None:
    inputs = InterfaceGeometryOptimisationInputs(interface=deepcopy(TEST_INTERFACE))

    restored = InterfaceGeometryOptimisationInputs.from_dict(inputs.as_dict())

    assert restored.as_dict() == inputs.as_dict()


def test_geometry_optimisation_result_msonable_roundtrip() -> None:
    result = GeometryOptimisationResult(
        selected_candidate_index=0,
        selected_structure=deepcopy(TEST_INTERFACE),
        selected_final_energy=-1.0,
        candidate_records=[
            GeometryOptimisationCandidateRecord(
                candidate_index=0, final_relax_energy=-1.0
            )
        ],
    )

    restored = GeometryOptimisationResult.from_dict(result.as_dict())

    assert restored.as_dict() == result.as_dict()


def test_qe_scf_simulation_settings_msonable_roundtrip_with_path(
    tmp_path: Path,
) -> None:
    pseudopotential_directory = tmp_path / "pseudopotentials"
    settings = QeScfSimulationSettings(
        pseudopotentials={"Si": "Si.pbe"},
        set_pseudopotential_directory=pseudopotential_directory,
    )
    settings_dict = settings.as_dict()
    settings_roundtrip = QeScfSimulationSettings.from_dict(settings_dict)
    assert isinstance(settings_roundtrip, QeScfSimulationSettings)
    assert settings_roundtrip.as_dict() == settings_dict
    assert settings_roundtrip.set_pseudopotential_directory == pseudopotential_directory


def test_mlip_relax_task_document_from_dict_preserves_nested_properties() -> None:
    document = MlipRelaxTaskDocument(properties=deepcopy(MOCK_MLIP_RELAX_PROPERTIES))

    restored = MlipRelaxTaskDocument.from_dict(document.as_dict())

    assert isinstance(restored, MlipRelaxTaskDocument)
    assert isinstance(restored.properties, RelaxProperties)
    assert isinstance(document.properties, RelaxProperties)
    assert restored.properties.final_energy == pytest.approx(
        document.properties.final_energy
    )
    assert restored.properties.relaxed_structure.as_dict() == (
        document.properties.relaxed_structure.as_dict()
    )
