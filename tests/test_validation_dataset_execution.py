"""Behavioral tests for validation-dataset execution stages."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jobflow.core.flow import Flow
from jobflow.core.job import job
from jobflow.managers.local import run_locally
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.exceptions import OverlappingAtomsException
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.mlip import MlipRelaxTaskDocument
from crystal.schemas.validation_dataset import (
    ValidationDatasetArtifactReference,
    ValidationDatasetInterfaceGenerationSettings,
    ValidationDatasetStructureBatchReference,
)
from crystal.structure.interfaces.builder import Interface, InterfaceFactory
from crystal.structure.validation_dataset.interface_generation import (
    generate_validation_dataset_interface_structures,
)
from crystal.structure.validation_dataset.md_execution import (
    build_validation_dataset_md_settings_at_temperature,
)
from crystal.structure.validation_dataset.slab_relaxation import (
    collect_successfully_relaxed_slabs,
)
from crystal.structure.validation_dataset.structure_artifacts import (
    load_validation_dataset_structure,
)
from tests.helpers.builders.properties import build_relax_properties_for_test
from tests.helpers.constants.structure import (
    TEST_001_FE_SLAB,
    TEST_111_FE_SLAB,
    TEST_INTERFACE,
    TEST_INTERFACE_FACTORY,
    TEST_SI_STRUCTURE,
    TEST_TOP_SLAB,
)
from tests.helpers.jobflow import make_blob_jobstore
from tests.helpers.validation_dataset import (
    build_validation_dataset_md_simulation_settings_for_test,
)


@job
def _count_structure_atoms(structure: PMG_Structure) -> int:
    return len(structure)


def test_slab_relaxation_collection_keeps_metadata_and_drops_failures() -> None:
    successful_document = MlipRelaxTaskDocument(
        properties=build_relax_properties_for_test(
            structure=deepcopy(TEST_TOP_SLAB), energy=0.0
        )
    )
    failed_document = MlipRelaxTaskDocument(
        properties=None, errors="MLIP relaxation failed"
    )

    result = collect_successfully_relaxed_slabs(
        source_structure_id="linbo3",
        input_slabs=[TEST_TOP_SLAB, TEST_TOP_SLAB],
        relax_task_documents=[successful_document, failed_document],
    )

    assert len(result.slabs) == 1
    assert result.failed_relaxation_count == 1
    assert result.slabs[0].miller_index == TEST_TOP_SLAB.miller_index


@run_in_temp_dir
def test_structure_loader_reads_one_structure_from_a_blob_batch() -> None:
    jobstore, structures_store = make_blob_jobstore()
    structures_store.update(
        {"blob_uuid": "slab-batch-0", "data": [TEST_SI_STRUCTURE.as_dict()]}
    )
    structure_batch = ValidationDatasetStructureBatchReference(
        batch_id="slab-0",
        structure_artifact=ValidationDatasetArtifactReference(
            artifact_id="slab-batch-0", store_name=STRUCTURE_BLOB_STORE_NAME
        ),
        item_count=1,
    )

    structure_job = load_validation_dataset_structure(
        structure_batch=structure_batch, structure_index=0
    )
    consumer_job = _count_structure_atoms(structure=structure_job.output)
    run_result = run_locally(
        Flow([structure_job, consumer_job]),
        root_dir=Path.cwd(),
        create_folders=True,
        store=jobstore,
    )

    loaded_structure = run_result[structure_job.uuid][1].output
    assert loaded_structure.composition == TEST_SI_STRUCTURE.composition
    assert run_result[consumer_job.uuid][1].output == len(TEST_SI_STRUCTURE)


def test_interface_generation_applies_match_and_atom_bounds() -> None:
    top_slab = TEST_INTERFACE_FACTORY.top_slab
    bottom_slab = TEST_INTERFACE_FACTORY.bottom_slab
    generation_settings = ValidationDatasetInterfaceGenerationSettings(
        assembly_settings={
            "offsets": [(0.0, 0.0), (0.5, 0.5)],
            "gaps": [1.0],
            "vacuum": 12.0,
        },
        match_settings={
            "maximum_atoms": len(top_slab) + len(bottom_slab),
            "maximum_matches_per_slab_pair": 2,
            "maximum_generation_attempts": 5,
        },
    )

    interfaces = generate_validation_dataset_interface_structures(
        top_slabs=[top_slab],
        bottom_slabs=[bottom_slab],
        interface_generation_settings=generation_settings,
    )

    assert interfaces
    assert all(
        len(interface) <= generation_settings.match_settings.maximum_atoms
        for interface in interfaces
    )
    assert len(interfaces) <= (
        generation_settings.match_settings.maximum_matches_per_slab_pair
        * len(generation_settings.assembly_settings.offsets)
    )


def test_interface_generation_returns_no_interfaces_without_zsl_matches() -> None:
    generation_settings = ValidationDatasetInterfaceGenerationSettings(
        assembly_settings={"offsets": [(0.0, 0.0)], "gaps": [1.0], "vacuum": 12.0},
        match_settings={
            "maximum_atoms": len(TEST_001_FE_SLAB) + len(TEST_111_FE_SLAB),
            "maximum_matches_per_slab_pair": 1,
            "maximum_generation_attempts": 2,
        },
    )

    interfaces = generate_validation_dataset_interface_structures(
        top_slabs=[TEST_001_FE_SLAB],
        bottom_slabs=[TEST_111_FE_SLAB],
        interface_generation_settings=generation_settings,
    )

    assert interfaces == []


def test_interface_generation_stops_consuming_matches_at_attempt_limit() -> None:
    generated_candidate_count = 0

    def generate_many_interfaces(*args: object, **kwargs: object) -> Iterator[object]:
        del args, kwargs
        nonlocal generated_candidate_count
        for _ in range(5):
            generated_candidate_count += 1
            yield TEST_INTERFACE

    generation_settings = ValidationDatasetInterfaceGenerationSettings(
        assembly_settings={
            "offsets": [(0.0, 0.0), (0.5, 0.5)],
            "gaps": [1.0],
            "vacuum": 12.0,
        },
        match_settings={
            "maximum_atoms": len(TEST_INTERFACE),
            "maximum_matches_per_slab_pair": 1,
            "maximum_generation_attempts": 2,
        },
    )

    with patch.object(
        InterfaceFactory, "generate_interfaces", generate_many_interfaces
    ):
        interfaces = generate_validation_dataset_interface_structures(
            top_slabs=[TEST_INTERFACE_FACTORY.top_slab],
            bottom_slabs=[TEST_INTERFACE_FACTORY.bottom_slab],
            interface_generation_settings=generation_settings,
        )

    assert len(interfaces) == 2
    assert generated_candidate_count == 4


def test_interface_generation_continues_after_a_slab_pair_fails() -> None:
    generation_call_count = 0

    def generate_interfaces(*args: object, **kwargs: object) -> Iterator[Interface]:
        del args, kwargs
        nonlocal generation_call_count
        generation_call_count += 1
        yield TEST_INTERFACE
        if generation_call_count == 1:
            raise OverlappingAtomsException

    generation_settings = ValidationDatasetInterfaceGenerationSettings(
        assembly_settings={"offsets": [(0.0, 0.0)], "gaps": [1.0], "vacuum": 12.0},
        match_settings={
            "maximum_atoms": len(TEST_INTERFACE),
            "maximum_matches_per_slab_pair": 1,
            "maximum_generation_attempts": 2,
        },
    )

    with patch.object(InterfaceFactory, "generate_interfaces", generate_interfaces):
        interfaces = generate_validation_dataset_interface_structures(
            top_slabs=[
                TEST_INTERFACE_FACTORY.top_slab,
                TEST_INTERFACE_FACTORY.top_slab,
            ],
            bottom_slabs=[TEST_INTERFACE_FACTORY.bottom_slab],
            interface_generation_settings=generation_settings,
        )

    assert interfaces == [TEST_INTERFACE, TEST_INTERFACE]


def test_temperature_sampling_updates_the_thermostat() -> None:
    settings = build_validation_dataset_md_simulation_settings_for_test()

    updated = build_validation_dataset_md_settings_at_temperature(
        settings, temperature_k=900.0
    )

    assert updated.run_settings.dynamics.thermostat is not None
    assert updated.run_settings.dynamics.thermostat.temperature_k == 900.0
    assert settings.run_settings.dynamics.thermostat is not None
    assert settings.run_settings.dynamics.thermostat.temperature_k == 300.0
