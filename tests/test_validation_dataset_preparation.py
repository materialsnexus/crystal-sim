"""Behavioral tests for validation-dataset bulk and slab preparation."""

from __future__ import annotations

from pathlib import Path

from jobflow.managers.local import run_locally
from monty.json import MontyDecoder

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.validation_dataset import ValidationDatasetSlabGenerationSettings
from crystal.structure.validation_dataset.slab_generation import (
    generate_validation_dataset_slab_preparation_properties,
    generate_validation_dataset_slab_structures,
)
from tests.helpers.constants.structure import TEST_LINBO3_STRUCTURE
from tests.helpers.jobflow import make_blob_jobstore


def test_signed_slab_generation_returns_all_eligible_terminations() -> None:
    settings = ValidationDatasetSlabGenerationSettings(
        miller_indices=((0, 0, 1), (-1, 0, 1)),
        maximum_atoms=220,
        maximum_generation_attempts=50,
        supercell=(1, 1, 5),
        vacuum_thickness_in_layers=2,
    )

    slabs = generate_validation_dataset_slab_structures(
        structure=TEST_LINBO3_STRUCTURE, slab_generation_settings=settings
    )

    assert slabs
    assert {tuple(slab.miller_index) for slab in slabs} == {(0, 0, 1), (-1, 0, 1)}
    assert all(len(slab) <= settings.maximum_atoms for slab in slabs)
    assert len(
        {
            (
                tuple(slab.miller_index),
                slab.top_layer_termination,
                slab.bottom_layer_termination,
            )
            for slab in slabs
        }
    ) == len(slabs)


@run_in_temp_dir
def test_slab_generation_job_persists_slabs_in_structure_blob_store() -> None:
    settings = ValidationDatasetSlabGenerationSettings(
        miller_indices=((0, 0, 1),),
        maximum_atoms=220,
        supercell=(1, 1, 5),
        vacuum_thickness_in_layers=2,
    )
    slab_job = generate_validation_dataset_slab_preparation_properties(
        source_structure_id="linbo3",
        structure=TEST_LINBO3_STRUCTURE,
        slab_generation_settings=settings,
    )
    jobstore, _ = make_blob_jobstore()

    run_locally(slab_job, root_dir=Path.cwd(), create_folders=True, store=jobstore)

    stored_job_document = next(jobstore.docs_store.query({"uuid": slab_job.uuid}))
    stored_slabs = stored_job_document["output"]["slabs"]
    assert stored_slabs["store"] == STRUCTURE_BLOB_STORE_NAME

    loaded_output = MontyDecoder().process_decoded(
        jobstore.get_output(slab_job.uuid, load=True)
    )
    assert loaded_output.source_structure_id == "linbo3"
    assert loaded_output.slabs
