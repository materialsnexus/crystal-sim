"""Integration test for DIRECT sampling maker."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jobflow.managers.local import run_locally

from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.properties import StructureGenerationProperties
from crystal.schemas.structure_generators import (
    BirchClusteringSettings,
    ClusterSamplingSettings,
    DirectSamplingSettings,
    DirectSamplingTaskDocument,
)
from crystal.structure.generators.perturbation.rattle import (
    RattleAtomDisplacementMode,
    rattle_atom_positions,
)
from crystal.structure.generators.sampling.direct_maker import DirectSamplingMaker
from tests.helpers.constants.structure import (
    TEST_BCC_FE_STRUCTURE,
    TEST_LINBO3_STRUCTURE,
    TEST_SI_STRUCTURE,
)


@run_in_temp_dir
def test_run_direct_maker_integration() -> None:
    structures = [
        deepcopy(TEST_BCC_FE_STRUCTURE),
        deepcopy(TEST_LINBO3_STRUCTURE),
        deepcopy(TEST_SI_STRUCTURE),
        deepcopy(TEST_BCC_FE_STRUCTURE),
        deepcopy(TEST_LINBO3_STRUCTURE),
        deepcopy(TEST_SI_STRUCTURE),
    ]
    for index in [3, 4, 5]:
        structures[index] = rattle_atom_positions(
            structure=structures[index],
            random_seed=42,
            noise_scaling_factor=0.1,
            displacement_mode=RattleAtomDisplacementMode.normal,
        )

    number_of_clusters = 2
    num_samples_per_cluster = 1
    simulation_settings = DirectSamplingSettings(
        structure_encoder_model=MlipModelRef(
            registry_model=QetModelRegistry.foundational_matpes_pbe
        ),
        birch_clustering_settings=BirchClusteringSettings(
            number_of_clusters=number_of_clusters
        ),
        cluster_sampling_settings=ClusterSamplingSettings(
            num_samples_per_cluster=num_samples_per_cluster
        ),
    )
    maker = DirectSamplingMaker(simulation_settings=simulation_settings)
    job = maker.make(structures=structures)
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)

    output_doc = run_job[job.uuid][1].output
    assert isinstance(output_doc, DirectSamplingTaskDocument)
    assert output_doc.errors == ""
    assert isinstance(output_doc.properties, StructureGenerationProperties)
    assert (
        len(output_doc.properties.structures)
        == number_of_clusters * num_samples_per_cluster
    )


@run_in_temp_dir
def test_run_direct_maker_with_no_structures_skips_sampling_and_extxyz_export() -> None:
    simulation_settings = DirectSamplingSettings(
        structure_encoder_model=MlipModelRef(
            registry_model=QetModelRegistry.foundational_matpes_pbe
        ),
        birch_clustering_settings=BirchClusteringSettings(number_of_clusters=2),
        cluster_sampling_settings=ClusterSamplingSettings(num_samples_per_cluster=1),
    )
    maker = DirectSamplingMaker(simulation_settings=simulation_settings)
    job = maker.make(structures=[])

    with patch(
        "crystal.structure.generators.sampling.direct_maker.write_generated_structures_extxyz",
        side_effect=AssertionError("DIRECT must not export an empty extxyz file."),
    ):
        run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)

    output_doc = run_job[job.uuid][1].output
    assert isinstance(output_doc, DirectSamplingTaskDocument)
    assert output_doc.errors == ""
    assert output_doc.warnings
    assert isinstance(output_doc.properties, StructureGenerationProperties)
    assert output_doc.properties.structures == []
