"""Tests for persisted structure-generation outputs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from ase.io import read
from jobflow.core.flow import Flow
from jobflow.core.job import Job, job
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import MemoryStore
from monty.json import MontyDecoder
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.core.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.constants.structure_generators import (
    STRUCTURE_BLOB_STORE_NAME,
    STRUCTURE_GENERATION_EXTXYZ_FILENAME,
)
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.base import BaseTaskDocument
from crystal.schemas.formula_factories import ChemicalSpaceFormulaFactorySettings
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationProperties,
    InitialStructureGenerationSettings,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.properties import StructureGenerationProperties
from crystal.schemas.structure_generators import (
    BirchClusteringSettings,
    ClusterSamplingSettings,
    DirectSamplingSettings,
    InitialStructureGenerationTaskDocument,
)
from crystal.structure.generators.perturbation.rattle import (
    RattleAtomDisplacementMode,
    rattle_atom_positions,
)
from crystal.structure.generators.pyxtal.initial_structure_generation_maker import (
    InitialStructureGenerationMaker,
)
from crystal.structure.generators.sampling.direct_maker import DirectSamplingMaker
from tests.helpers.constants.structure import (
    TEST_BCC_FE_STRUCTURE,
    TEST_LINBO3_STRUCTURE,
    TEST_SI_STRUCTURE,
)
from tests.helpers.constants.structure_generators import (
    TEST_CHEMICAL_SPACE,
    TEST_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE,
    TEST_MAX_NUMBER_OF_ATOMS,
)
from tests.helpers.jobflow import make_blob_jobstore

TEST_INITIAL_STRUCTURE_GENERATION_SETTINGS = InitialStructureGenerationSettings(
    random_seed=42,
    formula_factory_settings=ChemicalSpaceFormulaFactorySettings(
        random_seed=42,
        chemical_space=list(TEST_CHEMICAL_SPACE),
        max_number_of_atoms=TEST_MAX_NUMBER_OF_ATOMS,
    ),
    space_groups=[225],
    max_generation_attempts_per_structure=TEST_MAX_GENERATION_ATTEMPTS_PER_STRUCTURE,
)


class _FirstTwoStructureDirectSampler:
    def fit_transform(self, structures: list[PMG_Structure]) -> dict[str, list[int]]:
        _ = structures
        return {"selected_indexes": [0, 1]}


@job
def _count_loaded_structures(structures: list[PMG_Structure]) -> int:
    assert all(isinstance(structure, PMG_Structure) for structure in structures)
    return len(structures)


@pytest.fixture(scope="session", autouse=True, name="mongo_uri")
def structure_output_persistence_mongo_uri() -> str:
    return "mongodb://blob-storage-tests-use-memory-jobstore.invalid/test"


def _build_direct_sampling_job() -> Job:
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

    simulation_settings = DirectSamplingSettings(
        structure_encoder_model=MlipModelRef(
            registry_model=QetModelRegistry.foundational_matpes_pbe
        ),
        birch_clustering_settings=BirchClusteringSettings(number_of_clusters=2),
        cluster_sampling_settings=ClusterSamplingSettings(num_samples_per_cluster=1),
    )
    maker = DirectSamplingMaker(simulation_settings=simulation_settings)
    return maker.make(structures=structures)


def _run_job_with_structure_blob_store(job: Job) -> tuple[JobStore, MemoryStore]:
    jobstore, structures_store = make_blob_jobstore()
    run_locally(job, root_dir=Path.cwd(), create_folders=True, store=jobstore)
    return jobstore, structures_store


def _assert_extxyz_artifact_matches_output(loaded_output: BaseTaskDocument) -> None:
    assert isinstance(loaded_output.properties, StructureGenerationProperties)
    assert loaded_output.dir_name is not None

    extxyz_path = Path(loaded_output.dir_name) / STRUCTURE_GENERATION_EXTXYZ_FILENAME
    assert extxyz_path.exists()

    extxyz_structures: list[PMG_Structure] = [
        AseAtomsAdaptor().get_structure(structure_atoms)
        for structure_atoms in read(extxyz_path.as_posix(), index=":")
    ]
    assert len(extxyz_structures) == len(loaded_output.properties.structures)
    assert all(
        StructureMatcher().fit(output_structure, extxyz_structure)
        for output_structure, extxyz_structure in zip(
            loaded_output.properties.structures, extxyz_structures, strict=True
        )
    )


def _assert_job_persists_one_structure_list_in_blob_store(
    *, jobstore: JobStore, structures_store: MemoryStore, job_uuid: str
) -> None:
    [stored_job_document] = list(jobstore.docs_store.query({"uuid": job_uuid}))
    stored_structures = stored_job_document["output"]["properties"]["structures"]
    assert stored_structures["store"] == STRUCTURE_BLOB_STORE_NAME
    assert "blob_uuid" in stored_structures
    assert len(list(structures_store.query({"job_uuid": job_uuid}))) == 1


@run_in_temp_dir
def test_initial_structure_generation_persists_structures_in_structures_blob_store() -> (
    None
):
    maker = InitialStructureGenerationMaker(
        simulation_settings=TEST_INITIAL_STRUCTURE_GENERATION_SETTINGS
    )
    job = maker.make(number_of_structures=2)
    assert isinstance(job, Job)

    jobstore, structures_store = _run_job_with_structure_blob_store(job)

    _assert_job_persists_one_structure_list_in_blob_store(
        jobstore=jobstore, structures_store=structures_store, job_uuid=job.uuid
    )


@run_in_temp_dir
def test_initial_structure_generation_get_output_loads_real_structures() -> None:
    maker = InitialStructureGenerationMaker(
        simulation_settings=TEST_INITIAL_STRUCTURE_GENERATION_SETTINGS
    )
    job = maker.make(number_of_structures=2)
    assert isinstance(job, Job)

    jobstore, _ = _run_job_with_structure_blob_store(job)

    loaded_output = MontyDecoder().process_decoded(
        jobstore.get_output(job.uuid, load=True)
    )
    assert isinstance(loaded_output, InitialStructureGenerationTaskDocument)
    assert loaded_output.errors == ""
    assert isinstance(loaded_output.properties, InitialStructureGenerationProperties)
    assert all(
        isinstance(structure, PMG_Structure)
        for structure in loaded_output.properties.structures
    )


@run_in_temp_dir
def test_initial_structure_generation_passes_blob_backed_structures_to_downstream_job() -> (
    None
):
    maker = InitialStructureGenerationMaker(
        simulation_settings=TEST_INITIAL_STRUCTURE_GENERATION_SETTINGS
    )
    upstream_job = maker.make(number_of_structures=2)
    assert isinstance(upstream_job, Job)

    downstream_job = _count_loaded_structures(
        structures=upstream_job.output.properties.structures
    )

    jobstore, structures_store = make_blob_jobstore()
    run_locally(
        Flow([upstream_job, downstream_job]),
        root_dir=Path.cwd(),
        create_folders=True,
        store=jobstore,
    )

    loaded_output = jobstore.get_output(downstream_job.uuid, load=True)
    assert loaded_output == 2

    _assert_job_persists_one_structure_list_in_blob_store(
        jobstore=jobstore, structures_store=structures_store, job_uuid=upstream_job.uuid
    )


@run_in_temp_dir
def test_initial_structure_generation_runs_without_jobstore() -> None:
    maker = InitialStructureGenerationMaker(
        simulation_settings=TEST_INITIAL_STRUCTURE_GENERATION_SETTINGS
    )
    job = maker.make(number_of_structures=2)
    assert isinstance(job, Job)

    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    assert isinstance(run_job, dict)

    output_doc = run_job[job.uuid][1].output
    assert isinstance(output_doc, InitialStructureGenerationTaskDocument)
    assert output_doc.errors == ""
    assert isinstance(output_doc.properties, InitialStructureGenerationProperties)
    assert all(
        isinstance(structure, PMG_Structure)
        for structure in output_doc.properties.structures
    )


@run_in_temp_dir
def test_initial_structure_generation_fails_with_docs_only_jobstore() -> None:
    maker = InitialStructureGenerationMaker(
        simulation_settings=TEST_INITIAL_STRUCTURE_GENERATION_SETTINGS
    )
    job = maker.make(number_of_structures=1)
    assert isinstance(job, Job)

    jobstore = JobStore(docs_store=MemoryStore())
    jobstore.connect()

    with pytest.raises(
        ValueError, match="Unrecognised additional store name: structures"
    ):
        job.run(store=jobstore)


@run_in_temp_dir
def test_direct_sampling_writes_extxyz_artifact_and_persists_selected_structures() -> (
    None
):
    job = _build_direct_sampling_job()
    assert isinstance(job, Job)

    with patch(
        "crystal.structure.generators.sampling.direct_maker.DirectSamplingMaker._build_direct_sampler",
        return_value=_FirstTwoStructureDirectSampler(),
    ):
        jobstore, structures_store = _run_job_with_structure_blob_store(job)

    loaded_output = MontyDecoder().process_decoded(
        jobstore.get_output(job.uuid, load=True)
    )
    assert isinstance(loaded_output, BaseTaskDocument)
    _assert_extxyz_artifact_matches_output(loaded_output)
    assert isinstance(loaded_output.properties, StructureGenerationProperties)
    _assert_job_persists_one_structure_list_in_blob_store(
        jobstore=jobstore, structures_store=structures_store, job_uuid=job.uuid
    )
