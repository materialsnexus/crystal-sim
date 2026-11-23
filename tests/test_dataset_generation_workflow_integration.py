"""Integration tests for persisted dataset-generation workflows.

These tests use real MongoDB, Jobflow, and structure persistence. The slow test also
runs real augmentation and DIRECT sampling. It replaces PyXtal generation and MLIP
calculations with quick deterministic test makers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from pathlib import Path
from uuid import uuid4

import pytest
from jobflow.core.flow import Flow
from jobflow.core.job import job
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import GridFSStore, MongoStore
from monty.json import MontyDecoder
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.relax_maker import MlipRelaxBatchMaker
from crystal.calculators.mlip.single_point_maker import MlipSinglePointMaker
from crystal.constants.mlip import MLIP_RELAX_TASK_DOCUMENT_BLOB_STORE_NAME
from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationProperties,
    InitialStructureGenerationSettings,
    StructureIndexRange,
)
from crystal.schemas.mlip import (
    MlipRelaxMakerSettings,
    MlipRelaxSimulationSettings,
    MlipRelaxTaskDocument,
    MlipSinglePointMakerSettings,
    MlipSinglePointTaskDocument,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.properties import (
    RelaxProperties,
    SinglePointProperties,
    StructureGenerationProperties,
)
from crystal.schemas.relax_settings import OptimizerType
from crystal.schemas.structure_augmentation import (
    StructureAugmentationAndFilteringProperties,
    StructureAugmentationAndFilteringStatistics,
    StructureGenerationSourceStatistics,
)
from crystal.schemas.structure_generators import (
    BirchClusteringSettings,
    ClusterSamplingSettings,
    DatasetGenerationWorkflowSettings,
    DatasetGenerationWorkflowTaskDocument,
    DirectSamplingMakerSettings,
    DirectSamplingSettings,
    DirectSamplingTaskDocument,
    InitialStructureGenerationMakerSettings,
    InitialStructureGenerationTaskDocument,
    RattleAtomPositionsSettings,
    StretchCellSettings,
    StructureAugmentationAndFilteringMakerSettings,
    StructureAugmentationAndFilteringSettings,
    StructureAugmentationAndFilteringTaskDocument,
)
from crystal.schemas.structure_labelling import StructureLabellingWorkflowSettings
from crystal.structure.generators.dataset_generation_workflow_maker import (
    DatasetGenerationWorkflowMaker,
    build_structure_generation_workflow_task_document,
)
from crystal.structure.generators.pyxtal.initial_structure_generation_maker import (
    InitialStructureGenerationMaker,
)
from crystal.structure.labelling.structure_labelling_workflow_maker import (
    StructureLabellingWorkflowMaker,
)
from tests.helpers.builders.single_point_maker_settings import (
    build_mlip_single_point_maker_settings_for_test,
)
from tests.helpers.constants.structure import TEST_LINBO3_STRUCTURE

MONGO_TEST_DATABASE = "test"
DATASET_GENERATION_WORKFLOW_TEST_COLLECTION_PREFIX = "dataset_generation_workflow"
INITIAL_GENERATION_RANGE_PROPERTY_KEY = "generation_range"
AUGMENTATION_STATISTICS_PROPERTY_KEY = "statistics"
FINAL_AUGMENTATION_STATISTICS_PROPERTY_KEY = "augmentation_and_filtering_statistics"
FINAL_WORKFLOW_RESULT_STRUCTURE_COUNT = 400
LARGE_WORKFLOW_INITIAL_STRUCTURE_COUNT = 2_000
LARGE_WORKFLOW_RATTLED_VARIANTS_PER_RELAXED_STRUCTURE = 1
LARGE_WORKFLOW_STRETCHED_VARIANTS_PER_SOURCE = 1
LARGE_WORKFLOW_DIRECT_CLUSTER_COUNT = 1
LARGE_WORKFLOW_DIRECT_SELECTIONS_PER_CLUSTER = 10_000
LARGE_WORKFLOW_EXPECTED_SELECTED_STRUCTURE_COUNT = 10_000


@job
def _count_downstream_structures(structures: list[PMG_Structure]) -> int:
    assert all(isinstance(structure, PMG_Structure) for structure in structures)
    return len(structures)


def _structure_fingerprint(structure: PMG_Structure) -> str:
    serialized_structure = dumps(structure.as_dict(), sort_keys=True)
    return sha256(serialized_structure.encode("utf-8")).hexdigest()


@contextmanager
def _temporary_mongo_jobstore(
    mongo_uri: str,
) -> Generator[tuple[JobStore, GridFSStore], None, None]:
    collection_suffix = uuid4().hex
    docs_store = MongoStore(
        database=MONGO_TEST_DATABASE,
        collection_name=(
            f"{DATASET_GENERATION_WORKFLOW_TEST_COLLECTION_PREFIX}_jobs_"
            f"{collection_suffix}"
        ),
        host=mongo_uri,
    )
    structures_store = GridFSStore(
        database=MONGO_TEST_DATABASE,
        collection_name=(
            f"{DATASET_GENERATION_WORKFLOW_TEST_COLLECTION_PREFIX}_structures_"
            f"{collection_suffix}"
        ),
        host=mongo_uri,
    )
    mlip_relax_store = GridFSStore(
        database=MONGO_TEST_DATABASE,
        collection_name=(
            f"{DATASET_GENERATION_WORKFLOW_TEST_COLLECTION_PREFIX}_mlip_relax_"
            f"{collection_suffix}"
        ),
        host=mongo_uri,
    )
    jobstore = JobStore(
        docs_store=docs_store,
        additional_stores={
            STRUCTURE_BLOB_STORE_NAME: structures_store,
            MLIP_RELAX_TASK_DOCUMENT_BLOB_STORE_NAME: mlip_relax_store,
        },
    )
    jobstore.connect()
    try:
        yield jobstore, structures_store
    finally:
        docs_store._collection.drop()
        structures_store._files_collection.drop()
        structures_store._chunks_collection.drop()
        mlip_relax_store._files_collection.drop()
        mlip_relax_store._chunks_collection.drop()


def _completed_direct_sampling_task_document() -> DirectSamplingTaskDocument:
    return DirectSamplingTaskDocument(
        properties=StructureGenerationProperties(
            structures=[
                deepcopy(TEST_LINBO3_STRUCTURE)
                for _ in range(FINAL_WORKFLOW_RESULT_STRUCTURE_COUNT)
            ]
        )
    )


def _completed_initial_structure_generation_task_document() -> (
    InitialStructureGenerationTaskDocument
):
    return InitialStructureGenerationTaskDocument(
        properties=InitialStructureGenerationProperties(
            generation_range=StructureIndexRange(last_structure_index=0),
            structures=[deepcopy(TEST_LINBO3_STRUCTURE)],
        )
    )


def _completed_augmentation_task_document() -> (
    StructureAugmentationAndFilteringTaskDocument
):
    empty_source_statistics = StructureGenerationSourceStatistics()
    return StructureAugmentationAndFilteringTaskDocument(
        properties=StructureAugmentationAndFilteringProperties(
            structures=[],
            statistics=StructureAugmentationAndFilteringStatistics(
                initial=empty_source_statistics,
                mlip_relax=empty_source_statistics,
                rattle_after_mlip_relax=empty_source_statistics,
                stretch_after_rattle=empty_source_statistics,
                stretch_after_mlip_relax=empty_source_statistics,
            ),
        )
    )


def _large_dataset_generation_workflow_settings() -> DatasetGenerationWorkflowSettings:
    model = MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    return DatasetGenerationWorkflowSettings(
        number_of_initial_structures=LARGE_WORKFLOW_INITIAL_STRUCTURE_COUNT,
        mlip_relax_structures_per_job=LARGE_WORKFLOW_INITIAL_STRUCTURE_COUNT,
        initial_structure_generation_maker_settings=InitialStructureGenerationMakerSettings(
            simulation_settings=InitialStructureGenerationSettings(random_seed=42)
        ),
        mlip_relax_maker_settings=MlipRelaxMakerSettings(
            simulation_settings=MlipRelaxSimulationSettings(
                optimizer_type=OptimizerType.BFGS,
                max_steps=1,
                force_threshold_in_ev_per_angstrom=0.01,
                model=model,
            )
        ),
        structure_augmentation_and_filtering_maker_settings=StructureAugmentationAndFilteringMakerSettings(
            simulation_settings=StructureAugmentationAndFilteringSettings(
                number_of_rattled_variants=(
                    LARGE_WORKFLOW_RATTLED_VARIANTS_PER_RELAXED_STRUCTURE
                ),
                number_of_stretched_variants=(
                    LARGE_WORKFLOW_STRETCHED_VARIANTS_PER_SOURCE
                ),
                number_of_stretched_variants_after_rattle=(
                    LARGE_WORKFLOW_STRETCHED_VARIANTS_PER_SOURCE
                ),
                rattle_settings=RattleAtomPositionsSettings(random_seed=42),
                stretch_after_rattle_settings=StretchCellSettings(random_seed=42),
                stretch_after_mlip_relax_settings=StretchCellSettings(random_seed=42),
            )
        ),
        direct_sampling_maker_settings=DirectSamplingMakerSettings(
            simulation_settings=DirectSamplingSettings(
                structure_encoder_model=model,
                birch_clustering_settings=BirchClusteringSettings(
                    number_of_clusters=LARGE_WORKFLOW_DIRECT_CLUSTER_COUNT
                ),
                cluster_sampling_settings=ClusterSamplingSettings(
                    num_samples_per_cluster=(
                        LARGE_WORKFLOW_DIRECT_SELECTIONS_PER_CLUSTER
                    )
                ),
            )
        ),
        structure_labelling_workflow_settings=StructureLabellingWorkflowSettings(
            single_point_maker_settings=(
                build_mlip_single_point_maker_settings_for_test()
            )
        ),
    )


@dataclass(kw_only=True)
class _DeterministicInitialStructureGenerationMaker(InitialStructureGenerationMaker):
    def _generate_structures(
        self, generation_range: StructureIndexRange
    ) -> list[PMG_Structure]:
        structures: list[PMG_Structure] = []
        for structure_offset in range(generation_range.number_of_structures):
            structure = deepcopy(TEST_LINBO3_STRUCTURE)
            structure.translate_sites(
                indices=[0],
                vector=[
                    0.0,
                    0.0,
                    0.01
                    * (structure_offset + 1)
                    / generation_range.number_of_structures,
                ],
                frac_coords=True,
            )
            structures.append(structure)
        return structures


@dataclass(kw_only=True)
class _SyntheticMlipRelaxBatchMaker(MlipRelaxBatchMaker):
    @job(**{MLIP_RELAX_TASK_DOCUMENT_BLOB_STORE_NAME: MlipRelaxTaskDocument})
    def _run(self, structures: list[PMG_Structure]) -> list[MlipRelaxTaskDocument]:
        return [
            MlipRelaxTaskDocument(
                properties=RelaxProperties(
                    relax_steps=[
                        SinglePointProperties(
                            step=0,
                            structure=structure,
                            forces=[[0.0, 0.0, 0.0] for _ in structure],
                            stress=[0.0] * 6,
                            energy=-100.0,
                        )
                    ]
                )
            )
            for structure in structures
        ]


@dataclass(kw_only=True)
class _SyntheticMlipSinglePointMaker(MlipSinglePointMaker):
    @job(output_schema=MlipSinglePointTaskDocument)
    def _run(self, structure: PMG_Structure) -> MlipSinglePointTaskDocument:
        return MlipSinglePointTaskDocument(
            properties=SinglePointProperties(
                step=0,
                structure=structure,
                forces=[[0.0, 0.0, 0.0] for _ in structure],
                stress=[0.0] * 6,
                energy=-100.0,
            )
        )


@dataclass(kw_only=True)
class _SyntheticStructureLabellingWorkflowMaker(StructureLabellingWorkflowMaker):
    @property
    def single_point_maker(self) -> _SyntheticMlipSinglePointMaker:
        settings = self.simulation_settings.single_point_maker_settings
        assert isinstance(settings, MlipSinglePointMakerSettings)
        return _SyntheticMlipSinglePointMaker(
            simulation_settings=settings.simulation_settings,
            executor_settings=settings.executor_settings,
        )


@dataclass(kw_only=True)
class _DeterministicMockDatasetGenerationWorkflowMaker(DatasetGenerationWorkflowMaker):
    @property
    def _initial_structure_generation_maker(
        self,
    ) -> _DeterministicInitialStructureGenerationMaker:
        settings = self.simulation_settings.initial_structure_generation_maker_settings
        return _DeterministicInitialStructureGenerationMaker(
            simulation_settings=settings.simulation_settings,
            executor_settings=settings.executor_settings,
        )

    @property
    def _mlip_relax_batch_maker(self) -> _SyntheticMlipRelaxBatchMaker:
        settings = self.simulation_settings.mlip_relax_maker_settings
        return _SyntheticMlipRelaxBatchMaker(
            simulation_settings=settings.simulation_settings,
            executor_settings=settings.executor_settings,
        )

    @property
    def _structure_labelling_workflow_maker(
        self,
    ) -> _SyntheticStructureLabellingWorkflowMaker:
        settings = self.simulation_settings.structure_labelling_workflow_settings
        assert settings is not None
        return _SyntheticStructureLabellingWorkflowMaker(simulation_settings=settings)


def _assert_job_stores_one_multichunk_structure_blob(
    *, structures_store: GridFSStore, job_uuid: str
) -> None:
    assert (
        structures_store._files_collection.count_documents(
            {"metadata.job_uuid": job_uuid}
        )
        == 1
    )
    [stored_blob_file] = list(
        structures_store._files_collection.find({"metadata.job_uuid": job_uuid})
    )
    assert (
        structures_store._chunks_collection.count_documents(
            {"files_id": stored_blob_file["_id"]}
        )
        > 1
    )


def _structure_blob_job_uuids_by_workflow_stage(
    stored_documents: list[dict[str, object]],
) -> dict[str, str]:
    stage_job_uuids: dict[str, str] = {}
    for document in stored_documents:
        output = document.get("output")
        if not isinstance(output, dict):
            continue
        properties = output.get("properties")
        if not isinstance(properties, dict):
            continue
        structures = properties.get("structures")
        if not (
            isinstance(structures, dict)
            and structures.get("store") == STRUCTURE_BLOB_STORE_NAME
        ):
            continue
        job_uuid = document.get("uuid")
        assert isinstance(job_uuid, str)
        if INITIAL_GENERATION_RANGE_PROPERTY_KEY in properties:
            stage_job_uuids["initial"] = job_uuid
        elif AUGMENTATION_STATISTICS_PROPERTY_KEY in properties:
            stage_job_uuids["augmentation"] = job_uuid
        elif FINAL_AUGMENTATION_STATISTICS_PROPERTY_KEY in output:
            stage_job_uuids["final"] = job_uuid
        else:
            stage_job_uuids["direct"] = job_uuid
    return stage_job_uuids


@pytest.mark.needs_mongodb
def test_final_workflow_result_resolves_persisted_selected_structures_downstream(
    mongo_uri: str, tmp_path: Path
) -> None:
    workflow_task_document_job = build_structure_generation_workflow_task_document(
        initial_structure_generation_task_document=(
            _completed_initial_structure_generation_task_document()
        ),
        mlip_relax_task_documents=[],
        direct_sampling_task_document=_completed_direct_sampling_task_document(),
        augmentation_and_filtering_task_document=_completed_augmentation_task_document(),
    )
    downstream_job = _count_downstream_structures(
        structures=workflow_task_document_job.output.properties.structures
    )

    with _temporary_mongo_jobstore(mongo_uri) as (jobstore, structures_store):
        run_locally(
            Flow([workflow_task_document_job, downstream_job]),
            root_dir=tmp_path,
            create_folders=True,
            store=jobstore,
        )

        _assert_job_stores_one_multichunk_structure_blob(
            structures_store=structures_store, job_uuid=workflow_task_document_job.uuid
        )

        loaded_output = MontyDecoder().process_decoded(
            jobstore.get_output(workflow_task_document_job.uuid, load=True)
        )
        assert isinstance(loaded_output, DatasetGenerationWorkflowTaskDocument)
        assert loaded_output.properties is not None
        assert isinstance(loaded_output.properties, StructureGenerationProperties)
        assert (
            len(loaded_output.properties.structures)
            == FINAL_WORKFLOW_RESULT_STRUCTURE_COUNT
        )

        downstream_output = jobstore.get_output(downstream_job.uuid, load=True)
        assert downstream_output == FINAL_WORKFLOW_RESULT_STRUCTURE_COUNT


@pytest.mark.slow
@pytest.mark.needs_mongodb
def test_large_workflow_preserves_ten_thousand_direct_selections_for_individual_labelling(
    mongo_uri: str, tmp_path: Path
) -> None:
    simulation_settings = _large_dataset_generation_workflow_settings()
    workflow = _DeterministicMockDatasetGenerationWorkflowMaker(
        simulation_settings=simulation_settings
    ).make()
    downstream_job = _count_downstream_structures(
        structures=workflow.output.properties.structures
    )

    with _temporary_mongo_jobstore(mongo_uri) as (jobstore, structures_store):
        run_result = run_locally(
            Flow([workflow, downstream_job]),
            root_dir=tmp_path,
            create_folders=True,
            store=jobstore,
            ensure_success=True,
        )

        loaded_workflow_output = MontyDecoder().process_decoded(
            jobstore.get_output(workflow.output.uuid, load=True)
        )
        assert isinstance(loaded_workflow_output, DatasetGenerationWorkflowTaskDocument)
        assert loaded_workflow_output.properties is not None
        assert isinstance(
            loaded_workflow_output.properties, StructureGenerationProperties
        )
        assert (
            len(loaded_workflow_output.properties.structures)
            == LARGE_WORKFLOW_EXPECTED_SELECTED_STRUCTURE_COUNT
        )
        assert jobstore.get_output(downstream_job.uuid, load=True) == (
            LARGE_WORKFLOW_EXPECTED_SELECTED_STRUCTURE_COUNT
        )

        label_task_documents = [
            response[1].output
            for response in run_result.values()
            if isinstance(response[1].output, MlipSinglePointTaskDocument)
        ]
        assert (
            len(label_task_documents)
            == LARGE_WORKFLOW_EXPECTED_SELECTED_STRUCTURE_COUNT
        )
        labelled_structure_fingerprints: Counter[str] = Counter()
        for label_task_document in label_task_documents:
            assert isinstance(label_task_document.properties, SinglePointProperties)
            labelled_structure_fingerprints[
                _structure_fingerprint(label_task_document.properties.structure)
            ] += 1
        assert labelled_structure_fingerprints == Counter(
            _structure_fingerprint(structure)
            for structure in loaded_workflow_output.properties.structures
        )

        augmentation_statistics = (
            loaded_workflow_output.augmentation_and_filtering_statistics
        )
        assert augmentation_statistics is not None
        for source_statistics in (
            augmentation_statistics.initial,
            augmentation_statistics.mlip_relax,
            augmentation_statistics.rattle_after_mlip_relax,
            augmentation_statistics.stretch_after_rattle,
            augmentation_statistics.stretch_after_mlip_relax,
        ):
            assert (
                source_statistics.considered == LARGE_WORKFLOW_INITIAL_STRUCTURE_COUNT
            )
            assert source_statistics.passed == LARGE_WORKFLOW_INITIAL_STRUCTURE_COUNT
            assert source_statistics.filtered == 0

        stage_job_uuids = _structure_blob_job_uuids_by_workflow_stage(
            list(jobstore.docs_store.query())
        )
        assert set(stage_job_uuids) == {"initial", "augmentation", "direct", "final"}
        for structure_blob_job_uuid in stage_job_uuids.values():
            _assert_job_stores_one_multichunk_structure_blob(
                structures_store=structures_store, job_uuid=structure_blob_job_uuid
            )
