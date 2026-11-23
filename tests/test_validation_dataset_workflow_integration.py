"""Integration coverage for dynamic validation-dataset workflow execution."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from jobflow.core.flow import Flow
from jobflow.core.job import Job
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.core.store import Store
from maggma.stores import MemoryStore, MongoStore
from pymongo.errors import ServerSelectionTimeoutError

from crystal.calculators.md.md_runner import MlipMdRunner
from crystal.constants.fireworks import QUEUEADAPTER_NTASKS_KEY, QUEUEADAPTER_SPEC_KEY
from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.constants.validation_dataset import (
    VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME,
)
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.md.mlip_md_equilibration import MlipMdEquilibrationMakerSettings
from crystal.schemas.md.properties import MdStageResult
from crystal.schemas.md.settings import MdEnsemble
from crystal.schemas.mlip import MlipRelaxMakerSettings
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.settings import ExecutorSettings
from crystal.schemas.structure_generators import (
    DirectSamplingMakerSettings,
    DirectSamplingSettings,
)
from crystal.schemas.validation_dataset import (
    ValidationDatasetArtifactReference,
    ValidationDatasetBulkStructureInput,
    ValidationDatasetDirectSelectionSettings,
    ValidationDatasetMdRunKind,
    ValidationDatasetMdSamplingSettings,
    ValidationDatasetStructureBatchJobReference,
    ValidationDatasetStructureBatchReference,
    ValidationDatasetWorkflowInputs,
    ValidationDatasetWorkflowSettings,
)
from crystal.structure.generators.sampling.direct_maker import DirectSamplingMaker
from crystal.structure.validation_dataset.workflow_maker import (
    ValidationDatasetWorkflowMaker,
    build_validation_dataset_md_and_selection_flow,
    fanout_validation_dataset_md_and_selection,
)
from tests.helpers.constants.mlip import TEST_MLIP_RELAX_SIMULATION_SETTINGS
from tests.helpers.constants.structure import TEST_LINBO3_STRUCTURE, TEST_SI_STRUCTURE
from tests.helpers.validation_dataset import (
    build_validation_dataset_md_simulation_settings_for_test,
    write_validation_dataset_md_trajectory_for_test,
)


class _FirstStructureSampler:
    def fit_transform(self, structures: list[object]) -> dict[str, list[int]]:
        return {"selected_indexes": [0]} if structures else {"selected_indexes": []}


def _build_jobstore() -> JobStore:
    jobstore = JobStore(
        docs_store=MemoryStore(),
        additional_stores={
            STRUCTURE_BLOB_STORE_NAME: MemoryStore(),
            VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME: MemoryStore(),
        },
    )
    jobstore.connect()
    return jobstore


def _get_blob_store(jobstore: JobStore, store_name: str) -> Store:
    return jobstore.additional_stores[store_name]  # type: ignore[has-type, no-any-return]


def _jobs_in_flow(flow: Flow) -> list[Job]:
    jobs: list[Job] = []
    for flow_item in flow.jobs:
        if isinstance(flow_item, Job):
            jobs.append(flow_item)
        else:
            jobs.extend(_jobs_in_flow(flow_item))
    return jobs


@contextmanager
def _temporary_mongo_jobstore(mongo_uri: str) -> Generator[JobStore, None, None]:
    collection_suffix = uuid4().hex
    stores: dict[str, Store] = {
        store_name: MongoStore(
            database="validation_dataset_tests",
            collection_name=f"{store_name}_{collection_suffix}",
            host=mongo_uri,
        )
        for store_name in (
            STRUCTURE_BLOB_STORE_NAME,
            VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME,
        )
    }
    docs_store = MongoStore(
        database="validation_dataset_tests",
        collection_name=f"jobs_{collection_suffix}",
        host=mongo_uri,
    )
    jobstore = JobStore(docs_store=docs_store, additional_stores=stores)
    jobstore.connect()
    try:
        yield jobstore
    finally:
        try:
            docs_store.remove_docs({})
            for store in stores.values():
                store.remove_docs({})
        except ServerSelectionTimeoutError:
            pass


def _add_preparation_batch(jobstore: JobStore) -> None:
    structures_store = _get_blob_store(jobstore, STRUCTURE_BLOB_STORE_NAME)
    structures_store.update(
        {"blob_uuid": "slab-batch-0", "data": [TEST_SI_STRUCTURE.as_dict()]}
    )
    jobstore.docs_store.update(
        {
            "uuid": "slab-preparation-job",
            "index": 1,
            "output": {
                "properties": {
                    "slabs": {
                        "store": STRUCTURE_BLOB_STORE_NAME,
                        "blob_uuid": "slab-batch-0",
                    },
                    "slab_count": 1,
                }
            },
        }
    )


def _fake_md_run(runner: MlipMdRunner) -> MdStageResult:
    trajectory_path = runner._trajectory_path(ensemble=MdEnsemble.nvt)
    write_validation_dataset_md_trajectory_for_test(trajectory_path)
    return MdStageResult(
        ensemble=MdEnsemble.nvt,
        trajectory_path=trajectory_path,
        default_log_path=trajectory_path,
        user_defined_log_path=trajectory_path,
    )


def _failed_md_run(runner: MlipMdRunner) -> MdStageResult:
    del runner
    message = "synthetic MLIP-MD failure"
    raise RuntimeError(message)


def _build_fanout_job(
    *, temperatures_k: tuple[float, ...] = (300.0,), maximum_md_runs: int = 1
) -> Job:
    return fanout_validation_dataset_md_and_selection(
        slab_batch_job_references=[
            ValidationDatasetStructureBatchJobReference(
                batch_id="slab-0",
                source_kind=ValidationDatasetMdRunKind.slab,
                job_uuid="slab-preparation-job",
            )
        ],
        interface_batch_job_references=[],
        md_sampling_settings=ValidationDatasetMdSamplingSettings(
            temperatures_k=temperatures_k, maximum_md_runs=maximum_md_runs
        ),
        md_maker_settings=MlipMdEquilibrationMakerSettings(
            simulation_settings=build_validation_dataset_md_simulation_settings_for_test()
        ),
        direct_sampling_maker_settings=DirectSamplingMakerSettings(
            simulation_settings=DirectSamplingSettings(
                structure_encoder_model=MlipModelRef(
                    registry_model=QetModelRegistry.foundational_matpes_pbe
                )
            )
        ),
        direct_selection_settings=ValidationDatasetDirectSelectionSettings(
            target_number_of_structures=1,
            maximum_frames_per_md_run=2,
            maximum_frames_in_selection_pool=2,
        ),
    )


@run_in_temp_dir
def test_dynamic_validation_dataset_fanout_reuses_structure_loads_per_source() -> None:
    jobstore = _build_jobstore()
    _add_preparation_batch(jobstore)
    fanout_job = _build_fanout_job(temperatures_k=(300.0, 600.0), maximum_md_runs=2)

    with (
        patch.object(MlipMdRunner, "run", _fake_md_run),
        patch.object(
            DirectSamplingMaker,
            "_build_direct_sampler",
            return_value=_FirstStructureSampler(),
        ),
    ):
        run_locally(
            fanout_job,
            root_dir=Path.cwd(),
            create_folders=True,
            store=jobstore,
            raise_immediately=True,
        )

    assert (
        len(
            list(
                jobstore.docs_store.query({"name": "load_validation_dataset_structure"})
            )
        )
        == 1
    )
    assert (
        len(list(jobstore.docs_store.query({"name": "run_validation_dataset_md"}))) == 2
    )


@run_in_temp_dir
def test_dynamic_validation_dataset_fanout_keeps_failed_md_out_of_selection() -> None:
    jobstore = _build_jobstore()
    _add_preparation_batch(jobstore)
    fanout_job = _build_fanout_job()

    with patch.object(MlipMdRunner, "run", _failed_md_run):
        run_locally(
            fanout_job,
            root_dir=Path.cwd(),
            create_folders=True,
            store=jobstore,
            raise_immediately=True,
        )

    md_documents = list(
        jobstore.docs_store.query({"name": "run_validation_dataset_md"})
    )
    assert len(md_documents) == 1
    assert md_documents[0]["output"]["errors"]
    assert not list(
        _get_blob_store(jobstore, VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME).query(
            {}
        )
    )
    selected_documents = list(
        jobstore.docs_store.query({"name": "cap_direct_selection"})
    )
    assert len(selected_documents) == 1


@pytest.mark.needs_mongodb
@run_in_temp_dir
def test_dynamic_validation_dataset_fanout_persists_outputs_in_mongodb(
    mongo_uri: str,
) -> None:
    with _temporary_mongo_jobstore(mongo_uri) as jobstore:
        _add_preparation_batch(jobstore)
        fanout_job = _build_fanout_job()

        with (
            patch.object(MlipMdRunner, "run", _fake_md_run),
            patch.object(
                DirectSamplingMaker,
                "_build_direct_sampler",
                return_value=_FirstStructureSampler(),
            ),
        ):
            run_locally(
                fanout_job,
                root_dir=Path.cwd(),
                create_folders=True,
                store=jobstore,
                raise_immediately=True,
            )

        md_documents = list(
            jobstore.docs_store.query({"name": "run_validation_dataset_md"})
        )
        assert len(md_documents) == 1
        trajectory_documents = list(
            _get_blob_store(
                jobstore, VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME
            ).query({})
        )
        assert len(trajectory_documents) == 1
        assert isinstance(trajectory_documents[0]["data"], bytes)
        selected_documents = list(
            jobstore.docs_store.query({"name": "cap_direct_selection"})
        )
        assert len(selected_documents) == 1
        selected_output = selected_documents[0]["output"]
        assert selected_output["structures"]["store"] == STRUCTURE_BLOB_STORE_NAME


def test_validation_dataset_workflow_uses_nested_preparation_maker_settings() -> None:
    preparation_executor_settings = ExecutorSettings(number_of_processes=2)
    settings = ValidationDatasetWorkflowSettings(
        mlip_relax_maker_settings=MlipRelaxMakerSettings(
            simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS,
            executor_settings=preparation_executor_settings,
        ),
        md_equilibration_maker_settings=MlipMdEquilibrationMakerSettings(
            simulation_settings=build_validation_dataset_md_simulation_settings_for_test()
        ),
        direct_sampling_maker_settings=DirectSamplingMakerSettings(
            simulation_settings=DirectSamplingSettings(
                structure_encoder_model=MlipModelRef(
                    registry_model=QetModelRegistry.foundational_matpes_pbe
                )
            )
        ),
        slab_generation={"miller_indices": ((0, 0, 1),)},
        md_sampling={"temperatures_k": (300.0,), "maximum_md_runs": 1},
    )
    maker = ValidationDatasetWorkflowMaker(simulation_settings=settings)

    validation_flow = maker.make(
        inputs=ValidationDatasetWorkflowInputs(
            bulk_structures=(
                ValidationDatasetBulkStructureInput(
                    structure_id="linbo3", structure=TEST_LINBO3_STRUCTURE
                ),
                ValidationDatasetBulkStructureInput(
                    structure_id="sio2", structure=TEST_SI_STRUCTURE
                ),
            )
        )
    )

    preparation_flow = validation_flow.jobs[0]
    assert isinstance(preparation_flow, Flow)
    preparation_job = preparation_flow.jobs[0]
    assert isinstance(preparation_job, Job)
    assert preparation_job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 2


def test_validation_dataset_workflow_rejects_nve_temperature_sampling() -> None:
    md_simulation_settings = build_validation_dataset_md_simulation_settings_for_test()
    nve_dynamics = md_simulation_settings.run_settings.dynamics.model_copy(
        update={"ensemble": MdEnsemble.nve}
    )
    nve_md_simulation_settings = md_simulation_settings.model_copy(
        update={
            "run_settings": md_simulation_settings.run_settings.model_copy(
                update={"dynamics": nve_dynamics}
            )
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Validation-dataset temperature sampling requires an NVT or NPT ensemble\.",
    ):
        ValidationDatasetWorkflowSettings(
            mlip_relax_maker_settings=MlipRelaxMakerSettings(
                simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS
            ),
            md_equilibration_maker_settings=MlipMdEquilibrationMakerSettings(
                simulation_settings=nve_md_simulation_settings
            ),
            direct_sampling_maker_settings=DirectSamplingMakerSettings(
                simulation_settings=DirectSamplingSettings(
                    structure_encoder_model=MlipModelRef(
                        registry_model=QetModelRegistry.foundational_matpes_pbe
                    )
                )
            ),
        )


def test_validation_dataset_dynamic_flow_uses_md_and_selection_executor_settings() -> (
    None
):
    md_executor_settings = ExecutorSettings(number_of_processes=3)
    selection_executor_settings = ExecutorSettings(number_of_processes=4)
    response = build_validation_dataset_md_and_selection_flow(
        slab_sources=[
            ValidationDatasetStructureBatchReference(
                batch_id="slab-0",
                structure_artifact=ValidationDatasetArtifactReference(
                    artifact_id="slabs-0", store_name=STRUCTURE_BLOB_STORE_NAME
                ),
                item_count=1,
            )
        ],
        interface_sources=[],
        md_sampling_settings=ValidationDatasetMdSamplingSettings(
            temperatures_k=(300.0,)
        ),
        md_maker_settings=MlipMdEquilibrationMakerSettings(
            simulation_settings=build_validation_dataset_md_simulation_settings_for_test(),
            executor_settings=md_executor_settings,
        ),
        direct_sampling_maker_settings=DirectSamplingMakerSettings(
            simulation_settings=DirectSamplingSettings(
                structure_encoder_model=MlipModelRef(
                    registry_model=QetModelRegistry.foundational_matpes_pbe
                )
            ),
            executor_settings=selection_executor_settings,
        ),
        direct_selection_settings=ValidationDatasetDirectSelectionSettings(
            target_number_of_structures=1,
            maximum_frames_per_md_run=1,
            maximum_frames_in_selection_pool=1,
        ),
    )

    dynamic_flow = response.replace
    assert isinstance(dynamic_flow, Flow)
    md_job = next(
        job
        for job in _jobs_in_flow(dynamic_flow)
        if job.name == "run_validation_dataset_md"
    )
    direct_job = next(
        job for job in _jobs_in_flow(dynamic_flow) if job.name == "direct_sampling"
    )

    assert md_job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 3
    assert direct_job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 4
