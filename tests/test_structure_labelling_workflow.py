from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from jobflow.core.flow import Flow
from jobflow.core.job import Job, job
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import MemoryStore
from monty.json import MontyDecoder
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.base import BaseMaker
from crystal.schemas.mlip import MlipSinglePointTaskDocument
from crystal.schemas.properties import SinglePointProperties
from crystal.schemas.settings import SimulationSettings
from crystal.schemas.structure_labelling import StructureLabellingWorkflowSettings
from crystal.structure.labelling.structure_labelling_workflow import (
    fan_out_single_points,
)
from crystal.structure.labelling.structure_labelling_workflow_maker import (
    StructureLabellingWorkflowMaker,
)
from tests.helpers.builders.properties import build_single_point_properties_for_test
from tests.helpers.builders.single_point_maker_settings import (
    build_mlip_single_point_maker_settings_for_test,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE


@job(output_schema=MlipSinglePointTaskDocument)
def _produce_controlled_single_point_task_document(
    structure: PMG_Structure, *, failed: bool = False
) -> MlipSinglePointTaskDocument:
    if failed:
        return MlipSinglePointTaskDocument(properties=None, errors="mock label failure")

    return MlipSinglePointTaskDocument(
        properties=build_single_point_properties_for_test(structure, energy=0.0),
        errors="",
    )


@job
def _produce_structures(structures: list[PMG_Structure]) -> list[PMG_Structure]:
    """Produce structures to exercise upstream OutputReference resolution."""
    return structures


@dataclass(kw_only=True)
class _SinglePointMakerWithOptionalFailure(BaseMaker):
    simulation_settings: SimulationSettings = field(default_factory=SimulationSettings)
    failure_structure: PMG_Structure | None = None

    def make(self, structure: PMG_Structure) -> Job:
        return _produce_controlled_single_point_task_document(
            structure=structure,
            failed=(
                self.failure_structure is not None
                and structure == self.failure_structure
            ),
        )


def _distinct_source_structures() -> list[PMG_Structure]:
    first = deepcopy(TEST_SI_STRUCTURE)
    second = deepcopy(TEST_SI_STRUCTURE)
    second.translate_sites(0, [0.25, 0.0, 0.0])
    return [first, second]


def _run_structure_labelling_flow(
    flow: Flow, root_dir: Path, jobstore: JobStore
) -> list[MlipSinglePointTaskDocument]:
    run_result = run_locally(
        flow, root_dir=root_dir, create_folders=True, store=jobstore
    )
    outputs = [response[1].output for response in run_result.values()]
    return [
        output for output in outputs if isinstance(output, MlipSinglePointTaskDocument)
    ]


@run_in_temp_dir
def test_fanout_runs_one_label_job_per_source_structure_without_gathering_outputs() -> (
    None
):
    structures = _distinct_source_structures()
    fanout_job = fan_out_single_points(
        source_structures=structures,
        single_point_maker=_SinglePointMakerWithOptionalFailure(),
    )

    jobstore = JobStore(docs_store=MemoryStore())
    jobstore.connect()

    run_locally(fanout_job, root_dir=Path.cwd(), create_folders=True, store=jobstore)

    stored_fanout_run = jobstore.query_one(
        criteria={"uuid": fanout_job.uuid}, properties=["output"], sort={"index": -1}
    )
    assert stored_fanout_run is not None

    assert MontyDecoder().process_decoded(stored_fanout_run["output"]) is None

    label_documents = [
        MontyDecoder().process_decoded(task_document["output"])
        for task_document in jobstore.query(properties=["output"])
    ]
    completed_labels = [
        task_document
        for task_document in label_documents
        if isinstance(task_document, MlipSinglePointTaskDocument)
    ]

    labelled_structures = [
        task_document.properties.structure
        for task_document in completed_labels
        if task_document.properties is not None
    ]
    assert len(labelled_structures) == len(structures)
    assert all(structure in labelled_structures for structure in structures)


@run_in_temp_dir
def test_fanout_preserves_successful_and_failed_label_documents() -> None:
    structures = _distinct_source_structures()
    fanout_job = fan_out_single_points(
        source_structures=structures,
        single_point_maker=_SinglePointMakerWithOptionalFailure(
            failure_structure=structures[1]
        ),
    )

    jobstore = JobStore(docs_store=MemoryStore())
    jobstore.connect()
    child_documents = _run_structure_labelling_flow(
        Flow([fanout_job]), Path.cwd(), jobstore
    )

    assert len(child_documents) == len(structures)

    [successful_document] = [
        document for document in child_documents if document.properties is not None
    ]
    [failed_document] = [
        document for document in child_documents if document.properties is None
    ]
    assert successful_document.properties is not None
    assert successful_document.properties.structure == structures[0]
    assert successful_document.errors == ""
    assert failed_document.errors == "mock label failure"


@run_in_temp_dir
def test_structure_labelling_creates_label_documents_from_upstream_structures() -> None:
    structures = _distinct_source_structures()
    mlip_settings = build_mlip_single_point_maker_settings_for_test()
    workflow_settings = StructureLabellingWorkflowSettings(
        single_point_maker_settings=mlip_settings
    )

    def run_mlip_single_point(
        *, structure: PMG_Structure, **_kwargs: object
    ) -> SinglePointProperties:
        return SinglePointProperties(
            step=0,
            structure=structure,
            forces=[[0.0, 0.0, 0.0] for _ in structure],
            stress=[0.0] * 6,
            energy=-1.25,
        )

    with (
        patch(
            "crystal.calculators.mlip.single_point_maker.run_mlip_single_point",
            side_effect=run_mlip_single_point,
        ),
        patch(
            "crystal.calculators.mlip.single_point_maker.configure_default_torch_device",
            return_value=None,
        ),
    ):
        source_job = _produce_structures(structures)
        structure_labelling_maker = StructureLabellingWorkflowMaker(
            simulation_settings=workflow_settings
        )
        fanout_job = fan_out_single_points(
            source_structures=source_job.output,
            single_point_maker=structure_labelling_maker.single_point_maker,
        )
        workflow = Flow([source_job, fanout_job], output=None)
        jobstore = JobStore(docs_store=MemoryStore())
        jobstore.connect()

        child_documents = _run_structure_labelling_flow(workflow, Path.cwd(), jobstore)

    assert len(child_documents) == len(structures)

    assert all(
        document.properties is not None and document.properties.energy == -1.25
        for document in child_documents
    )
    assert all(document.errors == "" for document in child_documents)
    labelled_structures = [
        document.properties.structure
        for document in child_documents
        if document.properties is not None
    ]
    assert len(labelled_structures) == len(structures)
    assert all(structure in labelled_structures for structure in structures)


@run_in_temp_dir
def test_structure_labelling_creates_no_label_jobs_for_empty_source() -> None:
    workflow_settings = StructureLabellingWorkflowSettings(
        single_point_maker_settings=(build_mlip_single_point_maker_settings_for_test())
    )
    workflow = StructureLabellingWorkflowMaker(
        simulation_settings=workflow_settings
    ).make(source_structures=[])

    [fanout_job] = workflow.jobs
    assert isinstance(fanout_job, Job)
    jobstore = JobStore(docs_store=MemoryStore())
    jobstore.connect()
    child_documents = _run_structure_labelling_flow(workflow, Path.cwd(), jobstore)

    assert jobstore.get_output(fanout_job.uuid, load=True) == []
    assert child_documents == []
