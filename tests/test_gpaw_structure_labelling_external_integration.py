"""End-to-end GPAW and Chargemol structure-labelling integration coverage."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from jobflow.core.flow import Flow
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import MemoryStore
from monty.json import MontyDecoder

from crystal.schemas.gpaw import GpawScfChargemolTaskDocument
from crystal.schemas.settings import ExecutorSettings
from crystal.schemas.structure_labelling import StructureLabellingWorkflowSettings
from crystal.structure.labelling.structure_labelling_workflow_maker import (
    StructureLabellingWorkflowMaker,
)
from tests.helpers.builders.single_point_maker_settings import (
    build_gpaw_single_point_maker_settings_for_test,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.gpaw_chargemol_integration import (
    assert_external_gpaw_chargemol_configuration_is_present,
    assert_successful_silicon_gpaw_chargemol_labelling,
)

TEST_GPAW_EXTERNAL_PROCESS_COUNT = 1


def _run_local_structure_labelling_workflow(
    workflow: Flow, root_directory: Path
) -> list[GpawScfChargemolTaskDocument]:
    jobstore = JobStore(docs_store=MemoryStore())
    jobstore.connect()

    run_locally(workflow, root_dir=root_directory, create_folders=True, store=jobstore)

    output = jobstore.get_output(workflow.output.uuid, load=True)
    decoded_output = MontyDecoder().process_decoded(output)
    assert isinstance(decoded_output, list)
    assert all(
        isinstance(document, GpawScfChargemolTaskDocument)
        for document in decoded_output
    )
    return decoded_output


@pytest.mark.needs_third_party_executables
@pytest.mark.slow
def test_gpaw_structure_labelling_workflow_runs_real_gpaw_and_chargemol(
    tmp_path: Path,
) -> None:
    assert_external_gpaw_chargemol_configuration_is_present()
    workflow_settings = StructureLabellingWorkflowSettings(
        single_point_maker_settings=build_gpaw_single_point_maker_settings_for_test(
            executor_settings=ExecutorSettings(
                number_of_processes=TEST_GPAW_EXTERNAL_PROCESS_COUNT
            )
        )
    )
    workflow = StructureLabellingWorkflowMaker(
        simulation_settings=workflow_settings
    ).make(source_structures=[deepcopy(TEST_SI_STRUCTURE)])
    output_documents = _run_local_structure_labelling_workflow(workflow, tmp_path)

    assert len(output_documents) == 1
    output_document = output_documents[0]
    assert_successful_silicon_gpaw_chargemol_labelling(
        task_document=output_document, source_structure=TEST_SI_STRUCTURE
    )
