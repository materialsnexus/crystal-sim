"""SLURM integration coverage for the configured external GPAW runtime.

The test uses the repository's ``SLURM`` FireWorks profile and derives a
unique temporary Mongo database from it. Configure the SLURM profile templates
and external program paths in ignored ``crystal.toml``, including
``programs.mpi_environment_setup_script``. Then run:

    uv run pytest -q tests/test_gpaw_slurm_external_runtime_integration.py
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from monty.json import MontyDecoder

from crystal.calculators.gpaw.slurm_runtime_precheck import (
    run_external_gpaw_mpi_precheck,
)
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
from tests.helpers.slurm_runtime import execute_slurm_workflow

TEST_GPAW_SLURM_PROCESS_COUNT = 1
TEST_GPAW_SLURM_MEMORY_GB = 16
TEST_GPAW_SLURM_WALLTIME_MINUTES = 30


@pytest.mark.needs_mongodb
@pytest.mark.needs_slurm
@pytest.mark.needs_third_party_executables
@pytest.mark.slow
def test_slurm_job_loads_the_configured_external_gpaw_runtime() -> None:
    precheck_job = run_external_gpaw_mpi_precheck(
        number_of_processes=TEST_GPAW_SLURM_PROCESS_COUNT
    )
    with execute_slurm_workflow(precheck_job) as execution:
        assert execution.workflow.fw_states[execution.anchor_firework_id] == "COMPLETED"
        output = execution.manager.jobstore.get_output(precheck_job.uuid)
        assert isinstance(output, dict)
        assert output["slurm_job_id"]
        assert output["gpaw_runtime"].endswith(f":{TEST_GPAW_SLURM_PROCESS_COUNT}")


@pytest.mark.needs_mongodb
@pytest.mark.needs_slurm
@pytest.mark.needs_third_party_executables
@pytest.mark.slow
def test_slurm_workflow_runs_real_gpaw_and_chargemol() -> None:
    assert_external_gpaw_chargemol_configuration_is_present()
    slurm_executor_settings = ExecutorSettings(
        number_of_processes=TEST_GPAW_SLURM_PROCESS_COUNT,
        total_memory_in_gb=TEST_GPAW_SLURM_MEMORY_GB,
        maximum_walltime_in_minutes=TEST_GPAW_SLURM_WALLTIME_MINUTES,
    )
    workflow_settings = StructureLabellingWorkflowSettings(
        single_point_maker_settings=build_gpaw_single_point_maker_settings_for_test(
            executor_settings=slurm_executor_settings
        )
    )
    workflow = StructureLabellingWorkflowMaker(
        simulation_settings=workflow_settings, executor_settings=slurm_executor_settings
    ).make(source_structures=[deepcopy(TEST_SI_STRUCTURE)])

    with execute_slurm_workflow(workflow) as execution:
        assert all(
            state == "COMPLETED" for state in execution.workflow.fw_states.values()
        )
        output = execution.manager.jobstore.get_output(workflow.output.uuid, load=True)
        decoded_output = MontyDecoder().process_decoded(output)
        assert isinstance(decoded_output, list)
        assert len(decoded_output) == 1
        task_document = decoded_output[0]
        assert isinstance(task_document, GpawScfChargemolTaskDocument)
        assert_successful_silicon_gpaw_chargemol_labelling(
            task_document=task_document, source_structure=TEST_SI_STRUCTURE
        )
