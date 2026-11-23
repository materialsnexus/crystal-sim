from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path

import pytest
from jobflow.managers.local import run_locally

from crystal.calculators.gpaw.relax_maker import GpawRelaxMaker
from crystal.constants.gpaw import (
    GPAW_EXTERNAL_RELAX_SCRIPT_FILENAME,
    GPAW_SCF_INPUT_STRUCTURE_FILENAME,
    GPAW_SCF_KWARGS_FILENAME,
    GPAW_SCF_OUTPUT_FILENAME,
)
from crystal.schemas.gpaw import (
    GpawRelaxSimulationSettings,
    GpawRelaxTaskDocument,
    GpawTaskStatus,
)
from crystal.schemas.relax_settings import OptimizerType
from tests.helpers.constants.structure import (
    TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
    TEST_SI_STRUCTURE,
)
from tests.helpers.external_gpaw_runtime import (
    assert_external_gpaw_runtime_configuration_is_present,
)

TEST_GPAW_RELAX_MAX_STEPS = 1
TEST_GPAW_RELAX_MIN_STEP_COUNT = 1
EXPECTED_GPAW_RELAX_RUN_FILENAMES = (
    GPAW_EXTERNAL_RELAX_SCRIPT_FILENAME,
    GPAW_SCF_INPUT_STRUCTURE_FILENAME,
    GPAW_SCF_KWARGS_FILENAME,
    GPAW_SCF_OUTPUT_FILENAME,
)


@pytest.mark.needs_third_party_executables
def test_run_gpaw_relax_maker_with_external_runtime(tmp_path: Path) -> None:
    assert_external_gpaw_runtime_configuration_is_present()
    maker = GpawRelaxMaker(
        simulation_settings=GpawRelaxSimulationSettings(
            optimizer_type=OptimizerType.BFGS,
            max_steps=TEST_GPAW_RELAX_MAX_STEPS,
            force_threshold_in_ev_per_angstrom=TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
        )
    )
    job = maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)
    task_document = run_job[job.uuid][1].output
    assert isinstance(task_document, GpawRelaxTaskDocument)
    assert task_document.status is GpawTaskStatus.success
    assert task_document.converged
    assert task_document.errors == ""
    assert task_document.properties is not None
    assert len(task_document.properties.relax_steps) >= TEST_GPAW_RELAX_MIN_STEP_COUNT
    assert isfinite(task_document.properties.final_energy)
    assert task_document.properties.final_energy == (
        task_document.properties.relax_steps[-1].energy
    )
    assert len(task_document.properties.relax_steps[-1].forces) == len(
        TEST_SI_STRUCTURE
    )
    assert task_document.properties.relaxed_structure.composition == (
        TEST_SI_STRUCTURE.composition
    )

    run_directory = Path(task_document.dir_name).resolve()
    assert run_directory.is_relative_to(tmp_path.resolve())
    assert run_directory.is_dir()
    for filename in EXPECTED_GPAW_RELAX_RUN_FILENAMES:
        assert (run_directory / filename).is_file()
