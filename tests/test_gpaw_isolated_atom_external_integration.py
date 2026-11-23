"""Integration tests for isolated-atom GPAW calculations in the external runtime."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from shutil import which

import pytest
from fireworks import Workflow
from fireworks.core.firework import Firework
from fireworks.core.fworker import FWorker
from fireworks.core.launchpad import LaunchPad
from jobflow.core.store import JobStore
from jobflow.managers.fireworks import flow_to_workflow
from monty.json import MontyDecoder

from crystal.calculators.gpaw.isolated_atom_maker import GpawIsolatedAtomMaker
from crystal.calculators.gpaw.isolated_atom_runner import run_gpaw_isolated_atom
from crystal.constants.gpaw import (
    GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_FILENAME,
    GPAW_ISOLATED_ATOM_INPUT_STRUCTURE_FILENAME,
    GPAW_ISOLATED_ATOM_KWARGS_FILENAME,
    GPAW_ISOLATED_ATOM_OUTPUT_FILENAME,
    GPAW_RUNTIME_PROVENANCE_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
)
from crystal.firework_helpers import run_rapidfire_launcher
from crystal.schemas.gpaw import GpawEigensolverName, GpawIsolatedAtomSimulationSettings
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawIsolatedAtomReferenceCollection,
)
from tests.helpers.external_gpaw_runtime import (
    assert_external_gpaw_runtime_configuration_is_present,
)

TEST_GPAW_EXTERNAL_PROCESS_COUNT = 1
TEST_GPAW_EXTERNAL_SILICON_ELEMENT = "Si"
TEST_GPAW_EXTERNAL_PAW_SETUP_SHA256_DIGEST_LENGTH = 64
EXPECTED_GPAW_ISOLATED_ATOM_ARTIFACT_FILENAMES = (
    GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_FILENAME,
    GPAW_ISOLATED_ATOM_INPUT_STRUCTURE_FILENAME,
    GPAW_ISOLATED_ATOM_KWARGS_FILENAME,
    GPAW_ISOLATED_ATOM_OUTPUT_FILENAME,
    GPAW_RUNTIME_PROVENANCE_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
)


@pytest.mark.needs_third_party_executables
@pytest.mark.slow
def test_run_gpaw_isolated_atom_with_external_runtime(tmp_path: Path) -> None:
    assert_external_gpaw_runtime_configuration_is_present()
    if which("mpirun") is None:
        pytest.skip("The configured external GPAW runtime requires mpirun.")

    isolated_atom_result = run_gpaw_isolated_atom(
        element=TEST_GPAW_EXTERNAL_SILICON_ELEMENT,
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
        run_directory=tmp_path,
        number_of_processes=TEST_GPAW_EXTERNAL_PROCESS_COUNT,
    )

    assert isolated_atom_result.element == TEST_GPAW_EXTERNAL_SILICON_ELEMENT
    assert isfinite(isolated_atom_result.energy)
    assert isfinite(isolated_atom_result.magnetic_moment)
    assert isolated_atom_result.runtime_provenance.gpaw_version
    assert isolated_atom_result.runtime_provenance.elements == frozenset(
        {TEST_GPAW_EXTERNAL_SILICON_ELEMENT}
    )
    assert (
        len(
            isolated_atom_result.runtime_provenance.atomic_data_by_element[
                TEST_GPAW_EXTERNAL_SILICON_ELEMENT
            ].paw_setup_sha256_digest
        )
        == TEST_GPAW_EXTERNAL_PAW_SETUP_SHA256_DIGEST_LENGTH
    )
    for artifact_filename in EXPECTED_GPAW_ISOLATED_ATOM_ARTIFACT_FILENAMES:
        assert (tmp_path / artifact_filename).is_file()


@pytest.mark.needs_third_party_executables
@pytest.mark.slow
def test_run_gpaw_isolated_atom_recovers_lithium_with_ppcg(tmp_path: Path) -> None:
    assert_external_gpaw_runtime_configuration_is_present()
    if which("mpirun") is None:
        pytest.skip("The configured external GPAW runtime requires mpirun.")

    isolated_atom_result = run_gpaw_isolated_atom(
        element="Li",
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
        run_directory=tmp_path,
        number_of_processes=TEST_GPAW_EXTERNAL_PROCESS_COUNT,
    )

    assert isolated_atom_result.element == "Li"
    assert isolated_atom_result.eigensolver is GpawEigensolverName.ppcg
    assert isfinite(isolated_atom_result.energy)
    assert isfinite(isolated_atom_result.magnetic_moment)


@pytest.mark.needs_mongodb
@pytest.mark.needs_third_party_executables
@pytest.mark.slow
def test_isolated_atom_flow_roundtrips_through_fireworks_and_jobstore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_launchpad: LaunchPad,
    mock_jobstore: JobStore,
) -> None:
    assert_external_gpaw_runtime_configuration_is_present()
    if which("mpirun") is None:
        pytest.skip("The configured external GPAW runtime requires mpirun.")
    monkeypatch.chdir(tmp_path)

    flow = GpawIsolatedAtomMaker(
        simulation_settings=GpawIsolatedAtomSimulationSettings()
    ).make(elements=frozenset({"Li"}))
    workflow = flow_to_workflow(flow, store=mock_jobstore)
    roundtripped_workflow = Workflow.from_dict(workflow.to_dict())

    mock_launchpad.add_wf(roundtripped_workflow)
    fireworker = FWorker(name="isolated-atom-test", query={})
    run_rapidfire_launcher(mock_launchpad, fireworker)

    firework_ids = mock_launchpad.get_fw_ids()
    collector_firework_id = next(
        firework_id
        for firework_id in firework_ids
        if _firework_job_uuid(mock_launchpad.get_fw_by_id(firework_id))
        == flow.output.uuid
    )
    stored_workflow = mock_launchpad.get_wf_by_fw_id(collector_firework_id)
    assert all(state == "COMPLETED" for state in stored_workflow.fw_states.values())

    stored_output = mock_jobstore.get_output(flow.output.uuid, load=True)
    collection = MontyDecoder().process_decoded(stored_output)
    assert isinstance(collection, GpawIsolatedAtomReferenceCollection)
    assert collection.is_success
    assert collection.reference is not None
    assert set(collection.reference.energies_by_element) == {"Li"}
    assert collection.task_documents[0].eigensolver is GpawEigensolverName.ppcg


def _firework_job_uuid(firework: Firework) -> str:
    job = firework.spec["_tasks"][0]["job"]
    job_uuid: object = job["uuid"] if isinstance(job, dict) else job.uuid
    if not isinstance(job_uuid, str):
        msg = "Firework job UUID must be a string."
        raise TypeError(msg)
    return job_uuid
