"""Behavioral integration tests for registry-backed GPAW QET fine-tuning."""

from __future__ import annotations

from pathlib import Path

import matgl
import numpy as np
import pytest
import torch
from jobflow.core.flow import Flow
from jobflow.core.job import job
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import MongoStore
from matgl.apps.pes import Potential
from matgl.models import QET

from crystal.calculators.gpaw.isolated_atom_maker import GpawIsolatedAtomMaker
from crystal.calculators.gpaw.isolated_atom_runner import GpawIsolatedAtomRunResult
from crystal.calculators.mlip.qet_finetuning_maker import QetFineTuningMaker
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.gpaw import (
    GpawEigensolverName,
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
    GpawScfChargemolTaskDocument,
)
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawIsolatedAtomReferenceCollection,
    GpawManualIsolatedAtomReferenceSource,
    GpawRegistryIsolatedAtomReferenceSource,
    IsolatedAtomEnergy,
    IsolatedAtomEnergyReference,
)
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.qet_finetuning import (
    QetFineTuningLabelInput,
    QetFineTuningTrainingSettings,
)
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.qet_finetuning import create_gpaw_label_task_document

TEST_GPAW_VERSION = "25.7.0"
TEST_SILICON_SETUP_DIGEST = "a" * 64
TEST_SILICON_ISOLATED_ATOM_ENERGY = -5.0


def _runtime_provenance(element: str) -> GpawRuntimeProvenance:
    return GpawRuntimeProvenance(
        gpaw_version=TEST_GPAW_VERSION,
        atomic_data=(
            GpawAtomicDataInfo(
                element_symbol=element,
                paw_setup_sha256_digest=TEST_SILICON_SETUP_DIGEST,
            ),
        ),
    )


def _fake_isolated_atom_run(*, element: str, **_: object) -> GpawIsolatedAtomRunResult:
    return GpawIsolatedAtomRunResult(
        element=element,
        energy=TEST_SILICON_ISOLATED_ATOM_ENERGY,
        magnetic_moment=2.0,
        eigensolver=GpawEigensolverName.etdm_fdpw,
        runtime_provenance=_runtime_provenance(element),
    )


@job
def _build_gpaw_label_documents(
    _isolated_atom_collection: GpawIsolatedAtomReferenceCollection,
) -> list[GpawScfChargemolTaskDocument]:
    return [create_gpaw_label_task_document(index=index) for index in range(5)]


@pytest.mark.needs_mongodb
@pytest.mark.slow
def test_registry_backed_qet_finetuning_discovers_generated_isolated_atom_energy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_jobstore: JobStore,
    gpaw_isolated_atom_registry_store: MongoStore,
) -> None:
    """Run the real maker chain with only external GPAW data synthesized."""
    monkeypatch.setattr(
        "crystal.calculators.gpaw.isolated_atom_maker.run_gpaw_isolated_atom",
        _fake_isolated_atom_run,
    )

    isolated_atom_settings = GpawIsolatedAtomSimulationSettings()
    isolated_atom_flow = GpawIsolatedAtomMaker(
        simulation_settings=isolated_atom_settings
    ).make(elements=frozenset({"Si"}))
    label_job = _build_gpaw_label_documents(isolated_atom_flow.output)

    pretrained_model_path = tmp_path / "pretrained-model"
    Potential(
        model=QET(element_types=("Si",), cutoff=5.0, nblocks=1, width=0.5),
        data_mean=2.5,
        data_std=0.5,
        element_refs=np.asarray([0.0]),
        calc_forces=True,
        calc_stresses=True,
        calc_charge=True,
    ).save(pretrained_model_path)
    qet_job = QetFineTuningMaker(
        simulation_settings=QetFineTuningTrainingSettings(
            model=MlipModelRef(path=pretrained_model_path), max_epochs=1
        ),
        executor_settings=ExecutorSettings(number_of_gpus=0),
    ).make(
        label_input=QetFineTuningLabelInput(label_task_documents=label_job.output),
        isolated_atom_reference_source=GpawRegistryIsolatedAtomReferenceSource(
            simulation_settings=isolated_atom_settings
        ),
    )

    integration_flow = Flow(
        jobs=[isolated_atom_flow, label_job, qet_job], output=qet_job.output
    )
    run_results = run_locally(
        integration_flow, root_dir=tmp_path, create_folders=True, store=mock_jobstore
    )

    qet_result = run_results[qet_job.uuid][1].output
    assert qet_result.isolated_atom_reference.energies_by_element == {
        "Si": TEST_SILICON_ISOLATED_ATOM_ENERGY
    }
    assert qet_result.accepted_label_count == 5
    assert qet_result.failed_label_count == 0
    assert qet_result.model is not None
    exported_potential = matgl.load_model(qet_result.model.model_path)
    assert isinstance(exported_potential, Potential)
    assert exported_potential.calc_charge
    assert exported_potential.element_refs is not None
    assert torch.equal(
        exported_potential.element_refs.property_offset,
        torch.tensor([TEST_SILICON_ISOLATED_ATOM_ENERGY]),
    )
    assert gpaw_isolated_atom_registry_store.count({}) == 1


@pytest.mark.slow
def test_qet_continuation_consumes_prior_job_output_and_real_checkpoint(
    tmp_path: Path,
) -> None:
    """Continue a real run through Jobflow using its persisted checkpoint artifact."""
    labels = [create_gpaw_label_task_document(index=index) for index in range(5)]
    reference_source = GpawManualIsolatedAtomReferenceSource(
        isolated_atom_reference=IsolatedAtomEnergyReference(
            simulation_settings=GpawIsolatedAtomSimulationSettings(),
            isolated_atom_energies=(
                IsolatedAtomEnergy(
                    element="Si",
                    energy=TEST_SILICON_ISOLATED_ATOM_ENERGY,
                    runtime_provenance=_runtime_provenance("Si"),
                ),
            ),
        )
    )
    pretrained_model_path = tmp_path / "pretrained-model"
    Potential(
        model=QET(element_types=("Si",), cutoff=5.0, nblocks=1, width=0.5),
        data_mean=2.5,
        data_std=0.5,
        element_refs=np.asarray([0.0]),
        calc_forces=True,
        calc_stresses=True,
        calc_charge=True,
    ).save(pretrained_model_path)
    maker = QetFineTuningMaker(
        simulation_settings=QetFineTuningTrainingSettings(
            model=MlipModelRef(path=pretrained_model_path), max_epochs=1
        ),
        executor_settings=ExecutorSettings(number_of_gpus=0),
    )
    initial_job = maker.make(
        label_input=QetFineTuningLabelInput(label_task_documents=labels),
        isolated_atom_reference_source=reference_source,
    )
    continuation_job = maker.make_continuation(
        label_input=QetFineTuningLabelInput(label_task_documents=labels),
        isolated_atom_reference_source=reference_source,
        previous_result=initial_job.output,
        additional_epochs=1,
    )

    continuation_flow = Flow(
        jobs=[initial_job, continuation_job], output=continuation_job.output
    )
    run_results = run_locally(continuation_flow, root_dir=tmp_path, create_folders=True)

    initial_result = run_results[initial_job.uuid][1].output
    continuation_result = run_results[continuation_job.uuid][1].output
    assert initial_result.is_complete
    assert continuation_result.is_complete
    assert initial_result.input_document_count == 5
    assert continuation_result.input_document_count == 5
    assert initial_result.accepted_label_count == 5
    assert continuation_result.accepted_label_count == 5
    assert initial_result.failed_label_count == 0
    assert continuation_result.failed_label_count == 0
    assert continuation_result.checkpoint.epoch == initial_result.checkpoint.epoch + 1
    assert (
        continuation_result.checkpoint.global_step
        > initial_result.checkpoint.global_step
    )
    assert continuation_result.model.model_path.is_dir()
    assert initial_result.model.model_path.is_dir()
    assert continuation_result.model.model_path != initial_result.model.model_path
