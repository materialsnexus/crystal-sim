"""Tests for the GPAW-labelled Jobflow QET fine-tuning maker."""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from unittest.mock import patch

import pytest
from jobflow.core.store import JobStore
from jobflow.managers.fireworks import job_to_firework
from jobflow.managers.local import run_locally
from maggma.stores import MemoryStore

from crystal.calculators.mlip.qet_finetuning_maker import QetFineTuningMaker
from crystal.calculators.mlip.qet_finetuning_runner import QetFineTuningStorageError
from crystal.constants.fireworks import (
    QUEUEADAPTER_SIGNAL_KEY,
    SECONDS_PER_MINUTE,
    SLURM_TERM_SIGNAL_FORMAT,
)
from crystal.constants.storage import MONGODB_MAX_SAFE_DOCUMENT_BYTES
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.gpaw import (
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
)
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawManualIsolatedAtomReferenceSource,
    IsolatedAtomEnergy,
    IsolatedAtomEnergyReference,
)
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.qet_finetuning import (
    QetFineTuningCheckpointRef,
    QetFineTuningLabelFailure,
    QetFineTuningLabelInput,
    QetFineTuningResult,
    QetFineTuningSplitCounts,
    QetFineTuningStatus,
    QetFineTuningStoredJobQuery,
    QetFineTuningTrainingSettings,
)
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.qet_finetuning import (
    create_gpaw_label_task_document,
    store_gpaw_label_task_document_for_test,
)

FAILED_DOCUMENT_ERROR = "calculation failed"
TEST_GPAW_VERSION = "25.7.0"
TEST_SILICON_SETUP_DIGEST = "a" * 64
TEST_SILICON_ISOLATED_ATOM_ENERGY = -5.0


def _runtime_provenance() -> GpawRuntimeProvenance:
    return GpawRuntimeProvenance(
        gpaw_version=TEST_GPAW_VERSION,
        atomic_data=(
            GpawAtomicDataInfo(
                element_symbol="Si", paw_setup_sha256_digest=TEST_SILICON_SETUP_DIGEST
            ),
        ),
    )


def _isolated_atom_reference() -> IsolatedAtomEnergyReference:
    return IsolatedAtomEnergyReference(
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
        isolated_atom_energies=(
            IsolatedAtomEnergy(
                element="Si",
                energy=TEST_SILICON_ISOLATED_ATOM_ENERGY,
                runtime_provenance=_runtime_provenance(),
            ),
        ),
    )


def _result(tmp_path: Path, *, status: QetFineTuningStatus) -> QetFineTuningResult:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    return QetFineTuningResult(
        status=status,
        input_document_count=1,
        accepted_label_count=1,
        failed_label_count=0,
        split_counts=QetFineTuningSplitCounts(
            train_count=1, validation_count=0, test_count=0
        ),
        test_metrics={},
        checkpoint=QetFineTuningCheckpointRef(
            path=tmp_path / "last.ckpt", epoch=8, global_step=123
        ),
        model=MlipModelRef(path=model_path),
        isolated_atom_reference=_isolated_atom_reference(),
    )


def _maker(tmp_path: Path) -> QetFineTuningMaker:
    model_path = tmp_path / "pretrained-model"
    model_path.mkdir(exist_ok=True)
    return QetFineTuningMaker(
        simulation_settings=QetFineTuningTrainingSettings(
            model=MlipModelRef(path=model_path)
        ),
        executor_settings=ExecutorSettings(number_of_gpus=0),
    )


def test_maker_runs_manual_reference_with_inline_labels(tmp_path: Path) -> None:
    expected_result = QetFineTuningResult(
        input_document_count=3,
        accepted_label_count=1,
        failed_label_count=2,
        split_counts=QetFineTuningSplitCounts(
            train_count=1, validation_count=0, test_count=0
        ),
        failed_label_details=(
            QetFineTuningLabelFailure(
                input_document_index=1, reason=FAILED_DOCUMENT_ERROR
            ),
            QetFineTuningLabelFailure(input_document_index=2, reason="missing charges"),
        ),
        test_metrics={},
        checkpoint=QetFineTuningCheckpointRef(
            path=tmp_path / "last.ckpt", epoch=8, global_step=123
        ),
        model=MlipModelRef(path=tmp_path / "exported-model"),
        isolated_atom_reference=_isolated_atom_reference(),
    )
    maker = _maker(tmp_path)
    labels = [
        create_gpaw_label_task_document(),
        create_gpaw_label_task_document(errors=FAILED_DOCUMENT_ERROR),
        create_gpaw_label_task_document(charges=None),
    ]
    reference_source = GpawManualIsolatedAtomReferenceSource(
        isolated_atom_reference=_isolated_atom_reference()
    )

    with patch(
        "crystal.calculators.mlip.qet_finetuning_maker.QetFineTuningRunner"
    ) as runner_class:
        runner_class.return_value.run.return_value = expected_result
        qet_job = maker.make(
            label_input=QetFineTuningLabelInput(label_task_documents=labels),
            isolated_atom_reference_source=reference_source,
        )
        local_run_results = run_locally(qet_job, root_dir=tmp_path, create_folders=True)

    expected_signal_delay_seconds = ceil(
        maker.executor_settings.walltime_safety_margin_in_minutes * SECONDS_PER_MINUTE
    )
    assert qet_job.metadata["_queueadapter"][QUEUEADAPTER_SIGNAL_KEY] == (
        SLURM_TERM_SIGNAL_FORMAT.format(seconds=expected_signal_delay_seconds)
    )
    assert local_run_results[qet_job.uuid][1].output == expected_result
    runner_kwargs = runner_class.call_args.kwargs
    runner_input = runner_kwargs["qet_fine_tuning_input_data"]
    assert runner_input.input_document_count == 3
    assert runner_input.accepted_label_count == 1
    assert runner_input.failed_label_count == 2
    assert (
        runner_kwargs["isolated_atom_reference"]
        == reference_source.isolated_atom_reference
    )
    assert runner_kwargs["training_settings"] == maker.simulation_settings
    assert (
        runner_kwargs["training_walltime_seconds"]
        == (
            maker.executor_settings.maximum_walltime_in_minutes
            - maker.executor_settings.walltime_safety_margin_in_minutes
        )
        * SECONDS_PER_MINUTE
    )


def test_maker_keeps_stored_query_firework_specs_smaller_than_inline_labels(
    tmp_path: Path,
) -> None:
    maker = _maker(tmp_path)
    reference_source = GpawManualIsolatedAtomReferenceSource(
        isolated_atom_reference=_isolated_atom_reference()
    )
    stored_query_job = maker.make(
        label_input=QetFineTuningLabelInput(
            label_source=QetFineTuningStoredJobQuery(
                criteria={"name": "gpaw-scf-chargemol"}
            )
        ),
        isolated_atom_reference_source=reference_source,
    )
    inline_job = maker.make(
        label_input=QetFineTuningLabelInput(
            label_task_documents=[
                create_gpaw_label_task_document(index=index) for index in range(20)
            ]
        ),
        isolated_atom_reference_source=reference_source,
    )

    stored_query_spec_size = len(
        json.dumps(job_to_firework(stored_query_job).to_dict())
    )
    inline_spec_size = len(json.dumps(job_to_firework(inline_job).to_dict()))

    assert stored_query_spec_size < inline_spec_size


def test_maker_rejects_oversized_inline_firework_payload(tmp_path: Path) -> None:
    maker = _maker(tmp_path)
    reference_source = GpawManualIsolatedAtomReferenceSource(
        isolated_atom_reference=_isolated_atom_reference()
    )
    oversized_error = "x" * MONGODB_MAX_SAFE_DOCUMENT_BYTES

    with pytest.raises(QetFineTuningStorageError, match="stored JobStore label query"):
        maker.make(
            label_input=QetFineTuningLabelInput(
                label_task_documents=[
                    create_gpaw_label_task_document(),
                    create_gpaw_label_task_document(errors=oversized_error),
                ]
            ),
            isolated_atom_reference_source=reference_source,
        )


def test_maker_filters_failed_labels_loaded_from_the_jobstore(tmp_path: Path) -> None:
    jobstore = JobStore(docs_store=MemoryStore())
    jobstore.connect()
    successful_label_job = store_gpaw_label_task_document_for_test(
        create_gpaw_label_task_document(index=1)
    )
    failed_label_job = store_gpaw_label_task_document_for_test(
        create_gpaw_label_task_document(index=2, errors=FAILED_DOCUMENT_ERROR)
    )
    for label_job in (successful_label_job, failed_label_job):
        label_job.update_metadata({"label_query_group": "qet"}, dynamic=False)
        run_locally(label_job, root_dir=tmp_path, create_folders=True, store=jobstore)

    failed_label_index = sorted(
        (successful_label_job, failed_label_job), key=lambda item: item.uuid
    ).index(failed_label_job)
    expected_result = QetFineTuningResult(
        input_document_count=2,
        accepted_label_count=1,
        failed_label_count=1,
        failed_label_details=(
            QetFineTuningLabelFailure(
                input_document_index=failed_label_index, reason=FAILED_DOCUMENT_ERROR
            ),
        ),
        split_counts=QetFineTuningSplitCounts(
            train_count=1, validation_count=0, test_count=0
        ),
        test_metrics={},
        checkpoint=QetFineTuningCheckpointRef(
            path=tmp_path / "checkpoint.ckpt", epoch=1, global_step=1
        ),
        model=MlipModelRef(path=tmp_path / "model"),
        isolated_atom_reference=_isolated_atom_reference(),
    )
    maker = _maker(tmp_path)

    with patch(
        "crystal.calculators.mlip.qet_finetuning_maker.QetFineTuningRunner"
    ) as runner_class:
        runner_class.return_value.run.return_value = expected_result
        qet_job = maker.make(
            label_input=QetFineTuningLabelInput(
                label_source=QetFineTuningStoredJobQuery(
                    criteria={"metadata.label_query_group": "qet"}
                )
            ),
            isolated_atom_reference_source=GpawManualIsolatedAtomReferenceSource(
                isolated_atom_reference=_isolated_atom_reference()
            ),
        )
        local_run_results = run_locally(
            qet_job, root_dir=tmp_path, create_folders=True, store=jobstore
        )

    assert local_run_results[qet_job.uuid][1].output == expected_result
    input_data = runner_class.call_args.kwargs["qet_fine_tuning_input_data"]
    assert input_data.accepted_label_count == 1
    assert input_data.failed_label_count == 1
    assert input_data.failed_label_details[0].reason == FAILED_DOCUMENT_ERROR


def test_maker_continuation_reuses_result_checkpoint(tmp_path: Path) -> None:
    maker = _maker(tmp_path)
    previous_result = _result(tmp_path, status=QetFineTuningStatus.walltime_interrupted)
    reference_source = GpawManualIsolatedAtomReferenceSource(
        isolated_atom_reference=_isolated_atom_reference()
    )

    with patch(
        "crystal.calculators.mlip.qet_finetuning_maker.QetFineTuningRunner"
    ) as runner_class:
        runner_class.return_value.run.return_value = previous_result
        continuation_job = maker.make_continuation(
            label_input=QetFineTuningLabelInput(
                label_task_documents=[create_gpaw_label_task_document()]
            ),
            isolated_atom_reference_source=reference_source,
            previous_result=previous_result,
            additional_epochs=12,
        )
        run_locally(continuation_job, root_dir=tmp_path, create_folders=True)

    assert runner_class.call_args.kwargs["checkpoint"] == previous_result.checkpoint
    assert runner_class.call_args.kwargs["training_settings"].max_epochs == 21
    assert runner_class.call_args.kwargs["training_walltime_seconds"] == (1440 - 5) * 60
