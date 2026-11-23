"""Regression coverage for large JobStore-backed QET label datasets."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bson import BSON
from jobflow.core.schemas import JobStoreDocument
from jobflow.core.store import JobStore
from jobflow.managers.fireworks import job_to_firework
from jobflow.managers.local import run_locally
from monty.json import jsanitize
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.qet_finetuning_maker import QetFineTuningMaker
from crystal.calculators.mlip.qet_finetuning_runner import QetFineTuningStorageError
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawManualIsolatedAtomReferenceSource,
)
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.qet_finetuning import (
    QetFineTuningCheckpointRef,
    QetFineTuningLabelInput,
    QetFineTuningResult,
    QetFineTuningSplitCounts,
    QetFineTuningStoredJobQuery,
    QetFineTuningTrainingSettings,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.qet_finetuning import (
    create_gpaw_label_task_document,
    create_isolated_atom_reference_for_test,
)

LABEL_COUNT = 5_000
MONGODB_DOCUMENT_LIMIT_BYTES = 16 * 1024 * 1024
LARGE_STRUCTURE_CHARGES = (0.1, -0.1) * 5


def _large_test_structure() -> PMG_Structure:
    large_structure = TEST_SI_STRUCTURE.copy()
    large_structure.make_supercell((5, 1, 1))
    return large_structure


def _store_label_documents(
    *, jobstore: JobStore, label_documents: list[object]
) -> None:
    stored_documents = [
        JobStoreDocument(
            uuid=f"large-qet-label-{index:05d}",
            index=1,
            output=jsanitize(
                label_document, strict=True, enum_values=True, allow_bson=True
            ),
            completed_at="2026-01-01T00:00:00",
            metadata={"label_query_group": "large-qet-regression"},
            name="stored-qet-label",
        )
        for index, label_document in enumerate(label_documents)
    ]
    jobstore.update(stored_documents, key=["uuid", "index"])


@pytest.mark.needs_mongodb
@pytest.mark.slow
def test_qet_finetuning_retrieves_5000_labels_without_large_firework_payload(
    tmp_path: Path, mock_jobstore: JobStore
) -> None:
    large_structure = _large_test_structure()
    label_documents = [
        create_gpaw_label_task_document(
            index=index, structure=large_structure, charges=LARGE_STRUCTURE_CHARGES
        )
        for index in range(LABEL_COUNT)
    ]
    _store_label_documents(
        jobstore=mock_jobstore,
        label_documents=label_documents,  # type: ignore[arg-type]
    )

    maker = QetFineTuningMaker(
        simulation_settings=QetFineTuningTrainingSettings(
            model=MlipModelRef(path=tmp_path / "pretrained-model")
        )
    )
    isolated_atom_reference_source = GpawManualIsolatedAtomReferenceSource(
        isolated_atom_reference=create_isolated_atom_reference_for_test()
    )
    stored_query_job = maker.make(
        label_input=QetFineTuningLabelInput(
            label_source=QetFineTuningStoredJobQuery(
                criteria={"metadata.label_query_group": "large-qet-regression"}
            )
        ),
        isolated_atom_reference_source=isolated_atom_reference_source,
    )
    stored_payload_size = len(BSON.encode(job_to_firework(stored_query_job).to_dict()))
    with pytest.raises(QetFineTuningStorageError, match="stored JobStore label query"):
        maker.make(
            label_input=QetFineTuningLabelInput(label_task_documents=label_documents),
            isolated_atom_reference_source=isolated_atom_reference_source,
        )
    expected_result = QetFineTuningResult(
        input_document_count=LABEL_COUNT,
        accepted_label_count=LABEL_COUNT,
        failed_label_count=0,
        split_counts=QetFineTuningSplitCounts(
            train_count=4_000, validation_count=500, test_count=500
        ),
        test_metrics={},
        checkpoint=QetFineTuningCheckpointRef(
            path=tmp_path / "checkpoint.ckpt", epoch=1, global_step=1
        ),
        model=MlipModelRef(path=tmp_path / "model"),
        isolated_atom_reference=create_isolated_atom_reference_for_test(),
    )

    with patch(
        "crystal.calculators.mlip.qet_finetuning_maker.QetFineTuningRunner"
    ) as runner_class:
        runner = MagicMock()
        runner.run.return_value = expected_result
        runner_class.return_value = runner
        run_results = run_locally(
            stored_query_job,
            root_dir=tmp_path,
            create_folders=True,
            store=mock_jobstore,
        )

    input_data = runner_class.call_args.kwargs["qet_fine_tuning_input_data"]
    assert stored_payload_size < MONGODB_DOCUMENT_LIMIT_BYTES
    assert input_data.input_document_count == LABEL_COUNT
    assert input_data.accepted_label_count == LABEL_COUNT
    assert run_results[stored_query_job.uuid][1].output == expected_result
