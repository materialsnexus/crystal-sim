"""Tests for GPAW QET fine-tuning input conversion."""

from __future__ import annotations

from crystal.calculators.mlip.qet_finetuning_maker import (
    collect_qet_fine_tuning_input_data,
)
from crystal.constants.mlip import (
    QET_FINE_TUNING_FAILURE_REASON_TRUNCATION_SUFFIX,
    QET_FINE_TUNING_MAX_FAILURE_REASON_LENGTH,
)
from crystal.schemas.gpaw import GpawScfChargemolTaskDocument
from crystal.schemas.qet_finetuning import QetFineTuningLabelFailure
from tests.helpers.qet_finetuning import create_gpaw_label_task_document

FAILED_DOCUMENT_ERROR = "backend calculation failed"


def test_collection_converts_gpaw_labels_and_preserves_order() -> None:
    qet_fine_tuning_input_data = collect_qet_fine_tuning_input_data(
        [
            create_gpaw_label_task_document(energy=1.0),
            create_gpaw_label_task_document(energy=2.0),
        ]
    )

    assert tuple(
        qet_fine_tuning_record.energy
        for qet_fine_tuning_record in qet_fine_tuning_input_data.accepted_records
    ) == (1.0, 2.0)
    assert qet_fine_tuning_input_data.input_document_count == 2
    assert qet_fine_tuning_input_data.accepted_label_count == 2
    assert qet_fine_tuning_input_data.failed_label_count == 0


def test_collection_omits_failed_and_incomplete_gpaw_labels() -> None:
    qet_fine_tuning_input_data = collect_qet_fine_tuning_input_data(
        [
            create_gpaw_label_task_document(energy=1.0),
            GpawScfChargemolTaskDocument(
                dir_name="gpaw",
                errors=FAILED_DOCUMENT_ERROR,
                properties=None,
                structure=None,
                chargemol_run_info=None,
            ),
            create_gpaw_label_task_document(charges=None),
        ]
    )

    assert tuple(
        qet_fine_tuning_record.energy
        for qet_fine_tuning_record in qet_fine_tuning_input_data.accepted_records
    ) == (1.0,)
    assert qet_fine_tuning_input_data.failed_label_details == (
        QetFineTuningLabelFailure(input_document_index=1, reason=FAILED_DOCUMENT_ERROR),
        QetFineTuningLabelFailure(
            input_document_index=2,
            reason="Single-point properties are missing charges.",
        ),
    )


def test_collection_bounds_external_failure_reason() -> None:
    failure_reason = "backend calculation failed " + "x" * 1_000

    qet_fine_tuning_input_data = collect_qet_fine_tuning_input_data(
        [
            create_gpaw_label_task_document(),
            create_gpaw_label_task_document(errors=failure_reason),
        ]
    )

    stored_reason = qet_fine_tuning_input_data.failed_label_details[0].reason
    assert len(stored_reason) == QET_FINE_TUNING_MAX_FAILURE_REASON_LENGTH
    assert stored_reason.endswith(QET_FINE_TUNING_FAILURE_REASON_TRUNCATION_SUFFIX)
