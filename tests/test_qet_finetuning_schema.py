"""Tests for QET fine-tuning schemas."""

from __future__ import annotations

from pathlib import Path

import pytest

from crystal.constants.mlip import QET_FINE_TUNING_MAX_FAILURE_DETAILS
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.qet_finetuning import (
    QetFineTuningCheckpointRef,
    QetFineTuningLabelInput,
    QetFineTuningResult,
    QetFineTuningSplitCounts,
    QetFineTuningSplitSettings,
    QetFineTuningStoredJobQuery,
    QetFineTuningTrainingSettings,
)
from tests.helpers.qet_finetuning import create_isolated_atom_reference_for_test


@pytest.mark.parametrize(
    "label_input_kwargs",
    [
        {},
        {
            "label_task_documents": [],
            "label_source": QetFineTuningStoredJobQuery(criteria={}),
        },
    ],
)
def test_label_input_requires_exactly_one_source(
    label_input_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        QetFineTuningLabelInput(**label_input_kwargs)


def test_result_requires_failure_details_for_failed_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="failure details"):
        QetFineTuningResult(
            input_document_count=1,
            accepted_label_count=0,
            failed_label_count=1,
            split_counts=QetFineTuningSplitCounts(
                train_count=0, validation_count=0, test_count=0
            ),
            test_metrics={},
            checkpoint=QetFineTuningCheckpointRef(
                path=tmp_path / "checkpoint.ckpt", epoch=1, global_step=1
            ),
            model=MlipModelRef(path=tmp_path / "model"),
            isolated_atom_reference=create_isolated_atom_reference_for_test(),
        )


def test_result_bounds_persisted_failure_details(tmp_path: Path) -> None:
    failure_detail = {"input_document_index": 0, "reason": "label failed"}
    with pytest.raises(ValueError, match="at most"):
        QetFineTuningResult(
            input_document_count=QET_FINE_TUNING_MAX_FAILURE_DETAILS + 1,
            accepted_label_count=1,
            failed_label_count=QET_FINE_TUNING_MAX_FAILURE_DETAILS,
            failed_label_details=tuple(
                failure_detail | {"input_document_index": index}
                for index in range(QET_FINE_TUNING_MAX_FAILURE_DETAILS + 1)
            ),
            split_counts=QetFineTuningSplitCounts(
                train_count=1, validation_count=0, test_count=0
            ),
            test_metrics={},
            checkpoint=QetFineTuningCheckpointRef(
                path=tmp_path / "checkpoint.ckpt", epoch=1, global_step=1
            ),
            model=MlipModelRef(path=tmp_path / "model"),
            isolated_atom_reference=create_isolated_atom_reference_for_test(),
        )


def test_split_settings_require_fractions_to_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        QetFineTuningSplitSettings(
            train_fraction=0.7, validation_fraction=0.2, test_fraction=0.2
        )


def test_training_settings_default_to_matgl_aligned_training_behavior(
    tmp_path: Path,
) -> None:
    settings = QetFineTuningTrainingSettings(model=MlipModelRef(path=tmp_path))

    assert settings.loss == "huber_loss"
    assert settings.scheduler_t_max is None
    assert settings.effective_scheduler_t_max == settings.max_epochs


def test_training_settings_allow_explicit_loss_and_scheduler_horizon(
    tmp_path: Path,
) -> None:
    settings = QetFineTuningTrainingSettings(
        model=MlipModelRef(path=tmp_path), loss="mse_loss", scheduler_t_max=50
    )

    assert settings.loss == "mse_loss"
    assert settings.effective_scheduler_t_max == 50


def test_training_settings_reject_unknown_loss(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loss"):
        QetFineTuningTrainingSettings(model=MlipModelRef(path=tmp_path), loss="mae")
