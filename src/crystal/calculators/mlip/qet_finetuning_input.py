"""Validated QET fine-tuning input data."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crystal.schemas.qet_finetuning import (
    QetFineTuningLabelFailure,
    QetFineTuningRecord,
    validate_qet_fine_tuning_failed_label_details,
)


class QetFineTuningInputData(BaseModel):
    """Validated records and label-processing results for a fine-tuning run."""

    model_config = ConfigDict(frozen=True)

    accepted_records: tuple[QetFineTuningRecord, ...] = Field(
        min_length=1, description="Accepted labels available for QET fine-tuning."
    )
    failed_label_details: tuple[QetFineTuningLabelFailure, ...] = ()

    @property
    def accepted_label_count(self) -> int:
        """Return the number of task documents accepted for training."""
        return len(self.accepted_records)

    @property
    def failed_label_count(self) -> int:
        """Return the number of task documents excluded from training."""
        return len(self.failed_label_details)

    @property
    def input_document_count(self) -> int:
        """Return the total number of task documents processed."""
        return self.accepted_label_count + self.failed_label_count

    @model_validator(mode="after")
    def validate_failed_label_details(self) -> Self:
        """Require failure details to match the derived input outcome counts."""
        validate_qet_fine_tuning_failed_label_details(
            failed_label_details=self.failed_label_details,
            failed_label_count=self.failed_label_count,
            input_document_count=self.input_document_count,
        )
        return self
