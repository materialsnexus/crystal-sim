"""Configuration and bounded results for QET fine-tuning."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum, auto
from math import isclose
from pathlib import Path
from typing import Any, Literal, Self

from jobflow.core.reference import OutputReference
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.mlip import (
    QET_FINE_TUNING_FAILURE_REASON_TRUNCATION_SUFFIX,
    QET_FINE_TUNING_MAX_FAILURE_DETAILS,
    QET_FINE_TUNING_MAX_FAILURE_REASON_LENGTH,
)
from crystal.helpers.file import count_args_not_none
from crystal.schemas.gpaw import GpawScfChargemolTaskDocument
from crystal.schemas.gpaw_isolated_atom_reference import IsolatedAtomEnergyReference
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.properties import SinglePointProperties
from crystal.schemas.serialization import MSONableBaseModel
from crystal.schemas.settings import SimulationSettings


class QetFineTuningLabelFailure(MSONableBaseModel):
    """Reason a labelled input document was excluded from fine-tuning."""

    input_document_index: int = Field(
        ge=0, description="Zero-based index of the input document that failed."
    )
    reason: str = Field(
        min_length=1,
        max_length=QET_FINE_TUNING_MAX_FAILURE_REASON_LENGTH,
        description="Human-readable reason the document was excluded.",
    )

    @classmethod
    def from_unbounded_reason(cls, *, input_document_index: int, reason: str) -> Self:
        """Create bounded diagnostics from an external task error message."""
        if len(reason) > QET_FINE_TUNING_MAX_FAILURE_REASON_LENGTH:
            reason = (
                reason[
                    : QET_FINE_TUNING_MAX_FAILURE_REASON_LENGTH
                    - len(QET_FINE_TUNING_FAILURE_REASON_TRUNCATION_SUFFIX)
                ]
                + QET_FINE_TUNING_FAILURE_REASON_TRUNCATION_SUFFIX
            )
        return cls(input_document_index=input_document_index, reason=reason)


class QetFineTuningStoredJobQuery(MSONableBaseModel):
    """PyMongo criteria used to select stored label jobs."""

    criteria: dict[str, Any] = Field(
        description="PyMongo criteria applied to JobStore job documents."
    )


class QetFineTuningLabelInput(BaseModel):
    """Inline or JobStore-backed labels for QET fine-tuning."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label_task_documents: (
        list[GpawScfChargemolTaskDocument] | OutputReference | None
    ) = None
    label_source: QetFineTuningStoredJobQuery | None = None

    @model_validator(mode="after")
    def validate_exactly_one_label_source(self) -> Self:
        """Require exactly one way to provide QET fine-tuning labels."""
        if count_args_not_none(self.label_task_documents, self.label_source) != 1:
            msg = "Provide exactly one of label_task_documents or label_source."
            raise ValueError(msg)
        return self


def validate_qet_fine_tuning_failed_label_details(
    *,
    failed_label_details: Sequence[QetFineTuningLabelFailure],
    failed_label_count: int,
    input_document_count: int,
    details_are_truncated: bool = False,
) -> None:
    """Validate diagnostic details against aggregate input counts."""
    if not failed_label_details and failed_label_count == 0:
        return
    if not failed_label_details:
        msg = "QET fine-tuning failed labels must include failure details."
        raise ValueError(msg)

    if details_are_truncated:
        if len(failed_label_details) > failed_label_count:
            msg = (
                "QET fine-tuning failure details cannot exceed the failed label count."
            )
            raise ValueError(msg)
    elif len(failed_label_details) != failed_label_count:
        msg = "QET fine-tuning failure detail count must match the failed label count."
        raise ValueError(msg)

    input_document_indices = [
        failure.input_document_index for failure in failed_label_details
    ]
    if len(set(input_document_indices)) != len(input_document_indices):
        msg = "QET fine-tuning failure details must have unique input indices."
        raise ValueError(msg)
    if any(index >= input_document_count for index in input_document_indices):
        msg = (
            "QET fine-tuning failure input indices must be less than "
            "input_document_count."
        )
        raise ValueError(msg)


class QetFineTuningRecord(BaseModel):
    """One QET label in CRYSTAL SinglePointProperties units.

    Energy is eV, forces are eV/A, stress is ASE Voigt eV/A3, and charges are e.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    structure: PMG_Structure = Field(
        description="Structure associated with the labels."
    )
    energy: float = Field(description="Total energy in eV.")
    forces: tuple[tuple[float, ...], ...] = Field(
        description="Per-atom forces in eV/Å, with one three-component vector per atom."
    )
    stress: tuple[float, ...] = Field(
        description=(
            "Six-component ASE Voigt stress in eV/Å³, ordered as xx, yy, zz, yz, xz, xy."
        )
    )
    charges: tuple[float, ...] = Field(
        description="Per-atom charges in units of the elementary charge e."
    )

    @model_validator(mode="after")
    def validate_training_labels(self) -> Self:
        """Require labels that MatGL can consume for this structure."""
        SinglePointProperties(
            step=0,
            structure=self.structure,
            energy=self.energy,
            forces=self.forces,
            stress=self.stress,
            charges=self.charges,
        )
        return self

    @classmethod
    def from_single_point_properties(
        cls, properties: SinglePointProperties
    ) -> QetFineTuningRecord:
        """Create QET labels from charge-complete single-point properties."""
        if properties.charges is None:
            msg = "Single-point properties are missing charges."
            raise ValueError(msg)
        return cls(
            structure=properties.structure.copy(),
            energy=properties.energy,
            forces=properties.forces,
            stress=properties.stress,
            charges=properties.charges,
        )


class QetFineTuningSplitSettings(MSONableBaseModel):
    """Fractions and seed for a QET train-validation split with optional test data."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    train_fraction: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Fraction of accepted records assigned to model training.",
    )
    validation_fraction: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Fraction of accepted records used for checkpoint selection.",
    )
    test_fraction: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description=(
            "Optional fraction of accepted records reserved for final evaluation; "
            "set to 0 to skip test evaluation."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        description=(
            "Random seed used for the record split and training-loader shuffle order."
        ),
    )

    @model_validator(mode="after")
    def validate_fraction_total(self) -> Self:
        """Require every record to belong to exactly one configured split."""
        if not isclose(
            self.train_fraction + self.validation_fraction + self.test_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            msg = "QET fine-tuning split fractions must sum to 1.0."
            raise ValueError(msg)
        return self


class QetFineTuningStatus(StrEnum):
    """Outcome statuses for QET fine-tuning runs."""

    completed = auto()
    early_stopped = auto()
    walltime_interrupted = auto()


class QetFineTuningCheckpointRef(MSONableBaseModel):
    """Reference to a QET checkpoint and its training position."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    path: Path = Field(description="Shared-filesystem path to the checkpoint.")
    epoch: int = Field(
        ge=0, description="Completed training epoch represented by the checkpoint."
    )
    global_step: int = Field(
        ge=0, description="Optimizer step represented by the checkpoint."
    )


class QetFineTuningTrainingSettings(SimulationSettings):
    """Explicit settings for direct in-memory QET fine-tuning.

    The runner uses an explicit physical isolated-atom reference.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    model: MlipModelRef = Field(
        description="Reference to the pretrained MatGL Potential containing a QET model."
    )
    split: QetFineTuningSplitSettings = Field(
        default_factory=QetFineTuningSplitSettings,
        description="Fractions and seed for the accepted-record data split.",
    )
    batch_size: int = Field(
        default=8,
        gt=0,
        description="Number of structures processed in one training batch.",
    )
    max_epochs: int = Field(
        default=500,
        gt=0,
        description="Maximum number of complete passes through the training split.",
    )
    early_stopping_patience: int = Field(
        default=20,
        ge=0,
        description=(
            "Number of validation checks without improvement before early stopping."
        ),
    )
    early_stopping_min_delta: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum decrease in validation loss required to count as an improvement."
        ),
    )
    learning_rate: float = Field(
        default=1e-5,
        gt=0.0,
        description="Initial learning rate for the AdamW optimizer.",
    )
    weight_decay: float = Field(
        default=1e-5, ge=0.0, description="AdamW decoupled weight-decay coefficient."
    )
    scheduler_t_max: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Number of training epochs in the cosine-annealing schedule; defaults "
            "to max_epochs."
        ),
    )
    loss: Literal["huber_loss", "mse_loss", "smooth_l1_loss", "l1_loss"] = Field(
        default="huber_loss",
        description=(
            "MatGL loss function used for energy, force, stress, and charge terms."
        ),
    )
    scheduler_eta_min: float = Field(
        default=1e-6,
        ge=0.0,
        description="Minimum learning rate reached by cosine annealing.",
    )
    energy_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative weight of the energy term in the training loss.",
    )
    force_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative weight of the force term in the training loss.",
    )
    stress_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Relative weight of the stress term in the training loss.",
    )
    charge_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Relative weight of the charge term in the training loss.",
    )

    @property
    def effective_scheduler_t_max(self) -> int:
        """Return the configured scheduler horizon or the training horizon."""
        if self.scheduler_t_max is None:
            return self.max_epochs
        return self.scheduler_t_max


class QetFineTuningSplitCounts(MSONableBaseModel):
    """Bounded record counts for one QET train-validation-test split."""

    model_config = ConfigDict(frozen=True)

    train_count: int = Field(
        ge=0, description="Number of accepted records assigned to model training."
    )
    validation_count: int = Field(
        ge=0, description="Number of accepted records used for checkpoint selection."
    )
    test_count: int = Field(
        ge=0, description="Number of accepted records reserved for final evaluation."
    )


class QetFineTuningResult(MSONableBaseModel):
    """Bounded outcome and artifacts from one QET fine-tuning run."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False, extra="forbid")

    status: QetFineTuningStatus = Field(
        default=QetFineTuningStatus.completed,
        description="Outcome status of the QET fine-tuning run.",
    )
    input_document_count: int = Field(
        ge=0, description="Number of input documents received by the maker."
    )
    accepted_label_count: int = Field(
        ge=0, description="Number of input documents converted into training labels."
    )
    failed_label_count: int = Field(
        ge=0, description="Number of input documents excluded from training."
    )
    failed_label_details: tuple[QetFineTuningLabelFailure, ...] = Field(
        default=(),
        max_length=QET_FINE_TUNING_MAX_FAILURE_DETAILS,
        description="Indexes and reasons for documents excluded from training.",
    )
    failed_label_details_truncated: bool = Field(
        default=False,
        description="Whether only a bounded prefix of failure details was retained.",
    )
    split_counts: QetFineTuningSplitCounts = Field(
        description="Counts of accepted labels in the train, validation, and test splits."
    )
    test_metrics: dict[str, float] = Field(
        description=(
            "Scalar metric names and values from the selected checkpoint's test run; "
            "empty when test evaluation is skipped."
        )
    )
    checkpoint: QetFineTuningCheckpointRef = Field(
        description="Selected or latest training checkpoint."
    )
    model: MlipModelRef = Field(
        description="Exported MatGL Potential containing the fine-tuned QET model."
    )
    isolated_atom_reference: IsolatedAtomEnergyReference = Field(
        description="Physical GPAW isolated-atom reference used for QET energies."
    )

    @model_validator(mode="after")
    def validate_outcome_counts(self) -> Self:
        """Require bounded result counts to describe the same records."""
        validate_qet_fine_tuning_failed_label_details(
            failed_label_details=self.failed_label_details,
            failed_label_count=self.failed_label_count,
            input_document_count=self.input_document_count,
            details_are_truncated=self.failed_label_details_truncated,
        )
        if (
            self.accepted_label_count + self.failed_label_count
            != self.input_document_count
        ):
            msg = (
                "QET fine-tuning accepted and failed labels must equal input documents."
            )
            raise ValueError(msg)
        split_count = (
            self.split_counts.train_count
            + self.split_counts.validation_count
            + self.split_counts.test_count
        )
        if split_count != self.accepted_label_count:
            msg = "QET fine-tuning split counts must equal accepted labels."
            raise ValueError(msg)
        return self

    @property
    def is_complete(self) -> bool:
        """Whether the run reached its normal completed outcome."""
        return self.status is QetFineTuningStatus.completed

    @property
    def is_partial(self) -> bool:
        """Whether the run stopped before its configured epoch limit."""
        return not self.is_complete
