"""Jobflow adapter for direct in-memory QET fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

from bson import BSON
from jobflow.core.job import Job, JobConfig, job
from jobflow.core.reference import OutputReference
from jobflow.core.state import CURRENT_JOB
from jobflow.managers.fireworks import job_to_firework
from pydantic import ValidationError

from crystal.calculators.gpaw.isolated_atom_registry_jobflow import (
    get_gpaw_isolated_atom_registry_repository,
)
from crystal.calculators.gpaw.isolated_atom_registry_resolver import (
    resolve_gpaw_isolated_atom_reference,
)
from crystal.calculators.mlip.qet_finetuning_input import QetFineTuningInputData
from crystal.calculators.mlip.qet_finetuning_runner import (
    QetFineTuningRunner,
    QetFineTuningStorageError,
)
from crystal.constants.fireworks import (
    QUEUEADAPTER_SIGNAL_KEY,
    SECONDS_PER_MINUTE,
    SLURM_TERM_SIGNAL_FORMAT,
)
from crystal.constants.storage import MONGODB_MAX_SAFE_DOCUMENT_BYTES
from crystal.repositories.qet_finetuning_label_repository import (
    load_labelled_documents_for_qet_fine_tuning,
)
from crystal.schemas.base import BaseMaker
from crystal.schemas.fireworks import SlurmSettings
from crystal.schemas.gpaw import GpawScfChargemolTaskDocument
from crystal.schemas.gpaw_isolated_atom_reference import (
    GpawIsolatedAtomReferenceSource,
    GpawRegistryIsolatedAtomReferenceSource,
    IsolatedAtomEnergyReference,
)
from crystal.schemas.maker_names import MakerName
from crystal.schemas.properties import SinglePointProperties
from crystal.schemas.qet_finetuning import (
    QetFineTuningCheckpointRef,
    QetFineTuningLabelFailure,
    QetFineTuningLabelInput,
    QetFineTuningRecord,
    QetFineTuningResult,
    QetFineTuningTrainingSettings,
)


@dataclass(kw_only=True)
class QetFineTuningMaker(BaseMaker):
    """Fine-tune one QET potential from GPAW SCF plus Chargemol labels."""

    name: MakerName = MakerName.qet_finetuning
    simulation_settings: QetFineTuningTrainingSettings

    # Jobflow cannot serialize this MSONable output class as output_schema:
    # Monty calls as_dict on the schema class without an instance.
    # Keep the decorator schema-free until the upstream issue is fixed.
    @job(config=JobConfig(expose_store=True))
    def _run_qet_fine_tuning(
        self,
        label_input: QetFineTuningLabelInput,
        isolated_atom_reference_source: GpawIsolatedAtomReferenceSource,
        previous_result: QetFineTuningResult | None = None,
        additional_epochs: int = 0,
    ) -> QetFineTuningResult:
        """Load labels, resolve references, and run one QET fine-tuning job."""
        resolved_label_task_documents = self._resolve_label_task_documents(label_input)
        registry_repository = (
            get_gpaw_isolated_atom_registry_repository(CURRENT_JOB.store)
            if isinstance(
                isolated_atom_reference_source, GpawRegistryIsolatedAtomReferenceSource
            )
            else None
        )
        isolated_atom_reference = resolve_gpaw_isolated_atom_reference(
            label_task_documents=resolved_label_task_documents,
            reference_source=isolated_atom_reference_source,
            registry_repository=registry_repository,
        )
        return self._run_resolved_qet_fine_tuning(
            label_task_documents=resolved_label_task_documents,
            isolated_atom_reference=isolated_atom_reference,
            previous_result=previous_result,
            additional_epochs=additional_epochs,
        )

    @staticmethod
    def _resolve_label_task_documents(
        label_input: QetFineTuningLabelInput,
    ) -> list[GpawScfChargemolTaskDocument]:
        """Resolve inline labels or retrieve them from the current JobStore."""
        if label_input.label_source is not None:
            return load_labelled_documents_for_qet_fine_tuning(
                jobstore=CURRENT_JOB.store, stored_job_query=label_input.label_source
            )
        if label_input.label_task_documents is None:
            msg = "Inline QET fine-tuning labels are missing."
            raise ValueError(msg)
        if not isinstance(label_input.label_task_documents, list):
            msg = "Inline QET fine-tuning label references did not resolve to a list."
            raise TypeError(msg)
        return label_input.label_task_documents

    def _run_resolved_qet_fine_tuning(
        self,
        *,
        label_task_documents: list[GpawScfChargemolTaskDocument],
        isolated_atom_reference: IsolatedAtomEnergyReference,
        previous_result: QetFineTuningResult | None = None,
        additional_epochs: int = 0,
    ) -> QetFineTuningResult:
        checkpoint = _get_continuation_checkpoint(
            previous_result=previous_result, additional_epochs=additional_epochs
        )
        training_settings = self.simulation_settings
        if checkpoint is not None:
            training_settings = self.simulation_settings.model_copy(
                update={"max_epochs": checkpoint.epoch + 1 + additional_epochs}
            )
        qet_fine_tuning_runner = QetFineTuningRunner(
            qet_fine_tuning_input_data=collect_qet_fine_tuning_input_data(
                label_task_documents
            ),
            isolated_atom_reference=isolated_atom_reference,
            training_settings=training_settings,
            use_gpu=self.executor_settings.number_of_gpus > 0,
            output_directory=Path.cwd() / "qet-finetuning",
            training_walltime_seconds=_calculate_training_walltime_seconds(
                maximum_walltime_in_minutes=(
                    self.executor_settings.maximum_walltime_in_minutes
                ),
                safety_margin_in_minutes=(
                    self.executor_settings.walltime_safety_margin_in_minutes
                ),
            ),
            checkpoint=checkpoint,
        )

        return qet_fine_tuning_runner.run()

    def make(
        self,
        *,
        label_input: QetFineTuningLabelInput,
        isolated_atom_reference_source: GpawIsolatedAtomReferenceSource,
    ) -> Job:
        """Build the QET fine-tuning job and apply its Slurm metadata.

        Direct labels are suitable only for small standalone jobs. Registry lookup
        requires the configured Jobflow registry store; manual input is offline.
        """
        qet_fine_tuning_job = self._run_qet_fine_tuning(
            label_input=label_input,
            isolated_atom_reference_source=isolated_atom_reference_source,
        )
        return self._apply_qet_slurm_metadata(qet_fine_tuning_job)

    def make_continuation(
        self,
        *,
        label_input: QetFineTuningLabelInput | None = None,
        label_task_documents: (
            list[GpawScfChargemolTaskDocument] | OutputReference | None
        ) = None,
        isolated_atom_reference_source: GpawIsolatedAtomReferenceSource,
        previous_result: QetFineTuningResult | OutputReference,
        additional_epochs: int,
    ) -> Job:
        """Build a follow-up job from a prior partial-run result artifact."""
        if additional_epochs <= 0:
            message = "additional_epochs must be positive."
            raise ValueError(message)
        if label_input is not None and label_task_documents is not None:
            message = "Provide only one of label_input or label_task_documents."
            raise ValueError(message)
        if label_input is None:
            if label_task_documents is None:
                message = "A QET fine-tuning label input is required."
                raise ValueError(message)
            label_input = QetFineTuningLabelInput(
                label_task_documents=label_task_documents
            )
        qet_fine_tuning_job = self._run_qet_fine_tuning(
            label_input=label_input,
            isolated_atom_reference_source=isolated_atom_reference_source,
            previous_result=previous_result,
            additional_epochs=additional_epochs,
        )
        return self._apply_qet_slurm_metadata(qet_fine_tuning_job)

    def _apply_qet_slurm_metadata(self, qet_fine_tuning_job: Job) -> Job:
        """Apply the QET-specific pre-timeout scheduler metadata."""
        qet_fine_tuning_job.update_metadata(
            self._qet_slurm_settings().to_queueadapter_spec(), dynamic=False
        )
        _validate_qet_firework_payload_size(qet_fine_tuning_job)
        return qet_fine_tuning_job

    def _qet_slurm_settings(self) -> SlurmSettings:
        """Add a pre-timeout SIGTERM warning to the QET queue request."""
        signal_delay_seconds = ceil(
            self.executor_settings.walltime_safety_margin_in_minutes
            * SECONDS_PER_MINUTE
        )
        return self.slurm_settings.model_copy(
            update={
                QUEUEADAPTER_SIGNAL_KEY: SLURM_TERM_SIGNAL_FORMAT.format(
                    seconds=signal_delay_seconds
                )
            }
        )


def _get_continuation_checkpoint(
    *, previous_result: QetFineTuningResult | None, additional_epochs: int
) -> QetFineTuningCheckpointRef | None:
    """Return the checkpoint and validate a continuation request."""
    if previous_result is None:
        return None
    if additional_epochs <= 0:
        message = "additional_epochs must be positive."
        raise ValueError(message)
    try:
        validated_result = QetFineTuningResult.model_validate(previous_result)
    except ValidationError as exc:
        message = "QET continuation requires a prior QET fine-tuning result."
        raise ValueError(message) from exc
    return validated_result.checkpoint


def _calculate_training_walltime_seconds(
    *, maximum_walltime_in_minutes: float, safety_margin_in_minutes: float
) -> float:
    """Reserve scheduler cleanup time before the QET trainer's stop deadline."""
    effective_walltime_in_minutes = (
        maximum_walltime_in_minutes - safety_margin_in_minutes
    )
    if effective_walltime_in_minutes <= 0:
        message = (
            "QET fine-tuning requires maximum_walltime_in_minutes to exceed "
            "walltime_safety_margin_in_minutes."
        )
        raise ValueError(message)
    return effective_walltime_in_minutes * SECONDS_PER_MINUTE


def collect_qet_fine_tuning_input_data(
    label_task_documents: object,
) -> QetFineTuningInputData:
    """Convert GPAW labels into validated QET fine-tuning input data."""
    validated_label_task_documents = _validate_label_task_document_collection(
        label_task_documents
    )

    accepted_records: list[QetFineTuningRecord] = []
    failed_label_details: list[QetFineTuningLabelFailure] = []

    for input_document_index, task_document in enumerate(
        validated_label_task_documents
    ):
        if task_document.errors:
            failed_label_details.append(
                QetFineTuningLabelFailure.from_unbounded_reason(
                    input_document_index=input_document_index,
                    reason=task_document.errors,
                )
            )
            continue

        qet_fine_tuning_record, failure_reason = (
            _create_qet_fine_tuning_record_or_failure(task_document=task_document)
        )

        if qet_fine_tuning_record is None:
            if failure_reason is None:
                failure_reason = "QET fine-tuning label conversion failed."
            failed_label_details.append(
                QetFineTuningLabelFailure.from_unbounded_reason(
                    input_document_index=input_document_index, reason=failure_reason
                )
            )
            continue

        accepted_records.append(qet_fine_tuning_record)

    return QetFineTuningInputData(
        accepted_records=tuple(accepted_records),
        failed_label_details=tuple(failed_label_details),
    )


def _validate_label_task_document_collection(
    label_task_documents: object,
) -> list[GpawScfChargemolTaskDocument]:
    """Require a list containing only GPAW SCF plus Chargemol labels."""
    if not isinstance(label_task_documents, list):
        message = (
            "QET fine-tuning input must be a list of GpawScfChargemolTaskDocument."
        )
        raise TypeError(message)

    if not all(
        isinstance(label_task_document, GpawScfChargemolTaskDocument)
        for label_task_document in label_task_documents
    ):
        message = (
            "QET fine-tuning input must contain only "
            "GpawScfChargemolTaskDocument instances."
        )
        raise TypeError(message)

    return label_task_documents


def _validate_qet_firework_payload_size(qet_fine_tuning_job: Job) -> None:
    """Reject QET jobs that cannot fit in a MongoDB-backed FireWorks document."""
    firework_payload = job_to_firework(qet_fine_tuning_job).to_dict()
    payload_size = len(BSON.encode(firework_payload))
    if payload_size >= MONGODB_MAX_SAFE_DOCUMENT_BYTES:
        msg = (
            "QET fine-tuning FireWorks payload exceeds the safe MongoDB document "
            f"budget ({payload_size} >= {MONGODB_MAX_SAFE_DOCUMENT_BYTES} bytes). Use a stored "
            "JobStore label query instead of inline labels."
        )
        raise QetFineTuningStorageError(msg)


def _create_qet_fine_tuning_record_or_failure(
    *, task_document: GpawScfChargemolTaskDocument
) -> tuple[QetFineTuningRecord | None, str | None]:
    """Create a training record or reject an incomplete successful label."""
    single_point_properties: SinglePointProperties = task_document.properties  # type: ignore[assignment]

    try:
        return (
            QetFineTuningRecord.from_single_point_properties(single_point_properties),
            None,
        )
    except (ValidationError, ValueError) as exc:
        return None, str(exc)
