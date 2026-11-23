"""Run direct in-memory QET fine-tuning with MatGL and Lightning."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from pickle import UnpicklingError
from types import FrameType
from typing import Any, Literal

import lightning
import matgl
import numpy as np
import torch
from bson import BSON
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason
from matgl.apps.pes import Potential
from matgl.models import QET
from matgl.utils.training import PotentialLightningModule
from monty.json import jsanitize
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from crystal.calculators.mlip.qet_finetuning_dataset import (
    QetFineTuningGraphSettings,
    QetFineTuningRecordSplits,
    create_qet_fine_tuning_data_loaders,
    split_qet_fine_tuning_records,
)
from crystal.calculators.mlip.qet_finetuning_input import QetFineTuningInputData
from crystal.constants.mlip import QET_FINE_TUNING_MAX_FAILURE_DETAILS
from crystal.constants.storage import MONGODB_MAX_SAFE_DOCUMENT_BYTES
from crystal.schemas.gpaw_isolated_atom_reference import IsolatedAtomEnergyReference
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.qet_finetuning import (
    QetFineTuningCheckpointRef,
    QetFineTuningRecord,
    QetFineTuningResult,
    QetFineTuningSplitCounts,
    QetFineTuningStatus,
    QetFineTuningTrainingSettings,
)

QET_FINE_TUNING_VALIDATION_METRIC = "val_Total_Loss"
QET_FINE_TUNING_VALIDATION_MODE = "min"


class QetFineTuningTrainingDataError(ValueError):
    """Raised when direct QET fine-tuning records cannot form a training run."""


class QetFineTuningModelError(TypeError):
    """Raised when the configured pretrained model is not a MatGL QET potential."""


class QetFineTuningDeviceError(RuntimeError):
    """Raised when the requested QET fine-tuning device is unavailable."""


class QetFineTuningTrainingError(RuntimeError):
    """Raised when QET fine-tuning cannot produce a valid training outcome."""


class QetFineTuningCheckpointError(RuntimeError):
    """Raised when QET fine-tuning does not produce a usable checkpoint."""


class QetFineTuningStorageError(RuntimeError):
    """Raised when QET workflow data cannot fit in a persisted document."""


class QetFineTuningWalltimeCallback(Callback):
    """Turn SLURM's pre-timeout SIGTERM into a graceful Lightning stop."""

    def __init__(self) -> None:
        """Initialize a callback that has not yet installed a signal handler."""
        super().__init__()
        self.interrupted = False
        self._trainer: lightning.Trainer | None = None
        self._signal_installed = False
        self._previous_sigterm_handler: (
            signal.Handlers | int | Callable[[int, FrameType | None], object] | None
        ) = None

    def on_fit_start(self, trainer: lightning.Trainer, _pl_module: Any) -> None:
        """Install the process-local handler for the duration of trainer.fit."""
        self._trainer = trainer
        if threading.current_thread() is not threading.main_thread():
            return
        self._previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        self._signal_installed = True

    def on_fit_end(self, _trainer: lightning.Trainer, _pl_module: Any) -> None:
        """Restore the caller's SIGTERM handler after a normal fit."""
        self._restore_sigterm_handler()

    def on_exception(
        self, _trainer: lightning.Trainer, _pl_module: Any, _exception: BaseException
    ) -> None:
        """Restore the caller's SIGTERM handler when fit raises."""
        self._restore_sigterm_handler()

    def _handle_sigterm(self, _signum: int, _frame: FrameType | None) -> None:
        """Request a batch-boundary stop without raising through Lightning."""
        self.interrupted = True
        if self._trainer is not None:
            self._trainer.should_stop = True

    def _restore_sigterm_handler(self) -> None:
        """Restore the handler that was active before this fit."""
        if not self._signal_installed:
            return
        if self._previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._previous_sigterm_handler)
        self._signal_installed = False


class QetFineTuningRunner(BaseModel):
    """Own one complete in-memory QET fine-tuning run."""

    model_config = ConfigDict(frozen=True)

    qet_fine_tuning_input_data: QetFineTuningInputData
    isolated_atom_reference: IsolatedAtomEnergyReference
    training_settings: QetFineTuningTrainingSettings
    use_gpu: bool
    output_directory: Path
    training_walltime_seconds: float | None = Field(default=None, gt=0)
    checkpoint: QetFineTuningCheckpointRef | None = None

    def run(self) -> QetFineTuningResult:
        """Train, optionally test, and export the best-validation QET checkpoint."""
        qet_potential = _load_qet_potential(
            model_path=self.training_settings.model.model_path
        )
        resume_checkpoint_path = self._resolve_checkpoint_path(self.checkpoint)
        qet_model = _require_qet_model(qet_potential)
        graph_settings = QetFineTuningGraphSettings.from_qet_model(qet_model)
        record_splits = self._split_and_validate_qet_fine_tuning_records(
            supported_element_types=graph_settings.element_types
        )
        training_element_references = _build_qet_element_references(
            qet_potential=qet_potential,
            element_types=graph_settings.element_types,
            isolated_atom_reference=self.isolated_atom_reference,
        )
        trainer_accelerator, trainer_device_count = (
            _determine_lightning_trainer_devices(use_gpu=self.use_gpu)
        )

        data_loaders = create_qet_fine_tuning_data_loaders(
            record_splits=record_splits,
            graph_settings=graph_settings,
            batch_size=self.training_settings.batch_size,
            pin_memory=trainer_accelerator == "gpu",
            training_shuffle_seed=self.training_settings.split.seed,
        )
        lightning_module = self._create_qet_fine_tuning_lightning_module(
            qet_potential=qet_potential,
            qet_model=qet_model,
            training_element_references=training_element_references,
        )
        checkpoint_callback = self._create_checkpoint_callback()
        early_stopping_callback = self._create_early_stopping_callback()
        walltime_callback = QetFineTuningWalltimeCallback()
        walltime_timer = self._create_walltime_timer()
        trainer = self._create_lightning_trainer(
            accelerator=trainer_accelerator,
            device_count=trainer_device_count,
            checkpoint_callback=checkpoint_callback,
            early_stopping_callback=early_stopping_callback,
            walltime_callback=walltime_callback,
            walltime_timer=walltime_timer,
        )
        trainer.fit(
            model=lightning_module,
            train_dataloaders=data_loaders.training_data_loader,
            val_dataloaders=data_loaders.validation_data_loader,
            ckpt_path=(
                resume_checkpoint_path.as_posix()
                if resume_checkpoint_path is not None
                else None
            ),
        )
        walltime_interrupted = walltime_callback.interrupted or _walltime_was_reached(
            walltime_timer
        )
        selected_checkpoint_path = _require_checkpoint_path(
            checkpoint_callback=checkpoint_callback,
            allow_last_checkpoint=walltime_interrupted,
        )
        run_status = self._get_run_status(
            early_stopping_callback=early_stopping_callback,
            walltime_interrupted=walltime_interrupted,
        )
        checkpoint = self._read_checkpoint(selected_checkpoint_path)
        evaluation_results: list[Mapping[str, Any]] = []
        if data_loaders.test_data_loader is not None:
            evaluation_results = trainer.test(
                model=lightning_module,
                dataloaders=data_loaders.test_data_loader,
                ckpt_path=selected_checkpoint_path.as_posix(),
                verbose=False,
            )
        exported_model_path = self._export_qet_potential_from_checkpoint(
            qet_potential=qet_potential,
            qet_model=qet_model,
            training_element_references=training_element_references,
            checkpoint_path=selected_checkpoint_path,
        )
        return self._build_result(
            record_splits=record_splits,
            evaluation_results=evaluation_results,
            checkpoint=checkpoint,
            exported_model_path=exported_model_path,
            isolated_atom_reference=self.isolated_atom_reference,
            status=run_status,
        )

    def _create_checkpoint_callback(self) -> ModelCheckpoint:
        """Create the callback that selects the best validation checkpoint."""
        checkpoint_directory = self.output_directory / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        return ModelCheckpoint(
            dirpath=checkpoint_directory,
            filename=f"{{epoch:03d}}-{{{QET_FINE_TUNING_VALIDATION_METRIC}:.6f}}",
            monitor=QET_FINE_TUNING_VALIDATION_METRIC,
            mode=QET_FINE_TUNING_VALIDATION_MODE,
            save_top_k=1,
            save_last=True,
            save_on_exception=True,
        )

    def _create_early_stopping_callback(self) -> EarlyStopping:
        """Create validation-loss early stopping for one fine-tuning run."""
        return EarlyStopping(
            monitor=QET_FINE_TUNING_VALIDATION_METRIC,
            mode=QET_FINE_TUNING_VALIDATION_MODE,
            patience=self.training_settings.early_stopping_patience,
            min_delta=self.training_settings.early_stopping_min_delta,
        )

    def _create_lightning_trainer(
        self,
        *,
        accelerator: Literal["cpu", "gpu"],
        device_count: int,
        checkpoint_callback: ModelCheckpoint,
        early_stopping_callback: EarlyStopping,
        walltime_callback: QetFineTuningWalltimeCallback,
        walltime_timer: Timer | None,
    ) -> lightning.Trainer:
        """Create the Lightning trainer for one fine-tuning run."""
        callbacks: list[Callback] = [
            checkpoint_callback,
            early_stopping_callback,
            walltime_callback,
        ]
        if walltime_timer is not None:
            callbacks.append(walltime_timer)
        return lightning.Trainer(
            max_epochs=self.training_settings.max_epochs,
            accelerator=accelerator,
            devices=device_count,
            precision="32-true",
            callbacks=callbacks,
            inference_mode=False,
            default_root_dir=self.output_directory,
        )

    def _create_walltime_timer(self) -> Timer | None:
        """Create a step-level timer when the maker supplied a safe budget."""
        if self.training_walltime_seconds is None:
            return None
        return Timer(
            duration=timedelta(seconds=self.training_walltime_seconds),
            interval="step",
            verbose=False,
        )

    @staticmethod
    def _resolve_checkpoint_path(
        checkpoint: QetFineTuningCheckpointRef | None,
    ) -> Path | None:
        """Resolve a checkpoint reference to an existing shared-filesystem file."""
        if checkpoint is None:
            return None
        checkpoint_path = checkpoint.path

        if not checkpoint_path.is_file():
            msg = (
                "QET continuation checkpoint does not exist on the shared "
                f"filesystem: {checkpoint_path}."
            )
            raise QetFineTuningCheckpointError(msg)
        return checkpoint_path

    @staticmethod
    def _get_run_status(
        *, early_stopping_callback: EarlyStopping, walltime_interrupted: bool = False
    ) -> QetFineTuningStatus:
        """Derive the run outcome after trainer.fit."""
        if walltime_interrupted:
            return QetFineTuningStatus.walltime_interrupted
        stopping_reason = early_stopping_callback.stopping_reason
        if stopping_reason is EarlyStoppingReason.NOT_STOPPED:
            return QetFineTuningStatus.completed
        if stopping_reason is not EarlyStoppingReason.PATIENCE_EXHAUSTED:
            msg = (
                "QET fine-tuning stopped for an unsupported early-stopping reason: "
                f"{stopping_reason.name}."
            )
            raise QetFineTuningTrainingError(msg)
        return QetFineTuningStatus.early_stopped

    @staticmethod
    def _read_checkpoint(checkpoint_path: Path) -> QetFineTuningCheckpointRef:
        """Read the training position from a selected checkpoint file."""
        try:
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        except (EOFError, OSError, RuntimeError, UnpicklingError) as exc:
            msg = f"Unable to read QET checkpoint metadata from {checkpoint_path}."
            raise QetFineTuningCheckpointError(msg) from exc

        if not isinstance(checkpoint_payload, Mapping):
            msg = f"QET checkpoint metadata is missing or invalid in {checkpoint_path}."
            raise QetFineTuningCheckpointError(msg)
        try:
            return QetFineTuningCheckpointRef(
                path=checkpoint_path,
                epoch=checkpoint_payload["epoch"],
                global_step=checkpoint_payload["global_step"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"QET checkpoint metadata is missing or invalid in {checkpoint_path}."
            raise QetFineTuningCheckpointError(msg) from exc

    def _build_result(
        self,
        *,
        record_splits: QetFineTuningRecordSplits,
        evaluation_results: list[Mapping[str, Any]],
        checkpoint: QetFineTuningCheckpointRef,
        exported_model_path: Path,
        isolated_atom_reference: IsolatedAtomEnergyReference,
        status: QetFineTuningStatus,
    ) -> QetFineTuningResult:
        """Build the bounded result for one fine-tuning run outcome."""
        failed_label_details = self.qet_fine_tuning_input_data.failed_label_details[
            :QET_FINE_TUNING_MAX_FAILURE_DETAILS
        ]
        result = QetFineTuningResult(
            status=status,
            input_document_count=self.qet_fine_tuning_input_data.input_document_count,
            accepted_label_count=self.qet_fine_tuning_input_data.accepted_label_count,
            failed_label_count=self.qet_fine_tuning_input_data.failed_label_count,
            failed_label_details=failed_label_details,
            failed_label_details_truncated=(
                len(failed_label_details)
                < self.qet_fine_tuning_input_data.failed_label_count
            ),
            split_counts=QetFineTuningSplitCounts(
                train_count=len(record_splits.train_records),
                validation_count=len(record_splits.validation_records),
                test_count=len(record_splits.test_records),
            ),
            test_metrics=_convert_lightning_test_results_to_scalar_metrics(
                lightning_test_results=evaluation_results
            ),
            checkpoint=checkpoint,
            model=MlipModelRef(path=exported_model_path),
            isolated_atom_reference=isolated_atom_reference,
        )
        result_payload = jsanitize(
            result.as_dict(), strict=True, enum_values=True, allow_bson=True
        )
        result_size = len(BSON.encode({"output": result_payload}))
        if result_size >= MONGODB_MAX_SAFE_DOCUMENT_BYTES:
            msg = (
                "QET fine-tuning result exceeds the safe MongoDB document budget "
                f"({result_size} >= {MONGODB_MAX_SAFE_DOCUMENT_BYTES} bytes)."
            )
            raise QetFineTuningStorageError(msg)
        return result

    def _split_and_validate_qet_fine_tuning_records(
        self, *, supported_element_types: tuple[str, ...]
    ) -> QetFineTuningRecordSplits:
        """Validate model compatibility and create record splits."""
        _validate_record_elements(
            records=self.qet_fine_tuning_input_data.accepted_records,
            supported_element_types=supported_element_types,
        )
        _validate_isolated_atom_reference_coverage(
            records=self.qet_fine_tuning_input_data.accepted_records,
            isolated_atom_reference=self.isolated_atom_reference,
        )

        record_splits = split_qet_fine_tuning_records(
            self.qet_fine_tuning_input_data.accepted_records,
            self.training_settings.split,
        )
        _validate_training_split_element_coverage(record_splits=record_splits)

        return record_splits

    def _create_qet_fine_tuning_lightning_module(
        self,
        *,
        qet_potential: Potential,
        qet_model: QET,
        training_element_references: NDArray[np.float64],
    ) -> PotentialLightningModule:
        """Create the optimizer, scheduler, and MatGL Lightning module."""
        optimizer = torch.optim.AdamW(
            qet_model.parameters(),
            lr=self.training_settings.learning_rate,
            weight_decay=self.training_settings.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_settings.effective_scheduler_t_max,
            eta_min=self.training_settings.scheduler_eta_min,
        )

        return PotentialLightningModule(
            model=qet_model,
            element_refs=training_element_references,
            data_mean=qet_potential.data_mean,
            data_std=qet_potential.data_std,
            energy_weight=self.training_settings.energy_weight,
            force_weight=self.training_settings.force_weight,
            stress_weight=self.training_settings.stress_weight,
            charge_weight=self.training_settings.charge_weight,
            loss=self.training_settings.loss,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def _export_qet_potential_from_checkpoint(
        self,
        *,
        qet_potential: Potential,
        qet_model: QET,
        training_element_references: NDArray[np.float64],
        checkpoint_path: Path,
    ) -> Path:
        """Load a selected checkpoint and export a QET Potential."""
        qet_model.to("cpu")

        checkpoint_lightning_module = PotentialLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path.as_posix(),
            model=qet_model,
            element_refs=training_element_references,
            map_location="cpu",
        )
        checkpoint_lightning_module.to("cpu")

        exported_potential = Potential(
            model=checkpoint_lightning_module.model.model,  # type: ignore[arg-type]
            data_mean=qet_potential.data_mean,
            data_std=qet_potential.data_std,
            element_refs=training_element_references,
            calc_forces=qet_potential.calc_forces,
            calc_stresses=qet_potential.calc_stresses,
            calc_hessian=qet_potential.calc_hessian,
            calc_magmom=qet_potential.calc_magmom,
            calc_charge=True,
            calc_repuls=qet_potential.calc_repuls,
        )
        exported_model_path = self.output_directory / "model"
        exported_potential.save(exported_model_path)
        _validate_exported_qet_potential(exported_model_path)

        return exported_model_path


def _load_qet_potential(*, model_path: str | Path) -> Potential:
    """Load a MatGL Potential."""
    loaded_qet_potential = matgl.load_model(model_path)

    if not isinstance(loaded_qet_potential, Potential):
        message = (
            "QET fine-tuning requires a MatGL Potential: "
            f"loaded_type={type(loaded_qet_potential).__name__}."
        )
        raise QetFineTuningModelError(message)

    return loaded_qet_potential


def _require_qet_model(qet_potential: Potential) -> QET:
    """Return a Potential's QET model or reject an incompatible Potential."""
    qet_model = qet_potential.model

    if not isinstance(qet_model, QET):
        message = (
            "QET fine-tuning requires a MatGL Potential containing a QET model: "
            f"loaded_model_type={type(qet_model).__name__}."
        )
        raise QetFineTuningModelError(message)

    return qet_model


def _validate_exported_qet_potential(model_path: Path) -> None:
    """Require an exported model to be a reloadable, charge-capable QET Potential."""
    try:
        exported_potential = _load_qet_potential(model_path=model_path)
        _require_qet_model(exported_potential)
    except (OSError, RuntimeError, ValueError) as exc:
        message = f"QET fine-tuning exported an invalid MatGL model: {model_path}."
        raise QetFineTuningModelError(message) from exc

    if not exported_potential.calc_charge:
        message = (
            "QET fine-tuning exported a Potential without charge prediction enabled: "
            f"{model_path}."
        )
        raise QetFineTuningModelError(message)


def _validate_record_elements(
    *, records: Sequence[QetFineTuningRecord], supported_element_types: tuple[str, ...]
) -> None:
    """Reject records containing elements unsupported by the QET model."""
    supported_element_type_set = set(supported_element_types)

    unsupported_element_symbols = sorted(
        {
            site.specie.symbol
            for record in records
            for site in record.structure
            if site.specie.symbol not in supported_element_type_set
        }
    )

    if unsupported_element_symbols:
        message = (
            "QET model does not support training elements: "
            f"{unsupported_element_symbols}."
        )
        raise QetFineTuningTrainingDataError(message)


def _validate_training_split_element_coverage(
    *, record_splits: QetFineTuningRecordSplits
) -> None:
    """Require training records to contain every element used for evaluation."""
    training_element_symbols = {
        site.specie.symbol
        for training_record in record_splits.train_records
        for site in training_record.structure
    }
    evaluation_element_symbols = {
        site.specie.symbol
        for evaluation_record in (
            *record_splits.validation_records,
            *record_splits.test_records,
        )
        for site in evaluation_record.structure
    }
    missing_training_element_symbols = sorted(
        evaluation_element_symbols - training_element_symbols
    )

    if missing_training_element_symbols:
        message = (
            "Insufficient training coverage for evaluation elements: "
            f"{missing_training_element_symbols}. Add more input structures "
            "containing them or try another split seed."
        )
        raise QetFineTuningTrainingDataError(message)


def _validate_isolated_atom_reference_coverage(
    *,
    records: Sequence[QetFineTuningRecord],
    isolated_atom_reference: IsolatedAtomEnergyReference,
) -> None:
    """Require a physical reference for every element used in training."""
    record_element_symbols = {
        site.specie.symbol for record in records for site in record.structure
    }
    missing_reference_elements = sorted(
        record_element_symbols - isolated_atom_reference.elements
    )
    if missing_reference_elements:
        message = (
            "QET fine-tuning isolated-atom reference is missing training elements: "
            f"{missing_reference_elements}."
        )
        raise QetFineTuningTrainingDataError(message)


def _build_qet_element_references(
    *,
    qet_potential: Potential,
    element_types: tuple[str, ...],
    isolated_atom_reference: IsolatedAtomEnergyReference,
) -> NDArray[np.float64]:
    """Replace target QET element references with physical isolated-atom energies."""
    pretrained_element_references = qet_potential.element_refs
    if pretrained_element_references is None:
        element_references = np.zeros(len(element_types), dtype=float)
    else:
        element_references = (
            pretrained_element_references.property_offset.detach().cpu().numpy()
        )
        if element_references.ndim != 1 or len(element_references) != len(
            element_types
        ):
            msg = (
                "Pretrained QET element references must be one value per model element."
            )
            raise QetFineTuningModelError(msg)
        element_references = np.asarray(element_references, dtype=float).copy()

    element_index_by_type = {
        element: index for index, element in enumerate(element_types)
    }
    unsupported_reference_elements = sorted(
        isolated_atom_reference.elements - set(element_index_by_type)
    )
    if unsupported_reference_elements:
        msg = (
            "QET model does not support isolated-atom reference elements: "
            f"{unsupported_reference_elements}."
        )
        raise QetFineTuningTrainingDataError(msg)
    for element, energy in isolated_atom_reference.energies_by_element.items():
        element_references[element_index_by_type[element]] = energy
    return element_references


def _determine_lightning_trainer_devices(
    *, use_gpu: bool
) -> tuple[Literal["cpu", "gpu"], int]:
    """Translate the selected device into validated Lightning settings."""
    if not use_gpu:
        return "cpu", 1

    if not torch.cuda.is_available():
        message = "QET fine-tuning requested a GPU, but CUDA is unavailable."
        raise QetFineTuningDeviceError(message)

    return "gpu", 1


def _walltime_was_reached(walltime_timer: Timer | None) -> bool:
    """Return whether the configured training timer stopped the fit."""
    if walltime_timer is None:
        return False
    remaining_seconds = walltime_timer.time_remaining()
    return remaining_seconds is not None and remaining_seconds <= 0


def _require_checkpoint_path(
    *, checkpoint_callback: ModelCheckpoint, allow_last_checkpoint: bool
) -> Path:
    """Return the best checkpoint, or the latest checkpoint after walltime."""
    checkpoint_description = "best-validation"
    if allow_last_checkpoint:
        checkpoint_path_string = checkpoint_callback.last_model_path
        checkpoint_description = "latest"
    else:
        checkpoint_path_string = checkpoint_callback.best_model_path

    if not checkpoint_path_string:
        message = (
            "QET fine-tuning completed without producing a "
            f"{checkpoint_description} checkpoint."
        )
        raise QetFineTuningCheckpointError(message)

    checkpoint_path = Path(checkpoint_path_string)

    if not checkpoint_path.is_file():
        message = (
            f"QET fine-tuning selected a {checkpoint_description} checkpoint "
            "that does not exist: "
            f"{checkpoint_path}."
        )
        raise QetFineTuningCheckpointError(message)

    return checkpoint_path


def _convert_lightning_test_results_to_scalar_metrics(
    *, lightning_test_results: list[Mapping[str, Any]]
) -> dict[str, float]:
    """Convert the first Lightning test result mapping into plain floats."""
    if not lightning_test_results:
        return {}

    first_lightning_test_result = lightning_test_results[0]

    return {
        metric_name: float(metric_value)
        for metric_name, metric_value in first_lightning_test_result.items()
    }
