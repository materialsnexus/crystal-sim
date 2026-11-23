"""Tests for direct in-memory QET fine-tuning execution."""

from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch, sentinel

import matgl
import numpy as np
import pytest
import torch
from matgl.apps.pes import Potential
from matgl.models import QET

from crystal.calculators.mlip.qet_finetuning_dataset import QetFineTuningRecordSplits
from crystal.calculators.mlip.qet_finetuning_input import QetFineTuningInputData
from crystal.calculators.mlip.qet_finetuning_runner import (
    QetFineTuningCheckpointError,
    QetFineTuningDeviceError,
    QetFineTuningModelError,
    QetFineTuningRunner,
    QetFineTuningTrainingDataError,
    QetFineTuningWalltimeCallback,
    _require_checkpoint_path,
)
from crystal.constants.mlip import QET_FINE_TUNING_MAX_FAILURE_DETAILS
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.gpaw import (
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
)
from crystal.schemas.gpaw_isolated_atom_reference import (
    IsolatedAtomEnergy,
    IsolatedAtomEnergyReference,
)
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.qet_finetuning import (
    QetFineTuningCheckpointRef,
    QetFineTuningLabelFailure,
    QetFineTuningRecord,
    QetFineTuningStatus,
    QetFineTuningTrainingSettings,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.qet_finetuning import create_qet_fine_tuning_record

TEST_GPAW_VERSION = "25.7.0"
TEST_SILICON_SETUP_DIGEST = "a" * 64
TEST_SILICON_ISOLATED_ATOM_ENERGY = -5.0


def _isolated_atom_reference() -> IsolatedAtomEnergyReference:
    runtime_provenance = GpawRuntimeProvenance(
        gpaw_version=TEST_GPAW_VERSION,
        atomic_data=(
            GpawAtomicDataInfo(
                element_symbol="Si", paw_setup_sha256_digest=TEST_SILICON_SETUP_DIGEST
            ),
        ),
    )
    return IsolatedAtomEnergyReference(
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
        isolated_atom_energies=(
            IsolatedAtomEnergy(
                element="Si",
                energy=TEST_SILICON_ISOLATED_ATOM_ENERGY,
                runtime_provenance=runtime_provenance,
            ),
        ),
    )


def _create_qet_fine_tuning_runner(
    tmp_path: Path,
    *,
    accepted_records: list[QetFineTuningRecord],
    use_gpu: bool = False,
    test_fraction: float = 0.2,
) -> QetFineTuningRunner:
    pretrained_model_directory = tmp_path / "pretrained-model"
    pretrained_model_directory.mkdir(exist_ok=True)
    train_fraction = 0.6

    return QetFineTuningRunner(
        qet_fine_tuning_input_data=QetFineTuningInputData(
            accepted_records=tuple(accepted_records)
        ),
        isolated_atom_reference=_isolated_atom_reference(),
        training_settings=QetFineTuningTrainingSettings(
            model=MlipModelRef(path=pretrained_model_directory),
            split={
                "train_fraction": train_fraction,
                "validation_fraction": 1.0 - train_fraction - test_fraction,
                "test_fraction": test_fraction,
                "seed": 19,
            },
            max_epochs=1,
        ),
        use_gpu=use_gpu,
        output_directory=tmp_path / "output",
    )


def test_walltime_callback_installs_and_restores_sigterm_handler() -> None:
    callback = QetFineTuningWalltimeCallback()
    previous_handler = signal.getsignal(signal.SIGTERM)
    trainer = MagicMock(should_stop=False)

    callback.on_fit_start(trainer, MagicMock())
    assert signal.getsignal(signal.SIGTERM) == callback._handle_sigterm

    signal.raise_signal(signal.SIGTERM)
    assert callback.interrupted
    assert trainer.should_stop

    callback.on_fit_end(trainer, MagicMock())
    assert signal.getsignal(signal.SIGTERM) == previous_handler


def test_walltime_checkpoint_selection_prefers_latest_checkpoint(
    tmp_path: Path,
) -> None:
    best_checkpoint_path = tmp_path / "best.ckpt"
    last_checkpoint_path = tmp_path / "last.ckpt"
    best_checkpoint_path.touch()
    last_checkpoint_path.touch()
    callback = MagicMock(
        best_model_path=best_checkpoint_path.as_posix(),
        last_model_path=last_checkpoint_path.as_posix(),
    )

    selected_path = _require_checkpoint_path(
        checkpoint_callback=callback, allow_last_checkpoint=True
    )

    assert selected_path == last_checkpoint_path


def _create_pretrained_qet_potential(
    *,
    element_types: tuple[str, ...] = ("Si",),
    data_mean: float = 0.0,
    data_std: float = 1.0,
) -> Potential:
    qet_model = QET(element_types=element_types, cutoff=5.0)

    pretrained_qet_potential = MagicMock(spec=Potential)
    pretrained_qet_potential.model = qet_model
    pretrained_qet_potential.element_refs = None
    pretrained_qet_potential.data_mean = torch.tensor(data_mean)
    pretrained_qet_potential.data_std = torch.tensor(data_std)
    pretrained_qet_potential.calc_forces = True
    pretrained_qet_potential.calc_stresses = True
    pretrained_qet_potential.calc_hessian = False
    pretrained_qet_potential.calc_magmom = False
    pretrained_qet_potential.calc_repuls = False
    pretrained_qet_potential.calc_charge = True

    return pretrained_qet_potential


def test_runner_passes_explicit_matgl_defaults_to_training_components(
    tmp_path: Path,
) -> None:
    pretrained_qet_potential = _create_pretrained_qet_potential()
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path, accepted_records=[create_qet_fine_tuning_record(1.0)]
    )
    qet_model = cast("QET", pretrained_qet_potential.model)

    with (
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.PotentialLightningModule"
        ) as lightning_module,
        patch("crystal.calculators.mlip.qet_finetuning_runner.torch.optim.AdamW"),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.torch.optim.lr_scheduler.CosineAnnealingLR"
        ) as cosine_scheduler,
    ):
        qet_fine_tuning_runner._create_qet_fine_tuning_lightning_module(
            qet_potential=pretrained_qet_potential,
            qet_model=qet_model,
            training_element_references=np.asarray([0.0]),
        )

    assert lightning_module.call_args.kwargs["loss"] == "huber_loss"
    assert (
        cosine_scheduler.call_args.kwargs["T_max"]
        == qet_fine_tuning_runner.training_settings.max_epochs
    )


def test_runner_rejects_loaded_object_that_is_not_matgl_potential(
    tmp_path: Path,
) -> None:
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path, accepted_records=[create_qet_fine_tuning_record(1.0)]
    )

    with (
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.matgl.load_model",
            return_value=object(),
        ),
        pytest.raises(QetFineTuningModelError, match="MatGL Potential"),
    ):
        qet_fine_tuning_runner.run()


def test_runner_rejects_potential_without_qet_model(tmp_path: Path) -> None:
    pretrained_non_qet_potential = MagicMock(spec=Potential)
    pretrained_non_qet_potential.model = object()

    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path, accepted_records=[create_qet_fine_tuning_record(1.0)]
    )

    with (
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.matgl.load_model",
            return_value=pretrained_non_qet_potential,
        ),
        pytest.raises(QetFineTuningModelError, match="QET model"),
    ):
        qet_fine_tuning_runner.run()


def test_runner_rejects_training_elements_unsupported_by_qet_model(
    tmp_path: Path,
) -> None:
    pretrained_qet_potential = _create_pretrained_qet_potential(element_types=("H",))
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path, accepted_records=[create_qet_fine_tuning_record(1.0)]
    )

    with (
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.matgl.load_model",
            return_value=pretrained_qet_potential,
        ),
        pytest.raises(QetFineTuningTrainingDataError, match="Si"),
    ):
        qet_fine_tuning_runner.run()


def test_runner_rejects_training_elements_missing_from_isolated_atom_reference(
    tmp_path: Path,
) -> None:
    pretrained_qet_potential = _create_pretrained_qet_potential(
        element_types=("Si", "O")
    )
    oxygen_containing_structure = TEST_SI_STRUCTURE.copy()
    oxygen_containing_structure.replace(0, "O")
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path,
        accepted_records=[
            create_qet_fine_tuning_record(
                float(record_index), structure=oxygen_containing_structure
            )
            for record_index in range(5)
        ],
    )

    with (
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.matgl.load_model",
            return_value=pretrained_qet_potential,
        ),
        pytest.raises(QetFineTuningTrainingDataError, match="reference"),
    ):
        qet_fine_tuning_runner.run()


def test_runner_rejects_evaluation_elements_missing_from_training(
    tmp_path: Path,
) -> None:
    pretrained_qet_potential = _create_pretrained_qet_potential(
        element_types=("Si", "O")
    )
    oxygen_containing_structure = TEST_SI_STRUCTURE.copy()
    oxygen_containing_structure.replace(0, "O")
    records = [
        create_qet_fine_tuning_record(
            float(record_index),
            structure=(
                oxygen_containing_structure if record_index == 4 else TEST_SI_STRUCTURE
            ),
        )
        for record_index in range(5)
    ]
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path, accepted_records=records
    )

    with (
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.matgl.load_model",
            return_value=pretrained_qet_potential,
        ),
        pytest.raises(QetFineTuningTrainingDataError, match="O"),
    ):
        qet_fine_tuning_runner.run()


def test_runner_rejects_gpu_allocation_when_cuda_is_unavailable(tmp_path: Path) -> None:
    pretrained_qet_potential = _create_pretrained_qet_potential()
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path,
        accepted_records=[
            create_qet_fine_tuning_record(float(record_index))
            for record_index in range(5)
        ],
        use_gpu=True,
    )

    with (
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.matgl.load_model",
            return_value=pretrained_qet_potential,
        ),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.torch.cuda.is_available",
            return_value=False,
        ),
        pytest.raises(QetFineTuningDeviceError, match="CUDA"),
    ):
        qet_fine_tuning_runner.run()


def test_runner_fails_when_training_does_not_produce_best_checkpoint(
    tmp_path: Path,
) -> None:
    pretrained_qet_potential = _create_pretrained_qet_potential()
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path,
        accepted_records=[
            create_qet_fine_tuning_record(float(record_index))
            for record_index in range(5)
        ],
    )

    qet_fine_tuning_data_loaders = SimpleNamespace(
        training_data_loader=sentinel.training_data_loader,
        validation_data_loader=(sentinel.validation_data_loader),
        test_data_loader=sentinel.test_data_loader,
    )
    qet_fine_tuning_lightning_module = MagicMock()
    qet_fine_tuning_lightning_trainer = MagicMock()
    best_validation_checkpoint_callback = MagicMock(best_model_path="")

    with (
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.matgl.load_model",
            return_value=pretrained_qet_potential,
        ),
        patch(
            "crystal.calculators.mlip."
            "qet_finetuning_runner."
            "create_qet_fine_tuning_data_loaders",
            return_value=qet_fine_tuning_data_loaders,
        ),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.PotentialLightningModule",
            return_value=(qet_fine_tuning_lightning_module),
        ),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.ModelCheckpoint",
            return_value=(best_validation_checkpoint_callback),
        ),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.lightning.Trainer",
            return_value=(qet_fine_tuning_lightning_trainer),
        ),
        pytest.raises(QetFineTuningCheckpointError, match="without producing"),
    ):
        qet_fine_tuning_runner.run()


def test_runner_returns_walltime_interrupted_result_after_sigterm(
    tmp_path: Path,
) -> None:
    pretrained_qet_potential = _create_pretrained_qet_potential()
    records = [
        create_qet_fine_tuning_record(float(record_index)) for record_index in range(2)
    ]
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path, accepted_records=records
    )
    record_splits = QetFineTuningRecordSplits(
        train_records=(records[0],), validation_records=(records[1],), test_records=()
    )
    qet_fine_tuning_data_loaders = SimpleNamespace(
        training_data_loader=sentinel.training_data_loader,
        validation_data_loader=sentinel.validation_data_loader,
        test_data_loader=None,
    )
    latest_checkpoint_path = tmp_path / "last.ckpt"
    latest_checkpoint_path.touch()
    latest_checkpoint = QetFineTuningCheckpointRef(
        path=latest_checkpoint_path, epoch=1, global_step=2
    )
    checkpoint_callback = MagicMock(
        best_model_path="", last_model_path=latest_checkpoint_path.as_posix()
    )
    early_stopping_callback = MagicMock()
    lightning_trainer = MagicMock()

    def build_lightning_trainer(*, callbacks: list[object], **_: object) -> MagicMock:
        walltime_callback = next(
            callback
            for callback in callbacks
            if isinstance(callback, QetFineTuningWalltimeCallback)
        )

        def interrupt_training(*_: object, **__: object) -> None:
            walltime_callback._handle_sigterm(signal.SIGTERM, None)

        lightning_trainer.fit.side_effect = interrupt_training
        return lightning_trainer

    with (
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.matgl.load_model",
            return_value=pretrained_qet_potential,
        ),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.create_qet_fine_tuning_data_loaders",
            return_value=qet_fine_tuning_data_loaders,
        ),
        patch.object(
            QetFineTuningRunner,
            "_split_and_validate_qet_fine_tuning_records",
            return_value=record_splits,
        ),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner._build_qet_element_references",
            return_value=np.asarray([0.0]),
        ),
        patch.object(
            QetFineTuningRunner,
            "_create_qet_fine_tuning_lightning_module",
            return_value=MagicMock(),
        ),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.ModelCheckpoint",
            return_value=checkpoint_callback,
        ),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.EarlyStopping",
            return_value=early_stopping_callback,
        ),
        patch(
            "crystal.calculators.mlip.qet_finetuning_runner.lightning.Trainer",
            side_effect=build_lightning_trainer,
        ),
        patch.object(
            QetFineTuningRunner, "_read_checkpoint", return_value=latest_checkpoint
        ),
        patch.object(
            QetFineTuningRunner,
            "_export_qet_potential_from_checkpoint",
            return_value=tmp_path / "exported-model",
        ),
    ):
        result = qet_fine_tuning_runner.run()

    assert result.status is QetFineTuningStatus.walltime_interrupted
    assert result.checkpoint == latest_checkpoint
    assert lightning_trainer.fit.called


def test_runner_truncates_failure_details_in_persisted_result(tmp_path: Path) -> None:
    accepted_record = create_qet_fine_tuning_record(1.0)
    failure_details = tuple(
        QetFineTuningLabelFailure(input_document_index=index, reason="label failed")
        for index in range(QET_FINE_TUNING_MAX_FAILURE_DETAILS + 1)
    )
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path, accepted_records=[accepted_record]
    ).model_copy(
        update={
            "qet_fine_tuning_input_data": QetFineTuningInputData(
                accepted_records=(accepted_record, accepted_record),
                failed_label_details=failure_details,
            )
        }
    )
    record_splits = QetFineTuningRecordSplits(
        train_records=(accepted_record,),
        validation_records=(accepted_record,),
        test_records=(),
    )

    result = qet_fine_tuning_runner._build_result(
        record_splits=record_splits,
        evaluation_results=[],
        checkpoint=QetFineTuningCheckpointRef(
            path=tmp_path / "checkpoint.ckpt", epoch=1, global_step=1
        ),
        exported_model_path=tmp_path / "model",
        isolated_atom_reference=_isolated_atom_reference(),
        status=QetFineTuningStatus.early_stopped,
    )

    assert len(result.failed_label_details) == QET_FINE_TUNING_MAX_FAILURE_DETAILS
    assert result.failed_label_details_truncated
    assert result.failed_label_count == QET_FINE_TUNING_MAX_FAILURE_DETAILS + 1


def test_runner_minimally_trains_exports_and_reloads_qet(tmp_path: Path) -> None:
    oxidized_silicon_structure = TEST_SI_STRUCTURE.copy()
    oxidized_silicon_structure.add_oxidation_state_by_element({"Si": 4})
    qet_fine_tuning_runner = _create_qet_fine_tuning_runner(
        tmp_path,
        accepted_records=[
            create_qet_fine_tuning_record(
                float(record_index), structure=oxidized_silicon_structure
            )
            for record_index in range(5)
        ],
    )
    pretrained_qet_model = QET(element_types=("Si",), cutoff=5.0, nblocks=1, width=0.5)
    Potential(
        model=pretrained_qet_model,
        data_mean=2.5,
        data_std=0.5,
        element_refs=np.asarray([0.0]),
        calc_forces=True,
        calc_stresses=True,
        calc_charge=True,
    ).save(qet_fine_tuning_runner.training_settings.model.model_path)

    qet_fine_tuning_result = qet_fine_tuning_runner.run()

    assert qet_fine_tuning_result.status is QetFineTuningStatus.completed
    assert qet_fine_tuning_result.model is not None
    exported_potential = matgl.load_model(qet_fine_tuning_result.model.model_path)

    assert isinstance(exported_potential, Potential)
    assert isinstance(exported_potential.model, QET)
    assert exported_potential.calc_charge
    assert exported_potential.element_refs is not None
    assert torch.equal(
        exported_potential.element_refs.property_offset,
        torch.tensor([TEST_SILICON_ISOLATED_ATOM_ENERGY]),
    )
    assert (qet_fine_tuning_runner.output_directory / "lightning_logs").is_dir()
