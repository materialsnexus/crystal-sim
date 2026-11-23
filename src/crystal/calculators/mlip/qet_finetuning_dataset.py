"""Build deterministic in-memory QET fine-tuning datasets and data loaders."""

from __future__ import annotations

import random
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Self, TypedDict

import matgl
import torch
from ase.stress import voigt_6_to_full_3x3_stress
from matgl.apps.pes import EV_PER_ANG3_TO_GPA
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import collate_fn_pes
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from crystal.schemas.qet_finetuning import (
    QetFineTuningRecord,
    QetFineTuningSplitCounts,
    QetFineTuningSplitSettings,
)

if TYPE_CHECKING:
    from matgl.models import QET


class _QetFineTuningSampleLabels(TypedDict):
    """Per-structure labels consumed by MatGL's potential collator."""

    energies: torch.Tensor
    forces: torch.Tensor
    stresses: torch.Tensor
    charges: torch.Tensor


class QetFineTuningGraphSettings(BaseModel):
    """Graph-conversion settings copied from the target QET model."""

    model_config = ConfigDict(frozen=True)

    element_types: tuple[str, ...] = Field(
        min_length=1,
        description="Chemical elements supported by the pretrained QET model.",
    )
    cutoff: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Radial neighbor distance cutoff for constructing the QET atomic graph, in Å."
        ),
    )

    @classmethod
    def from_qet_model(cls, qet_model: QET) -> Self:
        """Copy graph-conversion settings from the target QET model."""
        return cls(
            element_types=tuple(qet_model.element_types), cutoff=float(qet_model.cutoff)
        )


class QetFineTuningRecordSplits(BaseModel):
    """One deterministic, disjoint train-validation-test record partition."""

    model_config = ConfigDict(frozen=True)

    train_records: tuple[QetFineTuningRecord, ...] = Field(
        min_length=1, description="Records used to optimize the QET model."
    )
    validation_records: tuple[QetFineTuningRecord, ...] = Field(
        min_length=1,
        description="Records used to select the best validation checkpoint.",
    )
    test_records: tuple[QetFineTuningRecord, ...] = Field(
        description="Records reserved for final evaluation of the selected checkpoint."
    )


class QetFineTuningDataLoaders(BaseModel):
    """Training, validation, and test loaders for one fine-tuning run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    training_data_loader: DataLoader[Any] = Field(
        description="Batched loader for the training records."
    )
    validation_data_loader: DataLoader[Any] = Field(
        description="Batched loader for the validation records."
    )
    test_data_loader: DataLoader[Any] | None = Field(
        description="Batched loader for the test records, when a test split exists."
    )


def split_qet_fine_tuning_records(
    records: Sequence[QetFineTuningRecord], settings: QetFineTuningSplitSettings
) -> QetFineTuningRecordSplits:
    """Shuffle a record snapshot once and partition it into configured splits."""
    shuffled_records = list(records)

    split_random_number_generator = random.Random(  # noqa: S311
        settings.seed
    )
    split_random_number_generator.shuffle(shuffled_records)

    split_counts = _allocate_qet_fine_tuning_split_counts(
        record_count=len(shuffled_records), settings=settings
    )
    validation_record_end_index = (
        split_counts.train_count + split_counts.validation_count
    )

    return QetFineTuningRecordSplits(
        train_records=tuple(shuffled_records[: split_counts.train_count]),
        validation_records=tuple(
            shuffled_records[split_counts.train_count : validation_record_end_index]
        ),
        test_records=tuple(shuffled_records[validation_record_end_index:]),
    )


def _allocate_qet_fine_tuning_split_counts(
    *, record_count: int, settings: QetFineTuningSplitSettings
) -> QetFineTuningSplitCounts:
    """Allocate records using the largest-remainder method."""
    split_fractions = (
        settings.train_fraction,
        settings.validation_fraction,
        settings.test_fraction,
    )
    exact_split_counts = tuple(
        record_count * split_fraction for split_fraction in split_fractions
    )
    split_counts = [int(exact_count) for exact_count in exact_split_counts]
    remaining_record_count = record_count - sum(split_counts)
    fractional_remainders = tuple(
        exact_count - split_count
        for exact_count, split_count in zip(
            exact_split_counts, split_counts, strict=True
        )
    )
    split_priority = sorted(
        range(len(split_fractions)),
        key=lambda split_index: (-fractional_remainders[split_index], split_index),
    )
    for split_index in split_priority[:remaining_record_count]:
        split_counts[split_index] += 1

    return QetFineTuningSplitCounts(
        train_count=split_counts[0],
        validation_count=split_counts[1],
        test_count=split_counts[2],
    )


class QetFineTuningPyGDataset(
    Dataset[tuple[Data, torch.Tensor, torch.Tensor, _QetFineTuningSampleLabels]]
):
    """Create one CPU PyG training sample lazily for each QET record."""

    def __init__(
        self,
        records: Sequence[QetFineTuningRecord],
        graph_settings: QetFineTuningGraphSettings,
    ) -> None:
        """Snapshot records and configure model-compatible graph conversion."""
        self._records = tuple(records)
        self._graph_converter = Structure2Graph(
            element_types=graph_settings.element_types, cutoff=graph_settings.cutoff
        )

    def __len__(self) -> int:
        """Return the number of available QET fine-tuning records."""
        return len(self._records)

    def __getitem__(
        self, index: int
    ) -> tuple[Data, torch.Tensor, torch.Tensor, _QetFineTuningSampleLabels]:
        """Convert one record into the sample shape required by MatGL."""
        qet_fine_tuning_record = self._records[index]

        # Graph conversion must remain on the CPU even when another part of
        # the process has changed PyTorch's default device.
        with torch.device("cpu"):
            graph, lattice, state_attributes = self._graph_converter.get_graph(
                qet_fine_tuning_record.structure
            )

        cpu_graph = graph.to(device="cpu")

        sample_labels = _create_qet_fine_tuning_sample_labels(
            qet_fine_tuning_record=qet_fine_tuning_record
        )

        return (
            cpu_graph,
            torch.as_tensor(lattice, dtype=matgl.float_th, device="cpu"),
            torch.as_tensor(state_attributes, dtype=matgl.float_th, device="cpu"),
            sample_labels,
        )


def create_qet_fine_tuning_data_loaders(
    record_splits: QetFineTuningRecordSplits,
    graph_settings: QetFineTuningGraphSettings,
    *,
    batch_size: int,
    pin_memory: bool,
    training_shuffle_seed: int,
) -> QetFineTuningDataLoaders:
    """Create deterministic loaders for all fine-tuning record splits."""
    if batch_size < 1:
        message = "QET fine-tuning batch_size must be positive."
        raise ValueError(message)

    training_shuffle_generator = torch.Generator(device="cpu")
    training_shuffle_generator.manual_seed(training_shuffle_seed)

    return QetFineTuningDataLoaders(
        training_data_loader=(
            _create_qet_fine_tuning_data_loader(
                records=record_splits.train_records,
                graph_settings=graph_settings,
                batch_size=batch_size,
                shuffle_records=True,
                pin_memory=pin_memory,
                shuffle_generator=training_shuffle_generator,
            )
        ),
        validation_data_loader=(
            _create_qet_fine_tuning_data_loader(
                records=record_splits.validation_records,
                graph_settings=graph_settings,
                batch_size=batch_size,
                shuffle_records=False,
                pin_memory=pin_memory,
                shuffle_generator=None,
            )
        ),
        test_data_loader=(
            _create_qet_fine_tuning_data_loader(
                records=record_splits.test_records,
                graph_settings=graph_settings,
                batch_size=batch_size,
                shuffle_records=False,
                pin_memory=pin_memory,
                shuffle_generator=None,
            )
            if record_splits.test_records
            else None
        ),
    )


def _create_qet_fine_tuning_data_loader(
    *,
    records: Sequence[QetFineTuningRecord],
    graph_settings: QetFineTuningGraphSettings,
    batch_size: int,
    shuffle_records: bool,
    pin_memory: bool,
    shuffle_generator: torch.Generator | None,
) -> DataLoader[Any]:
    """Create one MatGL-compatible loader from an explicit record split."""
    qet_fine_tuning_dataset = QetFineTuningPyGDataset(
        records=records, graph_settings=graph_settings
    )

    return DataLoader(
        qet_fine_tuning_dataset,
        batch_size=batch_size,
        shuffle=shuffle_records,
        pin_memory=pin_memory,
        generator=shuffle_generator,
        collate_fn=partial(collate_fn_pes, include_stress=True, include_charge=True),
    )


def _create_qet_fine_tuning_sample_labels(
    *, qet_fine_tuning_record: QetFineTuningRecord
) -> _QetFineTuningSampleLabels:
    """Convert one record into MatGL potential labels on the CPU."""
    return {
        "energies": torch.as_tensor(
            qet_fine_tuning_record.energy, dtype=matgl.float_th, device="cpu"
        ),
        "forces": torch.as_tensor(
            qet_fine_tuning_record.forces, dtype=matgl.float_th, device="cpu"
        ),
        "stresses": torch.as_tensor(
            voigt_6_to_full_3x3_stress(qet_fine_tuning_record.stress),  # type: ignore  [no-untyped-call]
            dtype=matgl.float_th,
            device="cpu",
        )
        * EV_PER_ANG3_TO_GPA,
        "charges": torch.as_tensor(
            qet_fine_tuning_record.charges, dtype=matgl.float_th, device="cpu"
        ),
    }
