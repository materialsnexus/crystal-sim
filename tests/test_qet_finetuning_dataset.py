"""Tests for in-memory QET fine-tuning splitting and PyG conversion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matgl
import pytest
import torch
from matgl.apps.pes import EV_PER_ANG3_TO_GPA
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from crystal.calculators.mlip.qet_finetuning_dataset import (
    QetFineTuningGraphSettings,
    QetFineTuningPyGDataset,
    QetFineTuningRecordSplits,
    create_qet_fine_tuning_data_loaders,
    split_qet_fine_tuning_records,
)
from crystal.schemas.qet_finetuning import (
    QetFineTuningRecord,
    QetFineTuningSplitSettings,
)
from tests.helpers.constants.qet_finetuning import TEST_QET_FINE_TUNING_CHARGES
from tests.helpers.qet_finetuning import create_qet_fine_tuning_record

NONZERO_STRESS_EV_PER_ANGSTROM_CUBED = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _create_qet_fine_tuning_split_settings() -> QetFineTuningSplitSettings:
    return QetFineTuningSplitSettings(
        train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2, seed=19
    )


def _create_qet_fine_tuning_graph_settings() -> QetFineTuningGraphSettings:
    return QetFineTuningGraphSettings(element_types=("Si",), cutoff=5.0)


def _record_energies(records: Sequence[QetFineTuningRecord]) -> list[float]:
    return [record.energy for record in records]


def _data_loader_energy_order(data_loader: DataLoader[Any]) -> list[float]:
    energy_order: list[float] = []

    for data_loader_batch in data_loader:
        batch_energies = data_loader_batch[3]
        energy_order.extend(float(batch_energy) for batch_energy in batch_energies)

    return energy_order


def test_record_splitting_is_deterministic_disjoint_and_exhaustive() -> None:
    records = [
        create_qet_fine_tuning_record(float(record_index)) for record_index in range(10)
    ]
    original_record_energy_order = _record_energies(records)

    first_record_splits = split_qet_fine_tuning_records(
        records, _create_qet_fine_tuning_split_settings()
    )
    repeated_record_splits = split_qet_fine_tuning_records(
        records, _create_qet_fine_tuning_split_settings()
    )

    assert _record_energies(records) == original_record_energy_order

    assert _record_energies(first_record_splits.train_records) == _record_energies(
        repeated_record_splits.train_records
    )
    assert _record_energies(first_record_splits.validation_records) == _record_energies(
        repeated_record_splits.validation_records
    )
    assert _record_energies(first_record_splits.test_records) == _record_energies(
        repeated_record_splits.test_records
    )

    assert len(first_record_splits.train_records) == 6
    assert len(first_record_splits.validation_records) == 2
    assert len(first_record_splits.test_records) == 2

    partitioned_record_energies = [
        *_record_energies(first_record_splits.train_records),
        *_record_energies(first_record_splits.validation_records),
        *_record_energies(first_record_splits.test_records),
    ]

    assert sorted(partitioned_record_energies) == (original_record_energy_order)
    assert len(set(partitioned_record_energies)) == len(partitioned_record_energies)


@pytest.mark.parametrize(
    ("record_count", "expected_split_counts"),
    [
        pytest.param(9, (7, 1, 1), id="nine-records"),
        pytest.param(10, (8, 1, 1), id="ten-records"),
        pytest.param(11, (9, 1, 1), id="eleven-records"),
    ],
)
def test_record_splitting_uses_largest_remainder_allocation(
    record_count: int, expected_split_counts: tuple[int, int, int]
) -> None:
    records = [
        create_qet_fine_tuning_record(float(record_index))
        for record_index in range(record_count)
    ]

    record_splits = split_qet_fine_tuning_records(
        records, QetFineTuningSplitSettings(seed=19)
    )

    actual_split_counts = (
        len(record_splits.train_records),
        len(record_splits.validation_records),
        len(record_splits.test_records),
    )
    assert actual_split_counts == expected_split_counts


def test_record_splitting_allows_empty_optional_test_split_after_allocation() -> None:
    records = [create_qet_fine_tuning_record(1.0), create_qet_fine_tuning_record(2.0)]
    split_settings = QetFineTuningSplitSettings(
        train_fraction=0.5, validation_fraction=0.25, test_fraction=0.25, seed=19
    )

    record_splits = split_qet_fine_tuning_records(records, split_settings)

    assert len(record_splits.train_records) == 1
    assert len(record_splits.validation_records) == 1
    assert len(record_splits.test_records) == 0


def test_data_loader_factory_omits_empty_optional_test_split() -> None:
    record_splits = QetFineTuningRecordSplits(
        train_records=(create_qet_fine_tuning_record(1.0),),
        validation_records=(create_qet_fine_tuning_record(2.0),),
        test_records=(),
    )

    qet_fine_tuning_data_loaders = create_qet_fine_tuning_data_loaders(
        record_splits=record_splits,
        graph_settings=_create_qet_fine_tuning_graph_settings(),
        batch_size=2,
        pin_memory=False,
        training_shuffle_seed=31,
    )

    assert qet_fine_tuning_data_loaders.test_data_loader is None


def test_dataset_snapshots_records_and_creates_cpu_matgl_sample() -> None:
    source_records = [
        create_qet_fine_tuning_record(1.5, stress=NONZERO_STRESS_EV_PER_ANGSTROM_CUBED)
    ]
    qet_fine_tuning_dataset = QetFineTuningPyGDataset(
        records=source_records, graph_settings=_create_qet_fine_tuning_graph_settings()
    )

    source_records.clear()

    assert len(qet_fine_tuning_dataset) == 1

    (graph, lattice, state_attributes, sample_labels) = qet_fine_tuning_dataset[0]

    assert isinstance(graph, Data)
    assert graph.node_type.device.type == "cpu"
    assert lattice.device.type == "cpu"
    assert state_attributes.device.type == "cpu"

    assert sample_labels["energies"].device.type == "cpu"
    assert sample_labels["forces"].device.type == "cpu"
    assert sample_labels["stresses"].device.type == "cpu"
    assert sample_labels["charges"].device.type == "cpu"

    expected_charge_tensor = torch.tensor(
        TEST_QET_FINE_TUNING_CHARGES, dtype=matgl.float_th
    )
    assert torch.equal(sample_labels["charges"], expected_charge_tensor)

    assert sample_labels["energies"].item() == 1.5
    assert sample_labels["forces"].shape == (2, 3)
    assert sample_labels["charges"].shape == (2,)
    assert sample_labels["stresses"].shape == (3, 3)

    expected_stress_tensor = (
        torch.tensor(
            [[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]], dtype=matgl.float_th
        )
        * EV_PER_ANG3_TO_GPA
    )

    assert torch.allclose(sample_labels["stresses"], expected_stress_tensor)


def test_data_loader_factory_preserves_evaluation_order_and_seeds_training_shuffle() -> (
    None
):
    record_splits = QetFineTuningRecordSplits(
        train_records=tuple(
            create_qet_fine_tuning_record(float(energy)) for energy in range(4)
        ),
        validation_records=tuple(
            create_qet_fine_tuning_record(float(energy)) for energy in range(4, 6)
        ),
        test_records=tuple(
            create_qet_fine_tuning_record(float(energy)) for energy in range(6, 8)
        ),
    )

    first_data_loaders = create_qet_fine_tuning_data_loaders(
        record_splits=record_splits,
        graph_settings=_create_qet_fine_tuning_graph_settings(),
        batch_size=2,
        pin_memory=False,
        training_shuffle_seed=31,
    )
    repeated_data_loaders = create_qet_fine_tuning_data_loaders(
        record_splits=record_splits,
        graph_settings=_create_qet_fine_tuning_graph_settings(),
        batch_size=2,
        pin_memory=False,
        training_shuffle_seed=31,
    )

    first_training_energy_order = _data_loader_energy_order(
        first_data_loaders.training_data_loader
    )
    repeated_training_energy_order = _data_loader_energy_order(
        repeated_data_loaders.training_data_loader
    )

    assert first_training_energy_order == repeated_training_energy_order
    assert sorted(first_training_energy_order) == [0.0, 1.0, 2.0, 3.0]

    assert _data_loader_energy_order(first_data_loaders.validation_data_loader) == [
        4.0,
        5.0,
    ]
    assert first_data_loaders.test_data_loader is not None
    assert _data_loader_energy_order(first_data_loaders.test_data_loader) == [6.0, 7.0]


def test_data_loader_factory_collates_matgl_charge_training_batch() -> None:
    record_splits = QetFineTuningRecordSplits(
        train_records=(
            create_qet_fine_tuning_record(1.5),
            create_qet_fine_tuning_record(2.5),
        ),
        validation_records=(create_qet_fine_tuning_record(3.5),),
        test_records=(create_qet_fine_tuning_record(4.5),),
    )

    qet_fine_tuning_data_loaders = create_qet_fine_tuning_data_loaders(
        record_splits=record_splits,
        graph_settings=(_create_qet_fine_tuning_graph_settings()),
        batch_size=2,
        pin_memory=False,
        training_shuffle_seed=31,
    )

    training_batch = next(iter(qet_fine_tuning_data_loaders.training_data_loader))

    assert len(training_batch) == 7

    (
        batched_graph,
        batched_lattice,
        batched_state_attributes,
        batched_energies,
        batched_forces,
        batched_stresses,
        batched_charges,
    ) = training_batch

    assert batched_graph.num_nodes == 4
    assert batched_lattice.shape == (2, 3, 3)
    assert batched_state_attributes.shape == (2, 2)
    assert sorted(float(energy) for energy in batched_energies) == [1.5, 2.5]
    assert batched_forces.shape == (4, 3)
    assert batched_stresses.shape == (6, 3)

    expected_batched_charges = torch.tensor(
        TEST_QET_FINE_TUNING_CHARGES * 2, dtype=matgl.float_th
    )
    assert torch.equal(batched_charges, expected_batched_charges)
