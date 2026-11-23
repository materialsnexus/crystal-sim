"""Tests for crystal.helpers.ModelEvaluator"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, write
from ase.io.extxyz import XYZError
from matgl.ext.ase import Atoms2Graph

from crystal.helpers.model_evaluator import ModelEvaluator
from crystal.helpers.structure_summary import StructureSummary
from crystal.schemas.mlip_model_ref import MlipModelRef

ISOLATED_ATOM_ENERGIES = {"H": -1.0, "He": -2.0}

ASE_ATOMS_0 = Atoms(
    numbers=[1, 1], positions=np.zeros((2, 3)), cell=[10.0, 10.0, 10.0], pbc=True
)
ASE_ATOMS_0.calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
    ASE_ATOMS_0,
    energy=-5.0,
    forces=np.asarray([(3.0, 4.0, 0.0), (0.0, 0.0, 2.0)], dtype=np.float64),
    stress=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
    charges=np.asarray([0.1, -0.1], dtype=np.float64),
)

ASE_ATOMS_1 = Atoms(
    numbers=[1, 2], positions=np.zeros((2, 3)), cell=[10.0, 10.0, 10.0], pbc=True
)
ASE_ATOMS_1.calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
    ASE_ATOMS_1,
    energy=-8.0,
    forces=np.asarray([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float64),
    stress=np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=np.float64),
    charges=np.asarray([0.2, 0.3], dtype=np.float64),
)


class _MockCalculator(Calculator):
    """Return correctly shaped results populated only with `prediction_value`."""

    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "forces",
        "stress",
        "charges",
    ]

    def __init__(self, *, prediction_value: float) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.prediction_value = prediction_value

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(  # type: ignore[no-untyped-call]
            atoms, properties, system_changes
        )
        assert atoms is not None
        atom_count = len(atoms)
        self.results = {
            "energy": self.prediction_value,
            "forces": np.full((atom_count, 3), self.prediction_value, dtype=np.float64),
            "stress": np.full(6, self.prediction_value, dtype=np.float64),
            "charges": np.full(atom_count, self.prediction_value, dtype=np.float64),
        }


def test_model_evaluator_rejects_non_file_dataset_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.extxyz"

    with pytest.raises(
        ValueError, match=re.escape(f"Dataset path is not a file: {missing_path}.")
    ):
        _ = ModelEvaluator(
            dataset_path=missing_path, isolated_atom_energies=ISOLATED_ATOM_ENERGIES
        )


def test_model_evaluator_rejects_empty_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty.extxyz"
    dataset_path.touch()

    with pytest.raises(
        ValueError, match=re.escape(f"Dataset is empty: {dataset_path}.")
    ):
        _ = ModelEvaluator(
            dataset_path=dataset_path, isolated_atom_energies=ISOLATED_ATOM_ENERGIES
        )


def test_model_evaluator_rejects_corrupted_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "corrupted.extxyz"
    dataset_path.write_text("not an extxyz dataset", encoding="utf-8")

    with pytest.raises(XYZError, match="Expected xyz header"):
        _ = ModelEvaluator(
            dataset_path=dataset_path, isolated_atom_energies=ISOLATED_ATOM_ENERGIES
        )


def test_model_evaluator_builds_mlip_calculator(tmp_path: Path) -> None:
    model = MlipModelRef(path=tmp_path)
    dataset_path = tmp_path / "labelled.extxyz"
    write(dataset_path, [ASE_ATOMS_0], format="extxyz")
    evaluator = ModelEvaluator(
        dataset_path=dataset_path, isolated_atom_energies=ISOLATED_ATOM_ENERGIES
    )
    potential = Mock()
    calculator = Mock()

    with (
        patch("matgl.utils.io.load_model", return_value=potential) as load_model,
        patch(
            "matgl.ext.ase.PESCalculator", return_value=calculator
        ) as calculator_type,
        patch("torch.get_default_device", return_value="cpu"),
    ):
        built_calculator = evaluator._build_calculator(model=model)

    load_model.assert_called_once_with(model.model_path)
    potential.to.assert_called_once_with(device="cpu")
    calculator_type.assert_called_once_with(potential=potential, stress_unit="eV/A3")
    assert built_calculator is calculator


def test_model_evaluator_loads_reference_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "labelled.extxyz"
    write(dataset_path, [ASE_ATOMS_0, ASE_ATOMS_1], format="extxyz")
    evaluator = ModelEvaluator(
        dataset_path=dataset_path, isolated_atom_energies=ISOLATED_ATOM_ENERGIES
    )

    for input_atoms, evaluator_stored_atoms, evaluator_reference_summary in zip(
        [ASE_ATOMS_0, ASE_ATOMS_1],
        evaluator._dataset_atoms,
        evaluator.reference_structure_summaries,
        strict=True,
    ):
        assert evaluator_stored_atoms == input_atoms
        assert evaluator_stored_atoms.calc is None

        expected_summary = StructureSummary.from_atoms(input_atoms)
        for summary_field in fields(StructureSummary):
            np.testing.assert_equal(
                getattr(evaluator_reference_summary, summary_field.name),
                getattr(expected_summary, summary_field.name),
            )


def test_model_evaluator_reads_dataset_once_for_multiple_evaluations(
    tmp_path: Path,
) -> None:
    first_model = MlipModelRef(path=tmp_path / "first-model")
    second_model = MlipModelRef(path=tmp_path / "second-model")
    dataset_path = tmp_path / "labelled.extxyz"
    write(dataset_path, [ASE_ATOMS_0, ASE_ATOMS_1], format="extxyz")

    with (
        patch("crystal.helpers.model_evaluator.iread", wraps=iread) as read_dataset,
        patch.object(
            ModelEvaluator,
            "_build_calculator",
            side_effect=[
                _MockCalculator(prediction_value=1.0),
                _MockCalculator(prediction_value=2.0),
            ],
        ),
    ):
        evaluator = ModelEvaluator(
            dataset_path=dataset_path, isolated_atom_energies=ISOLATED_ATOM_ENERGIES
        )
        _ = evaluator.evaluate(model=first_model)
        _ = evaluator.evaluate(model=second_model)

    read_dataset.assert_called_once_with(dataset_path, index=":", format="extxyz")


def test_model_evaluator_normalizes_partial_pbc_before_prediction(
    tmp_path: Path,
) -> None:
    model = MlipModelRef(path=tmp_path / "model")
    partial_pbc_atoms = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 5.0], [3.5, 0.0, 5.0]],
        cell=[4.0, 10.0, 12.0],
        pbc=(True, True, False),
    )
    partial_pbc_atoms.calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
        partial_pbc_atoms,
        energy=-1.0,
        forces=np.zeros((2, 3), dtype=np.float64),
        stress=np.zeros(6, dtype=np.float64),
        charges=np.zeros(2, dtype=np.float64),
    )
    dataset_path = tmp_path / "partial-pbc.extxyz"
    write(dataset_path, partial_pbc_atoms, format="extxyz")
    calculator = _MockCalculator(prediction_value=1.0)
    evaluator = ModelEvaluator(
        dataset_path=dataset_path, isolated_atom_energies=ISOLATED_ATOM_ENERGIES
    )

    with patch.object(ModelEvaluator, "_build_calculator", return_value=calculator):
        evaluator.evaluate(model=model)

    assert calculator.atoms is not None
    matgl_graph_converter = Atoms2Graph(element_types=("H",), cutoff=1.0)
    graph_from_matgl_ready_atoms, _, _ = matgl_graph_converter.get_graph(
        calculator.atoms
    )
    partial_pbc_atoms = calculator.atoms.copy()
    partial_pbc_atoms.pbc = (True, True, False)
    graph_from_partial_pbc_atoms, _, _ = matgl_graph_converter.get_graph(
        partial_pbc_atoms
    )

    assert graph_from_partial_pbc_atoms.edge_index.shape[1] == 0
    assert graph_from_matgl_ready_atoms.edge_index.shape[1] == 2


def test_model_evaluator_evaluates_models_independently(tmp_path: Path) -> None:
    first_model = MlipModelRef(path=tmp_path / "first-model")
    second_model = MlipModelRef(path=tmp_path / "second-model")
    dataset_path = tmp_path / "labelled.extxyz"
    write(dataset_path, [ASE_ATOMS_0, ASE_ATOMS_1], format="extxyz")
    evaluator = ModelEvaluator(
        dataset_path=dataset_path, isolated_atom_energies=ISOLATED_ATOM_ENERGIES
    )

    with patch.object(
        ModelEvaluator,
        "_build_calculator",
        side_effect=[
            _MockCalculator(prediction_value=1.0),
            _MockCalculator(prediction_value=2.0),
        ],
    ) as build_calculator:
        first_evaluation_results = evaluator.evaluate(model=first_model)
        second_evaluation_results = evaluator.evaluate(model=second_model)

    assert build_calculator.call_args_list == [
        call(model=first_model),
        call(model=second_model),
    ]
    for evaluation_results, model, mock_prediction_value in [
        (first_evaluation_results, first_model, 1.0),
        (second_evaluation_results, second_model, 2.0),
    ]:
        assert evaluation_results.model is model
        assert (
            evaluation_results.reference_structure_summaries
            is evaluator.reference_structure_summaries
        )
        for prediction_summary, input_atoms in zip(
            evaluation_results.predicted_structure_summaries,
            [ASE_ATOMS_0, ASE_ATOMS_1],
            strict=True,
        ):
            atom_count = len(input_atoms)
            np.testing.assert_array_equal(
                prediction_summary.atomic_numbers, input_atoms.numbers
            )
            np.testing.assert_allclose(
                prediction_summary.forces,
                np.full((atom_count, 3), mock_prediction_value),
            )
            assert prediction_summary.charges is not None
            np.testing.assert_allclose(
                prediction_summary.charges, np.full(atom_count, mock_prediction_value)
            )
            np.testing.assert_allclose(
                prediction_summary.stress, np.full(6, mock_prediction_value)
            )
            assert prediction_summary.total_energy == pytest.approx(
                mock_prediction_value
            )
