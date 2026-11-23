"""Tests for MLIP model evaluation analysis helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from crystal.helpers.cohesive_energy import CohesiveEnergiesCalculator
from crystal.helpers.model_evaluator import (
    ModelEvaluationResult,
    ReferencePredictionPair,
)
from crystal.helpers.structure_summary import StructureSummary
from crystal.schemas.mlip_model_ref import MlipModelRef

ISOLATED_ATOM_ENERGIES = {"H": -1.0, "He": -2.0}

REFERENCE_STRUCTURE_0 = StructureSummary(
    atomic_numbers=np.asarray([1, 1], dtype=np.int_),
    forces=np.asarray([(3.0, 4.0, 0.0), (0.0, 0.0, 2.0)], dtype=np.float64),
    charges=np.asarray([0.1, -0.1], dtype=np.float64),
    stress=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
    total_energy=-5.0,
)
REFERENCE_STRUCTURE_1 = StructureSummary(
    atomic_numbers=np.asarray([1, 2], dtype=np.int_),
    forces=np.asarray([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float64),
    charges=np.asarray([0.2, 0.3], dtype=np.float64),
    stress=np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=np.float64),
    total_energy=-8.0,
)
PREDICTED_STRUCTURE_0 = StructureSummary(
    atomic_numbers=np.asarray([1, 1], dtype=np.int_),
    forces=np.asarray([(4.0, 5.0, 0.0), (0.0, 0.0, 3.0)], dtype=np.float64),
    charges=np.asarray([0.2, 0.0], dtype=np.float64),
    stress=np.asarray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64),
    total_energy=-4.0,
)
PREDICTED_STRUCTURE_1 = StructureSummary(
    atomic_numbers=np.asarray([1, 2], dtype=np.int_),
    forces=np.asarray([(2.0, 0.0, 0.0), (0.0, 3.0, 0.0)], dtype=np.float64),
    charges=np.asarray([0.3, 0.4], dtype=np.float64),
    stress=np.asarray([0.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64),
    total_energy=-7.0,
)


@pytest.fixture
def evaluation(tmp_path: Path) -> ModelEvaluationResult:
    return ModelEvaluationResult(
        model=MlipModelRef(path=tmp_path),
        reference_structure_summaries=(REFERENCE_STRUCTURE_0, REFERENCE_STRUCTURE_1),
        predicted_structure_summaries=(PREDICTED_STRUCTURE_0, PREDICTED_STRUCTURE_1),
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES,
    )


def test_model_evaluation_rejects_different_structure_counts(tmp_path: Path) -> None:
    model = MlipModelRef(path=tmp_path)
    with pytest.raises(
        ValueError, match="Reference and predicted structure counts must match: 2 != 1"
    ):
        _ = ModelEvaluationResult(
            model=model,
            reference_structure_summaries=(
                REFERENCE_STRUCTURE_0,
                REFERENCE_STRUCTURE_1,
            ),
            predicted_structure_summaries=(PREDICTED_STRUCTURE_0,),
            isolated_atom_energies=ISOLATED_ATOM_ENERGIES,
        )


def test_model_evaluation_rejects_different_atoms(tmp_path: Path) -> None:
    model = MlipModelRef(path=tmp_path)
    with pytest.raises(ValueError, match="mismatch at structure 0"):
        _ = ModelEvaluationResult(
            model=model,
            reference_structure_summaries=(REFERENCE_STRUCTURE_1,),
            predicted_structure_summaries=(PREDICTED_STRUCTURE_0,),
            isolated_atom_energies=ISOLATED_ATOM_ENERGIES,
        )


def test_model_evaluation_returns_cohesive_energies_per_atom(
    evaluation: ModelEvaluationResult,
) -> None:
    values = evaluation.cohesive_energies_per_atom
    calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES
    )

    assert isinstance(values, ReferencePredictionPair)
    np.testing.assert_allclose(
        values.reference,
        calculator.from_structure_summaries(
            structure_summaries=[REFERENCE_STRUCTURE_0, REFERENCE_STRUCTURE_1]
        ),
    )
    np.testing.assert_allclose(
        values.predicted,
        calculator.from_structure_summaries(
            structure_summaries=[PREDICTED_STRUCTURE_0, PREDICTED_STRUCTURE_1]
        ),
    )


def test_reference_prediction_pair_rejects_different_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Reference and predicted value shapes must match: (2,) != (1,)."
        ),
    ):
        _ = ReferencePredictionPair(reference=np.zeros(2), predicted=np.zeros(1))


def test_model_evaluation_returns_forces(evaluation: ModelEvaluationResult) -> None:
    values = evaluation.forces

    np.testing.assert_allclose(
        values.reference,
        np.vstack([REFERENCE_STRUCTURE_0.forces, REFERENCE_STRUCTURE_1.forces]),
    )
    np.testing.assert_allclose(
        values.predicted,
        np.vstack([PREDICTED_STRUCTURE_0.forces, PREDICTED_STRUCTURE_1.forces]),
    )


def test_model_evaluation_returns_force_norms(
    evaluation: ModelEvaluationResult,
) -> None:
    values = evaluation.force_norms

    np.testing.assert_allclose(
        values.reference,
        np.linalg.norm(
            np.vstack([REFERENCE_STRUCTURE_0.forces, REFERENCE_STRUCTURE_1.forces]),
            axis=1,
        ),
    )
    np.testing.assert_allclose(
        values.predicted,
        np.linalg.norm(
            np.vstack([PREDICTED_STRUCTURE_0.forces, PREDICTED_STRUCTURE_1.forces]),
            axis=1,
        ),
    )


def test_model_evaluation_returns_stress(evaluation: ModelEvaluationResult) -> None:
    values = evaluation.stress

    np.testing.assert_allclose(
        values.reference, [REFERENCE_STRUCTURE_0.stress, REFERENCE_STRUCTURE_1.stress]
    )
    np.testing.assert_allclose(
        values.predicted, [PREDICTED_STRUCTURE_0.stress, PREDICTED_STRUCTURE_1.stress]
    )


def test_model_evaluation_returns_stress_norms(
    evaluation: ModelEvaluationResult,
) -> None:
    values = evaluation.stress_norms

    np.testing.assert_allclose(
        values.reference,
        [
            np.linalg.norm(REFERENCE_STRUCTURE_0.stress),
            np.linalg.norm(REFERENCE_STRUCTURE_1.stress),
        ],
    )
    np.testing.assert_allclose(
        values.predicted,
        [
            np.linalg.norm(PREDICTED_STRUCTURE_0.stress),
            np.linalg.norm(PREDICTED_STRUCTURE_1.stress),
        ],
    )


def test_model_evaluation_returns_charges(evaluation: ModelEvaluationResult) -> None:
    values = evaluation.charges

    assert REFERENCE_STRUCTURE_0.charges is not None
    assert REFERENCE_STRUCTURE_1.charges is not None
    assert PREDICTED_STRUCTURE_0.charges is not None
    assert PREDICTED_STRUCTURE_1.charges is not None
    np.testing.assert_allclose(
        values.reference,
        np.concatenate([REFERENCE_STRUCTURE_0.charges, REFERENCE_STRUCTURE_1.charges]),
    )
    np.testing.assert_allclose(
        values.predicted,
        np.concatenate([PREDICTED_STRUCTURE_0.charges, PREDICTED_STRUCTURE_1.charges]),
    )


def test_reference_prediction_pair_calculates_difference_and_rmse() -> None:
    values = ReferencePredictionPair(
        reference=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        predicted=np.asarray([2.0, 0.0, 5.0], dtype=np.float64),
    )

    np.testing.assert_allclose(values.difference, [1.0, -2.0, 2.0])
    assert values.rmse == pytest.approx(np.sqrt(3.0))


def test_reference_prediction_pair_rmse_rejects_empty_values() -> None:
    values = ReferencePredictionPair(
        reference=np.asarray([], dtype=np.float64),
        predicted=np.asarray([], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="RMSE requires at least one"):
        _ = values.rmse
