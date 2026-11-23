"""Dataset-backed evaluation of MLIP models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.io import iread

from crystal.helpers.cohesive_energy import CohesiveEnergiesCalculator
from crystal.helpers.mlip_helpers import prepare_atoms_for_matgl
from crystal.helpers.properties import get_force_norms, get_stress_norms
from crystal.helpers.structure_summary import StructureSummary
from crystal.schemas.arrays import FloatArray
from crystal.schemas.isolated_atom_energies import IsolatedAtomEnergies
from crystal.schemas.mlip_model_ref import MlipModelRef

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator


@dataclass(frozen=True, slots=True)
class ReferencePredictionPair:
    """Reference and predicted property values."""

    reference: FloatArray
    predicted: FloatArray

    def __post_init__(self) -> None:
        """Validate that reference and predicted values are aligned."""
        if self.reference.shape != self.predicted.shape:
            error_msg = (
                "Reference and predicted value shapes must match: "
                f"{self.reference.shape} != {self.predicted.shape}."
            )
            raise ValueError(error_msg)

    @property
    def difference(self) -> FloatArray:
        """Return signed differences as ``predicted - reference``."""
        return np.asarray(self.predicted - self.reference, dtype=np.float64)

    @property
    def rmse(self) -> float:
        """Return the root mean square prediction error."""
        error = self.difference
        if error.size == 0:
            error_msg = "RMSE requires at least one paired value."
            raise ValueError(error_msg)
        return float(np.sqrt(np.mean(np.square(error))))


@dataclass(frozen=True, slots=True)
class ModelEvaluationResult:
    """Paired reference and MLIP-predicted structure results."""

    model: MlipModelRef
    reference_structure_summaries: tuple[StructureSummary, ...]
    predicted_structure_summaries: tuple[StructureSummary, ...]
    isolated_atom_energies: IsolatedAtomEnergies

    def __post_init__(self) -> None:
        """Validate that reference and predicted structures are aligned."""
        reference_count = len(self.reference_structure_summaries)
        predicted_count = len(self.predicted_structure_summaries)
        if reference_count != predicted_count:
            error_msg = (
                "Reference and predicted structure counts must match: "
                f"{reference_count} != {predicted_count}."
            )
            raise ValueError(error_msg)
        for structure_index, (reference, predicted) in enumerate(
            zip(
                self.reference_structure_summaries,
                self.predicted_structure_summaries,
                strict=True,
            )
        ):
            if not np.array_equal(reference.atomic_numbers, predicted.atomic_numbers):
                error_msg = (
                    "Reference and predicted structures must contain the same atoms "
                    f"in the same order; mismatch at structure {structure_index}."
                )
                raise ValueError(error_msg)

    @property
    def cohesive_energies_per_atom(self) -> ReferencePredictionPair:
        """Return paired cohesive energies per atom in eV/atom."""
        calculator = CohesiveEnergiesCalculator(
            isolated_atom_energies=self.isolated_atom_energies
        )
        return ReferencePredictionPair(
            reference=np.asarray(
                calculator.from_structure_summaries(
                    structure_summaries=self.reference_structure_summaries
                ),
                dtype=np.float64,
            ),
            predicted=np.asarray(
                calculator.from_structure_summaries(
                    structure_summaries=self.predicted_structure_summaries
                ),
                dtype=np.float64,
            ),
        )

    @property
    def forces(self) -> ReferencePredictionPair:
        """Return paired force components for all atoms."""
        return ReferencePredictionPair(
            reference=np.vstack(
                [
                    structure_summary.forces
                    for structure_summary in self.reference_structure_summaries
                ]
            ),
            predicted=np.vstack(
                [
                    structure_summary.forces
                    for structure_summary in self.predicted_structure_summaries
                ]
            ),
        )

    @property
    def force_norms(self) -> ReferencePredictionPair:
        """Return paired per-atom force norms."""
        forces = self.forces
        return ReferencePredictionPair(
            reference=get_force_norms(forces.reference),
            predicted=get_force_norms(forces.predicted),
        )

    @property
    def stress(self) -> ReferencePredictionPair:
        """Return paired stress components for all structures."""
        return ReferencePredictionPair(
            reference=np.vstack(
                [
                    structure_summary.stress
                    for structure_summary in self.reference_structure_summaries
                ]
            ),
            predicted=np.vstack(
                [
                    structure_summary.stress
                    for structure_summary in self.predicted_structure_summaries
                ]
            ),
        )

    @property
    def stress_norms(self) -> ReferencePredictionPair:
        """Return paired per-structure Voigt-stress norms."""
        stress = self.stress
        return ReferencePredictionPair(
            reference=get_stress_norms(stress.reference),
            predicted=get_stress_norms(stress.predicted),
        )

    @property
    def charges(self) -> ReferencePredictionPair:
        """Return paired charges flattened across atoms."""
        return ReferencePredictionPair(
            reference=np.concatenate(
                [
                    structure_summary.charges
                    for structure_summary in self.reference_structure_summaries
                ]
            ),
            predicted=np.concatenate(
                [
                    structure_summary.charges
                    for structure_summary in self.predicted_structure_summaries
                ]
            ),
        )


@dataclass(slots=True)
class ModelEvaluator:
    """Reusable evaluator owning an extxyz dataset.

    `reference_structure_summaries` contains the labelled property data used as
    the reference. `_dataset_atoms` contains calculator-free atoms used as model
    inputs to generate predictions.
    """

    dataset_path: Path
    isolated_atom_energies: IsolatedAtomEnergies
    reference_structure_summaries: tuple[StructureSummary, ...] = field(
        init=False, repr=False
    )
    _dataset_atoms: tuple[Atoms, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load and validate the reference dataset."""
        reference_atoms = _read_dataset(self.dataset_path)
        self.reference_structure_summaries = tuple(
            StructureSummary.from_atoms(atoms) for atoms in reference_atoms
        )
        self._dataset_atoms = tuple(
            atoms.copy()  # type: ignore[no-untyped-call]
            for atoms in reference_atoms
        )

    def evaluate(self, *, model: MlipModelRef) -> ModelEvaluationResult:
        """Evaluate one model against the cached reference dataset."""
        prediction_summaries: list[StructureSummary] = []
        calculator = self._build_calculator(model=model)

        for dataset_atoms in self._dataset_atoms:
            predicted_atoms = prepare_atoms_for_matgl(dataset_atoms)
            predicted_atoms.calc = calculator
            prediction_summaries.append(StructureSummary.from_atoms(predicted_atoms))

        return ModelEvaluationResult(
            model=model,
            reference_structure_summaries=self.reference_structure_summaries,
            predicted_structure_summaries=tuple(prediction_summaries),
            isolated_atom_energies=self.isolated_atom_energies,
        )

    def _build_calculator(self, *, model: MlipModelRef) -> PESCalculator:
        import torch  # noqa: PLC0415
        from matgl.ext.ase import PESCalculator  # noqa: PLC0415
        from matgl.utils.io import load_model  # noqa: PLC0415

        potential = load_model(model.model_path)
        potential.to(device=torch.get_default_device())
        return PESCalculator(potential=potential, stress_unit="eV/A3")


def _read_dataset(dataset_path: Path) -> tuple[Atoms, ...]:
    """Read and validate a non-empty extxyz dataset."""
    if not dataset_path.is_file():
        error_msg = f"Dataset path is not a file: {dataset_path}."
        raise ValueError(error_msg)

    dataset_atoms = tuple(iread(dataset_path, index=":", format="extxyz"))
    if not dataset_atoms:
        error_msg = f"Dataset is empty: {dataset_path}."
        raise ValueError(error_msg)

    return dataset_atoms
