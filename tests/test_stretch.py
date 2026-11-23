import numpy as np
import pytest
from numpy.typing import NDArray

from crystal.structure.generators.perturbation.stretch import stretch_cell
from tests.helpers.constants.structure import TEST_BCC_FE_STRUCTURE


def test_stretch_cell_returns_new_structure_without_mutating_input() -> None:
    structure = TEST_BCC_FE_STRUCTURE.copy()
    original_lattice = structure.lattice.matrix.copy()

    stretched = stretch_cell(
        structure=structure,
        random_seed=42,
        max_normal_strain_fraction=0.2,
        max_shear_strain_fraction=0.1,
    )
    assert stretched is not None
    assert stretched is not structure
    assert np.allclose(structure.lattice.matrix, original_lattice)
    assert not np.allclose(stretched.lattice.matrix, original_lattice)


def test_stretch_cell_applies_strain_within_requested_bounds() -> None:
    max_normal_strain_fraction = 0.2
    max_shear_strain_fraction = 0.1
    minimum_strain = 1e-3

    stretched = stretch_cell(
        structure=TEST_BCC_FE_STRUCTURE,
        random_seed=42,
        max_normal_strain_fraction=max_normal_strain_fraction,
        max_shear_strain_fraction=max_shear_strain_fraction,
    )
    assert stretched is not None

    original_lattice = TEST_BCC_FE_STRUCTURE.lattice.matrix
    stretched_lattice = stretched.lattice.matrix
    deformation = np.linalg.inv(original_lattice) @ stretched_lattice
    strain_applied = deformation - np.eye(3)

    assert np.allclose(strain_applied, strain_applied.T)
    diag_strains = np.diag(strain_applied)
    off_diag_indices = np.triu_indices(3, k=1)
    off_diag_strains = strain_applied[off_diag_indices]

    assert np.all(np.abs(diag_strains) >= minimum_strain)
    assert np.all(np.abs(diag_strains) <= max_normal_strain_fraction)
    assert np.all(np.abs(off_diag_strains) >= minimum_strain)
    assert np.all(np.abs(off_diag_strains) <= max_shear_strain_fraction)


def test_stretch_cell_is_reproducible_for_same_seed() -> None:
    structure = TEST_BCC_FE_STRUCTURE.copy()

    stretched_1 = stretch_cell(
        structure=structure,
        random_seed=42,
        max_normal_strain_fraction=0.2,
        max_shear_strain_fraction=0.1,
    )
    stretched_2 = stretch_cell(
        structure=structure,
        random_seed=42,
        max_normal_strain_fraction=0.2,
        max_shear_strain_fraction=0.1,
    )
    assert stretched_1 is not None
    assert stretched_2 is not None
    assert np.allclose(stretched_1.lattice.matrix, stretched_2.lattice.matrix)
    assert np.allclose(stretched_1.frac_coords, stretched_2.frac_coords)


def test_stretch_cell_logs_warning_and_returns_none_for_invalid_deformation_determinant(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sampled_strains = iter(
        [
            np.array([-1.5, 0.1, 0.1], dtype=np.float64),
            np.array([1e-3, 1e-3, 1e-3], dtype=np.float64),
        ]
    )

    def _mock_sample_signed_strains(
        *, max_strain_fraction: float, size: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        _ = (max_strain_fraction, size, rng)
        return next(sampled_strains)

    monkeypatch.setattr(
        "crystal.structure.generators.perturbation.stretch.sample_signed_strains",
        _mock_sample_signed_strains,
    )

    caplog.clear()
    result = stretch_cell(
        structure=TEST_BCC_FE_STRUCTURE,
        random_seed=42,
        max_normal_strain_fraction=0.2,
        max_shear_strain_fraction=0.1,
    )
    assert result is None
    assert "Sampled deformation matrix is invalid" in caplog.text
