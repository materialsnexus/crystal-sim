from unittest.mock import patch

import numpy as np
import pytest
from ase.data import covalent_radii
from numpy.typing import NDArray
from pydantic import ValidationError
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.structure.generators.perturbation.rattle import (
    RattleAtomDisplacementMode,
    rattle_atom_positions,
)
from tests.helpers.constants.structure import TEST_BCC_FE_STRUCTURE, TEST_LINBO3_SLAB


@pytest.mark.parametrize(
    ("structure", "displacement_mode"),
    [
        (TEST_LINBO3_SLAB, RattleAtomDisplacementMode.normal),
        (TEST_LINBO3_SLAB, RattleAtomDisplacementMode.covalent_radius_scaled),
    ],
)
def test_rattle_atom_positions(
    structure: PMG_Structure, displacement_mode: RattleAtomDisplacementMode
) -> None:
    new_structure = rattle_atom_positions(
        structure=structure,
        noise_scaling_factor=50,
        random_seed=42,
        displacement_mode=displacement_mode,
    )
    assert not np.any(np.isclose(new_structure.frac_coords, structure.frac_coords))
    assert np.all((new_structure.frac_coords >= 0) & (new_structure.frac_coords < 1.0))


def test_displacement_mode_with_mocked_rng() -> None:
    raw_noise_vector = np.array([1.0, -2.0, 0.5])

    class _MockRNG:
        def normal(self, *, scale: float, size: int) -> NDArray[np.float64]:
            assert size == 3
            return raw_noise_vector * scale

    with patch(
        "crystal.structure.generators.perturbation.rattle.np.random.default_rng",
        return_value=_MockRNG(),
    ):
        structure = PMG_Structure(
            lattice=Lattice.cubic(10.0), species=["Fe"], coords=[[0.5, 0.5, 0.5]]
        )
        noise_scaling_factor = 0.1
        expected_normal_displacement = raw_noise_vector * noise_scaling_factor

        rattled_normal = rattle_atom_positions(
            structure=structure,
            random_seed=42,
            noise_scaling_factor=noise_scaling_factor,
            displacement_mode=RattleAtomDisplacementMode.normal,
        )
        displacement_normal = rattled_normal.cart_coords[0] - structure.cart_coords[0]
        assert np.allclose(displacement_normal, expected_normal_displacement)

        rattled_covalent_scaled = rattle_atom_positions(
            structure=structure,
            random_seed=42,
            noise_scaling_factor=noise_scaling_factor,
            displacement_mode=RattleAtomDisplacementMode.covalent_radius_scaled,
        )
        displacement_covalent_scaled = (
            rattled_covalent_scaled.cart_coords[0] - structure.cart_coords[0]
        )
        expected_covalent_scaled_displacement = (
            expected_normal_displacement * covalent_radii[26]
        )
        assert np.allclose(
            displacement_covalent_scaled, expected_covalent_scaled_displacement
        )


def test_rattle_atom_positions_raises_for_negative_noise_scale() -> None:
    with pytest.raises(ValidationError):
        _ = rattle_atom_positions(
            structure=TEST_BCC_FE_STRUCTURE, random_seed=42, noise_scaling_factor=-0.1
        )
