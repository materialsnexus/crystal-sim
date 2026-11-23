"""Functions for rattling the atom positions of structures."""

from copy import deepcopy
from enum import StrEnum, auto

import numpy as np
from ase.data import covalent_radii
from numpy.typing import NDArray
from pydantic import PositiveFloat, validate_call
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure as PMG_Structure

DEFAULT_ATOM_POSITION_NOISE_SCALING_FACTOR: PositiveFloat = 0.5


class RattleAtomDisplacementMode(StrEnum):  # noqa: D101
    normal = auto()
    covalent_radius_scaled = auto()


def _normal_displacement(
    rng: np.random.Generator, standard_deviation: float
) -> NDArray[np.float64]:
    """Generate isotropic random displacement from a normal distribution."""
    displacement = rng.normal(scale=standard_deviation, size=3)
    assert isinstance(displacement, np.ndarray)
    return displacement


def _covalent_radius_scaled_normal_displacement(
    site: PeriodicSite, rng: np.random.Generator, noise_scaling_factor: float
) -> NDArray[np.float64]:
    """Generate random displacements scaled by covalent radius.
    Inspired by ASSYST's `element_scaled_rattle`.
    """
    covalent_radius = covalent_radii[site.specie.Z]
    displacement = rng.normal(scale=noise_scaling_factor, size=3) * covalent_radius
    assert isinstance(displacement, np.ndarray)
    return displacement


@validate_call
def rattle_atom_positions(
    structure: PMG_Structure,
    random_seed: int,
    noise_scaling_factor: PositiveFloat = DEFAULT_ATOM_POSITION_NOISE_SCALING_FACTOR,
    displacement_mode: RattleAtomDisplacementMode = RattleAtomDisplacementMode.normal,
) -> PMG_Structure:
    """Returns a new structure by adding noise to atom positions of the current structure."""
    rng = np.random.default_rng(seed=random_seed)
    rattled_structure = deepcopy(structure)
    for atom_idx, site in enumerate(rattled_structure):
        if displacement_mode is RattleAtomDisplacementMode.normal:
            displacement = _normal_displacement(
                rng=rng, standard_deviation=noise_scaling_factor
            )
        elif displacement_mode is RattleAtomDisplacementMode.covalent_radius_scaled:
            displacement = _covalent_radius_scaled_normal_displacement(
                site=site, rng=rng, noise_scaling_factor=noise_scaling_factor
            )
        rattled_structure.translate_sites(atom_idx, displacement, frac_coords=False)
    return rattled_structure
