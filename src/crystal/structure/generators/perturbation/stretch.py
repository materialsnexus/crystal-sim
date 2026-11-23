"""Functions for stretching the lattices of structures."""

import logging
from copy import deepcopy
from typing import Annotated

import numpy as np
from pydantic import Field, PositiveFloat, validate_call
from pymatgen.core.elasticity.strain import Deformation
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.core.tensors import Tensor

from crystal.schemas.arrays import FloatArray

MINIMUM_STRAIN_FRACTION: PositiveFloat = 1e-3
MINIMUM_VALID_DEFORMATION_DETERMINANT: PositiveFloat = 1e-8
DEFAULT_MAX_NORMAL_STRAIN_FRACTION: PositiveFloat = 0.8
DEFAULT_MAX_SHEAR_STRAIN_FRACTION: PositiveFloat = 0.2

logger = logging.getLogger(__name__)


def sample_signed_strains(
    *, max_strain_fraction: float, size: int, rng: np.random.Generator
) -> FloatArray:
    """Helper for `stretch_cell`. Implemented as in ASSYST repo."""
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=size)
    magnitudes = rng.uniform(MINIMUM_STRAIN_FRACTION, max_strain_fraction, size=size)
    return np.asarray(signs * magnitudes, dtype=np.float64)


@validate_call
def stretch_cell(
    structure: PMG_Structure,
    random_seed: int,
    max_normal_strain_fraction: Annotated[
        PositiveFloat,
        Field(default=DEFAULT_MAX_NORMAL_STRAIN_FRACTION, ge=MINIMUM_STRAIN_FRACTION),
    ],
    max_shear_strain_fraction: Annotated[
        PositiveFloat,
        Field(default=DEFAULT_MAX_SHEAR_STRAIN_FRACTION, ge=MINIMUM_STRAIN_FRACTION),
    ],
) -> PMG_Structure | None:
    """Randomly stretch the input structure using the scheme from the ASSYST repo."""
    rng = np.random.default_rng(seed=random_seed)

    normal_strain_fraction = 1 + sample_signed_strains(
        max_strain_fraction=max_normal_strain_fraction, size=3, rng=rng
    )
    shear_strain_fraction = sample_signed_strains(
        max_strain_fraction=max_shear_strain_fraction, size=3, rng=rng
    )
    deformation_voigt = np.concatenate([normal_strain_fraction, shear_strain_fraction])
    deformation_matrix = Tensor.from_voigt(deformation_voigt)
    deformation_determinant = float(np.linalg.det(np.asarray(deformation_matrix)))

    if deformation_determinant <= MINIMUM_VALID_DEFORMATION_DETERMINANT:
        msg = (
            "Sampled deformation matrix is invalid: determinant must be positive and "
            f"greater than {MINIMUM_VALID_DEFORMATION_DETERMINANT}, got "
            f"{deformation_determinant}."
        )
        logger.warning(msg)
        return None

    deformed_structure = Deformation(deformation_matrix).apply_to_structure(
        deepcopy(structure)
    )
    assert isinstance(deformed_structure, PMG_Structure)
    return deformed_structure
