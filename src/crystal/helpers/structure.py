# noqa: D100
import itertools
import logging
from collections.abc import Sequence
from copy import deepcopy
from typing import TypeAlias, TypeVar

import ase
import numpy as np
from ase.constraints import FixAtoms
from numpy.typing import NDArray
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.core.lattice import Lattice as PMG_Lattice
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from crystal.schemas.relaxation_constraints import RelaxationConstraints

DEFAULT_ANGLE_TOLERANCE_FOR_PLANE_VERSUS_C = 1e-3
Periodicity: TypeAlias = tuple[bool, bool, bool]
DEFAULT_PERIODICITY: Periodicity = (True, True, True)

logger = logging.getLogger(__name__)

BatchItemT = TypeVar("BatchItemT")


def get_ase_atoms_midpoint_along_c_axis(atoms: ase.Atoms) -> float:  # noqa: D103
    height_along_c_axis = (
        RelaxationConstraints.compute_atom_coordinates_projected_along_c_axis(atoms)
    )
    return float(0.5 * (height_along_c_axis.min() + height_along_c_axis.max()))


def get_centered_ase_atoms_along_c_axis(atoms: ase.Atoms) -> ase.Atoms:  # noqa: D103
    centered = atoms.copy()  # type: ignore[no-untyped-call]
    c_vector = centered.cell[2]
    c_hat = c_vector / np.linalg.norm(c_vector)
    midpoint = get_ase_atoms_midpoint_along_c_axis(centered)
    shift = -midpoint * c_hat
    centered.translate(shift)
    assert isinstance(centered, ase.Atoms)
    return centered


def is_structure_c_axis_orthogonal_to_ab_plane(  # noqa: D103
    structure: PMG_Structure,
    tolerance: float = DEFAULT_ANGLE_TOLERANCE_FOR_PLANE_VERSUS_C,
) -> bool:
    return bool(
        np.isclose(structure.lattice.alpha, 90.0, atol=tolerance)
        and np.isclose(structure.lattice.beta, 90.0, atol=tolerance)
    )


def _get_periodic_image_translations(
    periodicity: Periodicity,
) -> list[tuple[int, int, int]]:
    a_range = (-1, 0, 1) if periodicity[0] else (0,)
    b_range = (-1, 0, 1) if periodicity[1] else (0,)
    c_range = (-1, 0, 1) if periodicity[2] else (0,)
    return [
        image
        for image in list(itertools.product(a_range, b_range, c_range))
        if image != (0, 0, 0)
    ]


def modify_structure_periodicity(
    structure: PMG_Structure, periodicity: Periodicity = DEFAULT_PERIODICITY
) -> PMG_Structure:
    """Return a copy of a structure with updated lattice periodicity."""
    lattice = PMG_Lattice(matrix=structure.lattice.matrix, pbc=periodicity)
    return PMG_Structure(
        lattice=lattice,
        species=structure.species,
        coords=structure.frac_coords,
        coords_are_cartesian=False,
        site_properties=structure.site_properties,
    )


def get_minimum_interatomic_distance(structure: PMG_Structure) -> float:
    """Get minimum interatomic distance between two atoms in the structure.

    The structure periodicity is used to compute the image distances.
    """
    periodicity = getattr(structure.lattice, "pbc", DEFAULT_PERIODICITY)
    updated_structure = modify_structure_periodicity(structure, periodicity=periodicity)

    distance_with_other_atoms = updated_structure.distance_matrix[
        np.triu_indices(len(structure), 1)
    ]
    if distance_with_other_atoms.size > 0:
        min_distance_with_other_atoms = float(distance_with_other_atoms.min())
    else:
        min_distance_with_other_atoms = float("inf")

    periodic_images = _get_periodic_image_translations(periodicity=periodicity)
    min_self_distance = float("inf")
    for coords in updated_structure.frac_coords:
        for image in periodic_images:
            self_distance = updated_structure.lattice.get_distance_and_image(
                coords, coords, jimage=image
            )[0]
            min_self_distance = min(min_self_distance, self_distance)
    return min(min_distance_with_other_atoms, min_self_distance)


def get_maximum_box_aspect_ratio(structure: PMG_Structure) -> float:  # noqa: D103
    lengths = np.array(structure.lattice.abc)
    return float(lengths.max() / lengths.min())


def passes_geometric_structure_filter(
    *,
    minimum_interatomic_distance: float,
    maximum_box_aspect_ratio: float,
    minimum_allowed_interatomic_distance: float,
    maximum_allowed_box_aspect_ratio: float,
) -> bool:
    """Return whether geometric structure metrics satisfy filter thresholds."""
    return (
        minimum_interatomic_distance >= minimum_allowed_interatomic_distance
        and maximum_box_aspect_ratio <= maximum_allowed_box_aspect_ratio
    )


def apply_constraints_to_atoms(  # noqa: D103
    atoms: ase.Atoms, constraints: RelaxationConstraints | None
) -> ase.Atoms:
    atoms = deepcopy(atoms)
    fixed_atom_indices = (
        [] if constraints is None else constraints.build_fixed_atom_indices(atoms)
    )
    if fixed_atom_indices:
        constraint = FixAtoms(indices=fixed_atom_indices)  # type: ignore[no-untyped-call]
        atoms.set_constraint(constraint=constraint)  # type: ignore[no-untyped-call]
    return atoms


def get_strained_structures(  # noqa: D103
    structure: PMG_Structure, strains: NDArray[np.float64]
) -> list[PMG_Structure]:
    logger.info("Generating structures with strain %s", list(strains))

    deformations = [(np.identity(3) * (1.0 + eps)).tolist() for eps in strains]
    strained_structures: list[PMG_Structure] = []
    for deformation in deformations:
        deform_structure = DeformStructureTransformation(deformation=deformation)  # type:ignore[no-untyped-call]
        transformed_structure = TransformedStructure(
            structure, transformations=[deform_structure]
        )
        strained_structures.append(transformed_structure.final_structure)
    return strained_structures


def split_items_into_batches(  # noqa: D103
    items: Sequence[BatchItemT], items_per_batch: int | None
) -> list[list[BatchItemT]]:
    if items_per_batch is None:
        return [list(items)]
    if items_per_batch <= 0:
        msg = "items_per_batch must be a positive integer or None."
        raise ValueError(msg)
    return [
        list(items[index : index + items_per_batch])
        for index in range(0, len(items), items_per_batch)
    ]
