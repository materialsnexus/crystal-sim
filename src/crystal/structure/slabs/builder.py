# noqa: D100
from copy import deepcopy
from enum import StrEnum, auto
from functools import cached_property
from pathlib import Path
from typing import Self, TypeAlias

import numpy as np
import numpy.typing as npt
from monty.json import MontyEncoder
from pydantic import Field
from pymatgen.core.interface import label_termination
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import Species
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.core.surface import Slab as PMG_Slab
from pymatgen.core.surface import SlabGenerator as PMG_SlabGenerator

from crystal.exceptions import UnsupportedTerminationLayerException
from crystal.helpers.json import read_json_to_dict, write_data_to_json
from crystal.schemas.serialization import MSONableBaseModel

Termination: TypeAlias = str


class TerminationLayer(StrEnum):  # noqa: D101
    top = auto()
    bottom = auto()
    both = auto()


class Slab(PMG_Slab):
    """Extend the pymatgen slab."""

    # Unlike `interface_properties` in Interfaces, Slab during construction does not pass any argument for global structure properties
    @property
    def _normalized_c_vector(self) -> npt.NDArray[np.float64]:
        c_vec = np.asarray(self.lattice.matrix[2], dtype=float)
        return np.asarray(c_vec / np.linalg.norm(c_vec), dtype=np.float64)

    @property
    def height_in_angstrom_without_vacuum(self) -> float:
        """Return the plane-to-plane slab extent measured along lattice c."""
        fractional_c_coords = np.asarray(self.frac_coords[:, 2], dtype=float)
        fractional_extent = float(np.ptp(fractional_c_coords))
        return fractional_extent * self.lattice.c

    @property
    def vacuum_thickness_in_angstrom(self) -> float:  # noqa: D102
        return self.lattice.c - self.height_in_angstrom_without_vacuum

    @property
    def minimum_neighbor_distance_in_angstrom(self) -> float:  # noqa: D102
        distance_matrix = self.distance_matrix + np.eye(len(self.distance_matrix)) * 1e6
        return float(np.min(distance_matrix))

    @classmethod
    def read_from_json(cls, filepath: Path) -> Self:  # noqa: D102
        data_as_dict = read_json_to_dict(filepath=filepath)
        return cls.from_dict(data_as_dict)

    @property
    def z_max(self) -> float:  # noqa: D102
        max_value = self._z_coords.max()
        assert isinstance(max_value, float)
        return max_value

    @property
    def z_min(self) -> float:  # noqa: D102
        min_value = self._z_coords.min()
        assert isinstance(min_value, float)
        return min_value

    @property
    def z_center(self) -> float:  # noqa: D102
        return 0.5 * (self.z_max + self.z_min)

    @property
    def _z_coords(self) -> npt.NDArray[np.float64]:
        return np.array([site.coords for site in self.sites])[:, 2]

    def write_to_json(self, filepath: Path) -> None:  # noqa: D102
        write_data_to_json(
            filepath=filepath, data=self.as_dict(), mode="w", encoder=MontyEncoder
        )

    def get_terminal_sites(
        self, depth_from_surface: float
    ) -> dict[TerminationLayer, list[PeriodicSite]]:
        """Return sites that belong to terminal layers near the slab surfaces.

        This differs from `pymatgen`'s `get_surface_sites()` in that it is used
        for multi-element termination inspection in this workflow.

        Args:
            depth_from_surface: Distance in angstrom from each slab boundary used
                to classify sites as top or bottom terminal sites.

        Returns:
            dict[TerminationLayer, list[PeriodicSite]]: Terminal sites grouped by
            `TerminationLayer.top`, `TerminationLayer.bottom`, and
            `TerminationLayer.both` (concatenated top and bottom lists).
        """
        sites_as_np_array = np.asarray(self.sites)
        top_surface_atom_mask = self._z_coords > self.z_max - depth_from_surface
        bottom_surface_atom_mask = self._z_coords < self.z_min + depth_from_surface
        top_surface_sites = sites_as_np_array[top_surface_atom_mask]
        bottom_surface_sites = sites_as_np_array[bottom_surface_atom_mask]
        return {
            TerminationLayer.top: list(top_surface_sites),
            TerminationLayer.bottom: list(bottom_surface_sites),
            TerminationLayer.both: list(top_surface_sites) + list(bottom_surface_sites),
        }

    @property
    def top_layer_termination(self) -> Termination:  # noqa: D102
        return label_termination(self)

    def _new_instance_with_updated_frac_coords(
        self, frac_coords: npt.NDArray[np.float64]
    ) -> Self:
        frac_coords = frac_coords % 1.0
        return type(self)(
            lattice=self.lattice,
            species=self.species,
            coords=frac_coords,
            miller_index=self.miller_index,
            oriented_unit_cell=self.oriented_unit_cell,
            shift=self.shift,
            scale_factor=self.scale_factor,
            site_properties=self.site_properties,
        )

    def _get_rescaled_slab(self, total_thickness_in_angstrom: float) -> Self:
        a, b, cvec = self.lattice.matrix
        c_hat = cvec / np.linalg.norm(cvec)
        rescaled_lattice = Lattice(
            np.vstack([a, b, c_hat * total_thickness_in_angstrom])
        )
        return type(self)(
            lattice=rescaled_lattice,
            species=self.species,
            coords=self.cart_coords,
            miller_index=self.miller_index,
            oriented_unit_cell=self.oriented_unit_cell,
            shift=self.shift,
            scale_factor=self.scale_factor,
            coords_are_cartesian=True,
            site_properties=self.site_properties,
        )

    def strip_vacuum(self) -> Self:  # noqa: D102
        cartesian_coords = np.array(self.cart_coords)
        c_vector = np.asarray(self.lattice.matrix[2], dtype=float)
        min_fractional_c = float(np.min(self.frac_coords[:, 2]))
        shifted_cartesian_coords = cartesian_coords - min_fractional_c * c_vector
        reboxed_lattice_matrix = np.vstack(
            [
                self.lattice.matrix[0],
                self.lattice.matrix[1],
                self._normalized_c_vector * self.height_in_angstrom_without_vacuum,
            ]
        )
        return type(self)(
            lattice=Lattice(reboxed_lattice_matrix),
            species=self.species,
            coords=shifted_cartesian_coords,
            coords_are_cartesian=True,
            miller_index=self.miller_index,
            oriented_unit_cell=self.oriented_unit_cell,
            shift=self.shift,
            scale_factor=self.scale_factor,
            site_properties=self.site_properties,
        )

    def center_slab_in_place(self) -> None:  # noqa: D102
        center_of_mass = self.center_of_mass
        shift = 0.5 - center_of_mass[2]
        self.translate_sites(
            range(len(self)), [0, 0, float(shift)], frac_coords=True, to_unit_cell=False
        )

    def get_centered_slab(self, vacuum_in_angstrom: float | None = None) -> Self:  # noqa: D102
        if vacuum_in_angstrom is None:
            new_slab = self.copy()
            new_slab.center_slab_in_place()
            return new_slab

        slab_thickness_in_angstrom = self.height_in_angstrom_without_vacuum
        total_required_vacuum_thickness_in_angstrom = 2 * vacuum_in_angstrom
        new_box_thickness = (
            slab_thickness_in_angstrom + total_required_vacuum_thickness_in_angstrom
        )
        rescaled_slab = self._get_rescaled_slab(
            total_thickness_in_angstrom=new_box_thickness
        )
        rescaled_slab.center_slab_in_place()
        return rescaled_slab

    @cached_property
    def mirrored(self) -> Self:  # noqa: D102
        xy_mirror_rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        symmetry_operation = SymmOp.from_rotation_and_translation(
            rotation_matrix=xy_mirror_rotation_matrix, translation_vec=[0, 0, 0]
        )
        mirror_frac_coords = symmetry_operation.operate_multi(
            deepcopy(self.frac_coords)
        )
        return self._new_instance_with_updated_frac_coords(
            frac_coords=mirror_frac_coords
        )

    @property
    def bottom_layer_termination(self) -> Termination:  # noqa: D102
        return label_termination(self.mirrored)

    @property
    def orthogonalize_in_xy_plane(self) -> Self:  # noqa: D102
        if not self.is_orthogonal_in_plane:
            return self.get_orthogonal_c_slab()
        return self

    @property
    def is_orthogonal_in_plane(self) -> bool:  # noqa: D102
        return all(
            np.isclose(angle, 90, atol=1e-3)
            for angle in (self.lattice.alpha, self.lattice.beta)
        )

    @property
    def _terminations_dict(self) -> dict[str, Termination]:
        return {
            TerminationLayer.top: self.top_layer_termination,
            TerminationLayer.bottom: self.bottom_layer_termination,
        }

    def terminates_with(  # noqa: D102
        self, termination: Termination, at_layer: TerminationLayer
    ) -> bool:
        terminations_to_check = self._terminations_dict.get(
            at_layer, list(self._terminations_dict.values())
        )
        # convert to a list
        if isinstance(terminations_to_check, str):
            terminations_to_check = [terminations_to_check]

        for termination_to_check in terminations_to_check:
            if termination in termination_to_check:
                return True
        return False

    @classmethod
    def from_pmg_slab(cls, slab: PMG_Slab) -> Self:  # noqa: D102
        obj = cls.__new__(cls)
        obj.__dict__ = slab.__dict__.copy()
        return obj


class SlabFactory(MSONableBaseModel):
    """Generate slab candidates from a bulk structure and slab settings.

    Attributes:
        structure: Bulk structure used as the slab source.
        supercell: Replication factors `(nx, ny, nz)` applied before slab
            generation (`nx`, `ny`) and used as slab thickness layers (`nz`).
        vacuum_thickness_in_layers: Vacuum thickness in layer units passed to the
            slab generator.
        surface_miller_indices: Surface orientation `(h, k, l)` for cleavage.
        orthogonalize_slabs: Whether to request lattice reduction during slab
            generation so output slabs are suitable for downstream matching.
        reduce_slabs_to_primitive: After slab generation, reduces the slabs to their primitive cell
    """

    structure: PMG_Structure
    supercell: tuple[int, int, int]
    vacuum_thickness_in_layers: int
    surface_miller_indices: tuple[int, int, int]
    orthogonalize_slabs: bool = Field(default=True)
    reduce_slabs_to_primitive: bool = Field(default=True)

    @property
    def _xy_expanded_structure(self) -> PMG_Structure:
        assert isinstance(self.structure, PMG_Structure)
        return self.structure.make_supercell(
            scaling_matrix=[self.supercell[0], self.supercell[1], 1], in_place=False
        )

    @property
    def number_of_layers(self) -> int:
        """Assumes the slab thickness is in the z-direction."""
        return self.supercell[2]

    @property
    def _slab_generator(self) -> PMG_SlabGenerator:
        return PMG_SlabGenerator(
            initial_structure=self._xy_expanded_structure,
            miller_index=self.surface_miller_indices,
            min_slab_size=self.number_of_layers,
            min_vacuum_size=self.vacuum_thickness_in_layers,
            lll_reduce=self.orthogonalize_slabs,
            in_unit_planes=True,
            center_slab=True,
            reorient_lattice=True,
            primitive=self.reduce_slabs_to_primitive,
        )

    def generate_slabs(self) -> list[Slab]:
        """Generate slab candidates for the configured bulk and surface settings.

        Returns:
            list[Slab]: Slab candidates returned by the underlying pymatgen slab
            generator, converted to the local `Slab` type.
        """
        return [Slab.from_pmg_slab(slab) for slab in self._slab_generator.get_slabs()]

    def get_all_terminations(self) -> tuple[list[Termination], ...]:  # noqa: D102
        return tuple(
            [slab.top_layer_termination, slab.bottom_layer_termination]
            for slab in self.generate_slabs()
        )


def filter_slabs_by_termination(
    slabs: list[Slab], termination: Termination, at_layer: TerminationLayer
) -> list[Slab]:
    """Filter slab candidates by termination token and layer.

    Args:
        slabs: Slab candidates to filter.
        termination: Substring token matched against the selected termination
            label(s) of each slab.
        at_layer: Which slab side to check (`top`, `bottom`, or `both`).

    Returns:
        list[Slab]: Slabs that match the requested termination condition.
    """
    return [
        slab
        for slab in slabs
        if slab.terminates_with(termination=termination, at_layer=at_layer)
    ]


def passivate_slab(
    initial_slab: Slab,
    passivant: Species,
    bond_length_map: dict[str, float],
    layer: TerminationLayer = TerminationLayer.top,
    depth_from_surface: float = 0.25,  # in Angstroms for atoms to be considered as surface atoms
) -> Slab:
    """Returns a passivated surface slab.

    NOTE: Added as a function since there can be multiple passivation methods and will require additional steps for interfaces
    Can be later refactored into an appropriate class

    (1) Identifies the appropriate surface atoms to passivate
    (2) Checks if the appropriate bond length dict is supplied
    (3) Places passivant at relevant coordinates and returns a new slab
    """
    try:
        TerminationLayer(layer)
    except ValueError as ve:
        err_msg = f"Unsupported termination layer: {layer}"
        raise UnsupportedTerminationLayerException(err_msg) from ve

    surface_sites_as_dict = initial_slab.get_terminal_sites(
        depth_from_surface=depth_from_surface
    )

    # Remove already existing passivant from filtered species
    filtered_surface_sites = [
        site
        for site in surface_sites_as_dict[layer]
        if site.specie.symbol != passivant.symbol
    ]
    unique_element_symbols_in_surface_sites: set[str] = {
        site.specie.symbol for site in filtered_surface_sites
    }
    missing = unique_element_symbols_in_surface_sites.difference(bond_length_map.keys())
    if missing:
        raise ValueError(
            "Missing bond length entries for surface elements: "
            + ", ".join(sorted(missing))
        )
    passivated_slab = deepcopy(initial_slab)
    for site in filtered_surface_sites:
        element_symbol = site.specie.symbol
        bond_length = bond_length_map[element_symbol]
        multiplier = 1 if site.coords[2] > initial_slab.z_center else -1
        cart_coordinates_of_passivated_atom = (
            site.coords + np.array([0, 0, multiplier]) * bond_length
        )
        passivated_slab.append(
            species=passivant,
            coords=cart_coordinates_of_passivated_atom,
            coords_are_cartesian=True,
        )

    return type(initial_slab).from_dict(passivated_slab.as_dict())
