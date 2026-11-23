from copy import deepcopy

import ase
import numpy as np
import numpy.typing as npt
import pytest
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.structure.interfaces.assembler import (
    MIN_INTERFACE_PADDING_ANGSTROM,
    InterfaceAssembler,
)
from crystal.structure.interfaces.builder import (
    BOTTOM_SLAB_KEY,
    TOP_SLAB_KEY,
    Interface,
    InterfaceFactory,
    InterfaceMatchPolicy,
)
from crystal.structure.interfaces.utils import build_in_plane_offset_grid
from crystal.structure.slabs.builder import SlabFactory
from tests.helpers.constants.structure import (
    DEFAULT_INTERFACE_MATCH_MAX_AREA,
    FLOATING_POINT_TOLERANCE,
    RELAXED_CIFS_CELL_BOTTOM_PATH,
    RELAXED_CIFS_CELL_TOP_PATH,
    TEST_001_FE_SLAB,
    TEST_BOTTOM_SLAB,
    TEST_LINBO3_SLAB,
    TEST_TOP_SLAB,
)

DEFAULT_GAP_VALUES = (1.0, 2.0)
DEFAULT_GAP_SINGLE = 1.0
DEFAULT_OFFSETS = (2, 2)
RELAXED_OFFSETS = (3, 3)
NORMALIZATION_OFFSETS = [(0.0, 0.0), (1.0, -1.0)]
ZERO_OFFSET = (0.0, 0.0)
ZERO_VACUUM = 0.0
OFFSET_RANGE = (0.0, 1.0)
RELAXED_MATCH_TOLERANCE = 0.1
RELAXED_SLAB_SUPERCELL = (1, 1, 2)
RELAXED_SLAB_VACUUM_LAYERS = 2
RELAXED_SLAB_MILLER_INDEX = (0, 0, 1)
INTERFACE_LABEL_KEY = "interface_label"
IN_PLANE_OFFSET_KEY = "in_plane_offset"


def cartesian_distance_matrix(
    coords: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return np.asarray(distances, dtype=float)


def assert_intraslab_distances_preserved(
    assembled_cart: npt.NDArray[np.float64],
    top_indices: list[int],
    bottom_indices: list[int],
    top_cart: npt.NDArray[np.float64],
    bottom_cart: npt.NDArray[np.float64],
) -> None:
    top_subset_cart = assembled_cart[top_indices]
    bottom_subset_cart = assembled_cart[bottom_indices]
    top_subset_distances = cartesian_distance_matrix(top_subset_cart)
    bottom_subset_distances = cartesian_distance_matrix(bottom_subset_cart)
    top_distances = cartesian_distance_matrix(top_cart)
    bottom_distances = cartesian_distance_matrix(bottom_cart)
    assert np.allclose(
        top_subset_distances, top_distances, atol=FLOATING_POINT_TOLERANCE
    )
    assert np.allclose(
        bottom_subset_distances, bottom_distances, atol=FLOATING_POINT_TOLERANCE
    )


def test_interface_assembler_counts_and_lattice() -> None:
    top_slab = deepcopy(TEST_001_FE_SLAB)
    bottom_slab = deepcopy(TEST_001_FE_SLAB)
    assembler = InterfaceAssembler(top_slab=top_slab, bottom_slab=bottom_slab)

    in_plane_offsets = build_in_plane_offset_grid(DEFAULT_OFFSETS)
    gap_values = list(DEFAULT_GAP_VALUES)
    interfaces = list(
        assembler.generate_interfaces(
            offsets=in_plane_offsets, gaps=gap_values, vacuum=ZERO_VACUUM
        )
    )

    assert len(interfaces) == len(in_plane_offsets) * len(gap_values)

    interface = interfaces[0]
    assert len(interface) == len(top_slab) + len(bottom_slab)
    assert np.isclose(
        interface.lattice.a, top_slab.lattice.a, atol=FLOATING_POINT_TOLERANCE
    )
    assert np.isclose(
        interface.lattice.b, top_slab.lattice.b, atol=FLOATING_POINT_TOLERANCE
    )
    expected_c = (
        top_slab.height_in_angstrom_without_vacuum
        + bottom_slab.height_in_angstrom_without_vacuum
        + gap_values[0]
        + MIN_INTERFACE_PADDING_ANGSTROM
    )
    assert np.isclose(interface.lattice.c, expected_c, atol=FLOATING_POINT_TOLERANCE)
    observed_offsets = {
        tuple(candidate.interface_properties[IN_PLANE_OFFSET_KEY])
        for candidate in interfaces
    }
    assert observed_offsets == set(in_plane_offsets)
    assert all(
        OFFSET_RANGE[0] <= offset <= OFFSET_RANGE[1]
        for in_plane_offset in observed_offsets
        for offset in in_plane_offset
    )


def test_interface_assembler_normalizes_offsets() -> None:
    top_slab = deepcopy(TEST_001_FE_SLAB)
    bottom_slab = deepcopy(TEST_001_FE_SLAB)
    assembler = InterfaceAssembler(top_slab=top_slab, bottom_slab=bottom_slab)
    interfaces = list(
        assembler.generate_interfaces(
            offsets=NORMALIZATION_OFFSETS, gaps=[DEFAULT_GAP_SINGLE], vacuum=ZERO_VACUUM
        )
    )
    normalized_offsets = [
        tuple(candidate.interface_properties[IN_PLANE_OFFSET_KEY])
        for candidate in interfaces
    ]
    assert normalized_offsets == [ZERO_OFFSET, ZERO_OFFSET]


def test_interface_assembler_preserves_intraslab_distances() -> None:
    top_slab = deepcopy(TEST_001_FE_SLAB)
    bottom_slab = deepcopy(TEST_001_FE_SLAB)
    assembler = InterfaceAssembler(top_slab=top_slab, bottom_slab=bottom_slab)
    assembled = next(
        iter(
            assembler.generate_interfaces(
                offsets=[ZERO_OFFSET], gaps=[DEFAULT_GAP_SINGLE], vacuum=ZERO_VACUUM
            )
        )
    )
    labels = assembled.site_properties[INTERFACE_LABEL_KEY]
    film_indices = [idx for idx, label in enumerate(labels) if label == TOP_SLAB_KEY]
    substrate_indices = [
        idx for idx, label in enumerate(labels) if label == BOTTOM_SLAB_KEY
    ]
    assembled_cart = assembled.cart_coords
    top_cart = top_slab.cart_coords
    bottom_cart = bottom_slab.cart_coords
    assert_intraslab_distances_preserved(
        assembled_cart=assembled_cart,
        top_indices=film_indices,
        bottom_indices=substrate_indices,
        top_cart=top_cart,
        bottom_cart=bottom_cart,
    )


def test_interface_assembler_preserves_centered_slab_cartesian_distances() -> None:
    interface_factory = InterfaceFactory(
        top_slab=deepcopy(TEST_001_FE_SLAB),
        bottom_slab=deepcopy(TEST_LINBO3_SLAB),
        interface_match_policy=InterfaceMatchPolicy(
            maximum_area_in_angstrom_square=DEFAULT_INTERFACE_MATCH_MAX_AREA
        ),
    )
    [*_, interface] = list(interface_factory.generate_interfaces())
    top_slab = interface._get_individual_slab(interface.film)
    rescaled_top = top_slab._get_rescaled_slab(
        total_thickness_in_angstrom=top_slab.height_in_angstrom_without_vacuum
    )
    centered_top = interface.get_centered_top_slab(vacuum_in_angstrom=ZERO_VACUUM)
    rescaled_cart = rescaled_top.cart_coords
    centered_cart = centered_top.cart_coords
    rescaled_distances = cartesian_distance_matrix(rescaled_cart)
    centered_distances = cartesian_distance_matrix(centered_cart)
    assert np.allclose(
        centered_distances, rescaled_distances, atol=FLOATING_POINT_TOLERANCE
    )


def test_interface_assembler_with_relaxed_cell_cifs() -> None:
    top_structure = PMG_Structure.from_file(filename=RELAXED_CIFS_CELL_TOP_PATH)
    bottom_structure = PMG_Structure.from_file(filename=RELAXED_CIFS_CELL_BOTTOM_PATH)
    top_slab = SlabFactory(
        structure=top_structure,
        supercell=RELAXED_SLAB_SUPERCELL,
        vacuum_thickness_in_layers=RELAXED_SLAB_VACUUM_LAYERS,
        surface_miller_indices=RELAXED_SLAB_MILLER_INDEX,
    ).generate_slabs()[0]
    bottom_slab = SlabFactory(
        structure=bottom_structure,
        supercell=RELAXED_SLAB_SUPERCELL,
        vacuum_thickness_in_layers=RELAXED_SLAB_VACUUM_LAYERS,
        surface_miller_indices=RELAXED_SLAB_MILLER_INDEX,
    ).generate_slabs()[0]
    interface_factory = InterfaceFactory(
        top_slab=top_slab,
        bottom_slab=bottom_slab,
        interface_match_policy=InterfaceMatchPolicy(
            maximum_area_ratio_tolerance=RELAXED_MATCH_TOLERANCE,
            maximum_area_in_angstrom_square=DEFAULT_INTERFACE_MATCH_MAX_AREA,
            length_tolerance_in_angstrom=RELAXED_MATCH_TOLERANCE,
            angle_tolerance_in_radian=RELAXED_MATCH_TOLERANCE,
        ),
    )
    interfaces = list(interface_factory.generate_interfaces())
    assert interfaces
    interface = interfaces[0]
    assert interface.is_valid()
    centered_top = interface.get_centered_top_slab(vacuum_in_angstrom=ZERO_VACUUM)
    centered_bottom = interface.get_centered_bottom_slab(vacuum_in_angstrom=ZERO_VACUUM)
    assert centered_top.composition == interface.film.composition
    assert centered_bottom.composition == interface.substrate.composition
    assembler = InterfaceAssembler(top_slab=centered_top, bottom_slab=centered_bottom)
    in_plane_offsets = build_in_plane_offset_grid(RELAXED_OFFSETS)
    assembled_interfaces = list(
        assembler.generate_interfaces(
            offsets=in_plane_offsets,
            gaps=[assembler.estimated_vacuum_gap_in_angstrom],
            vacuum=ZERO_VACUUM,
        )
    )
    assert all(candidate.is_valid() for candidate in assembled_interfaces)
    assembled = assembled_interfaces[0]
    assert np.isclose(
        assembled.lattice.a, centered_top.lattice.a, atol=FLOATING_POINT_TOLERANCE
    )
    assert np.isclose(
        assembled.lattice.b, centered_top.lattice.b, atol=FLOATING_POINT_TOLERANCE
    )
    labels = assembled.site_properties[INTERFACE_LABEL_KEY]
    film_indices = [idx for idx, label in enumerate(labels) if label == TOP_SLAB_KEY]
    substrate_indices = [
        idx for idx, label in enumerate(labels) if label == BOTTOM_SLAB_KEY
    ]
    assembled_cart = assembled.cart_coords
    top_cart = centered_top.cart_coords
    bottom_cart = centered_bottom.cart_coords
    assert_intraslab_distances_preserved(
        assembled_cart=assembled_cart,
        top_indices=film_indices,
        bottom_indices=substrate_indices,
        top_cart=top_cart,
        bottom_cart=bottom_cart,
    )


def test_assembled_interface_round_trips_through_ase_and_recreates_interface() -> None:
    assembler = InterfaceAssembler(top_slab=TEST_TOP_SLAB, bottom_slab=TEST_BOTTOM_SLAB)
    interfaces = list(
        assembler.generate_interfaces(
            offsets=NORMALIZATION_OFFSETS, gaps=[DEFAULT_GAP_SINGLE], vacuum=ZERO_VACUUM
        )
    )
    assert all(isinstance(interface, Interface) for interface in interfaces)
    ase_atom_adaptor = AseAtomsAdaptor()
    for interface in interfaces:
        ase_atoms: ase.Atoms = ase_atom_adaptor.get_atoms(interface)
        structure_from_ase_atoms: PMG_Structure = ase_atom_adaptor.get_structure(
            ase_atoms
        )
        recreated_interface_from_structure = Interface.from_structure(
            structure_from_ase_atoms
        )
        assert recreated_interface_from_structure.gap == pytest.approx(
            interface.gap, abs=FLOATING_POINT_TOLERANCE
        )
        assert recreated_interface_from_structure.in_plane_offset == pytest.approx(
            interface.in_plane_offset, abs=FLOATING_POINT_TOLERANCE
        )
        assert np.allclose(
            recreated_interface_from_structure.frac_coords,
            interface.frac_coords,
            atol=FLOATING_POINT_TOLERANCE,
        )
