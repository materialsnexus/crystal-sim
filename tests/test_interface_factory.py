import math
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import ase
import numpy as np
import pytest
from monty.json import MontyEncoder
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import Species
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.core.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.exceptions import OverlappingAtomsException
from crystal.helpers.json import write_data_to_json
from crystal.helpers.temp import run_in_temp_dir
from crystal.structure.interfaces.builder import (
    Interface,
    InterfaceFactory,
    InterfaceMatchPolicy,
    RequiredInterfaceFieldsInStructure,
)
from crystal.structure.slabs.builder import Slab, TerminationLayer, passivate_slab
from tests.helpers.constants.structure import (
    DEFAULT_INTERFACE_MATCH_MAX_AREA,
    FLOATING_POINT_TOLERANCE,
    SLAB_CENTER_FRACTION,
    TEST_001_FE_SLAB,
    TEST_111_FE_SLAB,
    TEST_INTERFACE_FACTORY,
    TEST_LINBO3_SLAB,
)
from tests.helpers.test_paths import (
    TEST_INTERFACE_JSON_FILENAME,
    TEST_INTERFACE_NO_MONTY_JSON_FILENAME,
)


def test_orthogonalize() -> None:
    op = SymmOp.from_rotation_and_translation(
        rotation_matrix=([1, 0, 0], [0, 1, 0], [0, 0.1, 1]), translation_vec=(0, 0, 0)
    )
    non_xy_orthogonal_slab = deepcopy(TEST_001_FE_SLAB).apply_operation(op)
    assert not math.isclose(non_xy_orthogonal_slab.lattice.alpha, 90.0, abs_tol=1e-3)
    assert math.isclose(
        non_xy_orthogonal_slab.orthogonalize_in_xy_plane.lattice.alpha,
        90.0,
        abs_tol=1e-3,
    )
    interface_factory = InterfaceFactory(
        top_slab=non_xy_orthogonal_slab, bottom_slab=non_xy_orthogonal_slab
    )
    assert isinstance(interface_factory.top_slab, Slab)
    assert isinstance(interface_factory.bottom_slab, Slab)

    assert math.isclose(interface_factory.top_slab.lattice.alpha, 90.0, abs_tol=1e-3)
    assert math.isclose(interface_factory.bottom_slab.lattice.alpha, 90.0, abs_tol=1e-3)


@patch("crystal.structure.interfaces.builder.Interface.is_valid", return_value=False)
def test_overlapping_atoms_error(mock_invalid_interface: MagicMock) -> None:  # noqa: ARG001
    interface_factory = deepcopy(TEST_INTERFACE_FACTORY)
    with pytest.raises(
        OverlappingAtomsException, match="Overlapping atoms found in interface"
    ):
        _ = list(interface_factory.generate_interfaces())


def test_interface_with_bcc_fe_slabs() -> None:
    interface_factory = InterfaceFactory(
        top_slab=deepcopy(TEST_001_FE_SLAB), bottom_slab=deepcopy(TEST_001_FE_SLAB)
    )
    [interface] = list(interface_factory.generate_interfaces())
    assert len(interface) == len(interface_factory.top_slab) + len(
        interface_factory.bottom_slab
    )
    assert np.isclose(interface_factory.estimated_vacuum_gap_in_angstrom, 2.72, 1e-1)
    assert isinstance(interface_factory.top_slab, Slab)
    assert isinstance(interface_factory.bottom_slab, Slab)
    total_size_of_top_and_bottom_slab_without_vacuum = (
        interface_factory.top_slab.height_in_angstrom_without_vacuum
        + interface_factory.bottom_slab.height_in_angstrom_without_vacuum
    )
    # interface also includes vacuum and gap
    assert interface.lattice.c > total_size_of_top_and_bottom_slab_without_vacuum


def test_interface_with_different_miller_index() -> None:
    interface_factory = InterfaceFactory(
        top_slab=deepcopy(TEST_001_FE_SLAB),
        bottom_slab=deepcopy(TEST_111_FE_SLAB),
        interface_match_policy=InterfaceMatchPolicy(
            maximum_area_in_angstrom_square=DEFAULT_INTERFACE_MATCH_MAX_AREA
        ),
    )
    assert isinstance(interface_factory.top_slab, Slab)
    assert isinstance(interface_factory.bottom_slab, Slab)
    assert interface_factory.top_slab.bottom_layer_termination == "Fe_P4/mmm_1"
    assert interface_factory.bottom_slab.top_layer_termination == "Fe_P6/mmm_1"
    interfaces = list(interface_factory.generate_interfaces())
    # multiple interfaces due to different combinations (shear)
    assert len(interfaces) == 6
    assert all(
        math.isclose(
            interface.interface_properties["von_mises_strain"], 0.001, abs_tol=1e-3
        )
        for interface in interfaces
    )
    assert all(
        math.isclose(
            interface.interface_properties["match_area_in_angstrom_square"],
            155.41,
            abs_tol=1e-2,
        )
        for interface in interfaces
    )

    all_interface_terminations = {
        interface.interface_properties["terminations"] for interface in interfaces
    }
    assert all_interface_terminations == {("Fe_P4/mmm_1", "Fe_P6/mmm_1")}


def test_interface_with_different_top_and_bottom_structures() -> None:
    interface_factory = InterfaceFactory(
        top_slab=deepcopy(TEST_001_FE_SLAB),
        bottom_slab=deepcopy(TEST_LINBO3_SLAB),
        interface_match_policy=InterfaceMatchPolicy(
            maximum_area_in_angstrom_square=DEFAULT_INTERFACE_MATCH_MAX_AREA
        ),
    )
    assert isinstance(interface_factory.top_slab, Slab)
    assert isinstance(interface_factory.bottom_slab, Slab)
    assert interface_factory.top_slab.bottom_layer_termination == "Fe_P4/mmm_1"
    assert interface_factory.bottom_slab.top_layer_termination == "Nb_Cmmm_1"
    interfaces = list(interface_factory.generate_interfaces())
    assert len(interfaces) == 8
    assert all(
        math.isclose(
            interface.interface_properties["von_mises_strain"], 0.004, abs_tol=1e-3
        )
        for interface in interfaces
    )
    assert all(
        math.isclose(
            interface.interface_properties["match_area_in_angstrom_square"],
            253.567,
            abs_tol=1e-2,
        )
        for interface in interfaces
    )
    all_interface_terminations = {
        interface.interface_properties["terminations"] for interface in interfaces
    }
    assert all_interface_terminations == {("Fe_P4/mmm_1", "Nb_Cmmm_1")}


def test_interface_with_passivated_linbo3_structure_on_top_and_unpassivated_fe_structure_on_bottom() -> (
    None
):
    passivated_linbo3_slab = passivate_slab(
        initial_slab=deepcopy(TEST_LINBO3_SLAB),
        passivant=Species("H"),
        bond_length_map={"Li": 1.2, "Nb": 2, "O": 0.8},
        layer=TerminationLayer.top,
    )
    interface_factory = InterfaceFactory(
        top_slab=passivated_linbo3_slab,
        bottom_slab=deepcopy(TEST_001_FE_SLAB),
        interface_match_policy=InterfaceMatchPolicy(
            maximum_area_in_angstrom_square=DEFAULT_INTERFACE_MATCH_MAX_AREA
        ),
    )
    assert isinstance(interface_factory.top_slab, Slab)
    assert isinstance(interface_factory.bottom_slab, Slab)
    assert interface_factory.bottom_slab.top_layer_termination == "Fe_P4/mmm_1"
    assert interface_factory.top_slab.bottom_layer_termination == "O2_Cmmm_1"
    interfaces = list(interface_factory.generate_interfaces())
    assert len(interfaces) == 8
    assert all(
        math.isclose(
            interface.interface_properties["von_mises_strain"], 0.005, abs_tol=1e-2
        )
        for interface in interfaces
    )
    assert all(
        math.isclose(
            interface.interface_properties["match_area_in_angstrom_square"],
            252.254,
            abs_tol=1e-2,
        )
        for interface in interfaces
    )
    all_interface_terminations = {
        interface.interface_properties["terminations"] for interface in interfaces
    }
    assert all_interface_terminations == {("O2_Cmmm_1", "Fe_P4/mmm_1")}

    # assert the passivated atoms are always at the top
    for interface in interfaces:
        max_z_of_all_species_except_h = max(
            site.frac_coords[2]
            for site in interface.film.sites
            if site.specie.symbol != "H"
        )
        min_z_of_all_h_species = min(
            site.frac_coords[2]
            for site in interface.film.sites
            if site.specie.symbol == "H"
        )
        assert max_z_of_all_species_except_h < min_z_of_all_h_species


@run_in_temp_dir
def test_interface_io() -> None:
    interface_factory = deepcopy(TEST_INTERFACE_FACTORY)

    interface_json_file = Path(TEST_INTERFACE_JSON_FILENAME)
    assert not interface_json_file.is_file()
    [example_interface] = list(interface_factory.generate_interfaces())
    example_interface.write_to_json(filepath=interface_json_file)

    assert interface_json_file.is_file()

    interface_from_file = Interface.read_from_json(interface_json_file)
    assert isinstance(interface_from_file, Interface)
    assert StructureMatcher().fit(interface_from_file, example_interface)


@run_in_temp_dir
def test_read_from_file_without_monty_serialized_dict_items() -> None:
    interface_factory = deepcopy(TEST_INTERFACE_FACTORY)
    interface_json_file = Path(TEST_INTERFACE_JSON_FILENAME)
    assert not interface_json_file.is_file()
    [example_interface] = list(interface_factory.generate_interfaces())

    interface_as_dict = example_interface.as_dict()
    interface_as_dict_without_monty_serialized_objects = {
        k: v for k, v in interface_as_dict.items() if "@" not in k
    }

    temp_file = Path(TEST_INTERFACE_NO_MONTY_JSON_FILENAME)
    # Monty Encoder is necessary as numpy ndarray is not JSON serializable,
    # could add a default serializer if this is not desirable
    write_data_to_json(
        temp_file,
        data=interface_as_dict_without_monty_serialized_objects,
        encoder=MontyEncoder,
    )

    interface_from_file = Interface.read_from_json(temp_file)
    assert isinstance(interface_from_file, Interface)


def test_get_slabs_from_interface() -> None:
    interface_factory = InterfaceFactory(
        top_slab=deepcopy(TEST_001_FE_SLAB),
        bottom_slab=deepcopy(TEST_LINBO3_SLAB),
        interface_match_policy=InterfaceMatchPolicy(
            maximum_area_in_angstrom_square=DEFAULT_INTERFACE_MATCH_MAX_AREA
        ),
    )
    [*_, test_interface] = list(interface_factory.generate_interfaces())
    top_slab = test_interface.get_centered_top_slab()
    assert isinstance(top_slab, Slab)
    assert np.isclose(
        top_slab.center_of_mass[2], SLAB_CENTER_FRACTION, atol=FLOATING_POINT_TOLERANCE
    )
    assert np.isclose(
        top_slab.lattice.c, test_interface.lattice.c, atol=FLOATING_POINT_TOLERANCE
    )
    assert interface_factory.top_slab.symbol_set == top_slab.symbol_set

    bottom_slab = test_interface.get_centered_bottom_slab()
    assert isinstance(bottom_slab, Slab)
    assert np.isclose(
        bottom_slab.center_of_mass[2],
        SLAB_CENTER_FRACTION,
        atol=FLOATING_POINT_TOLERANCE,
    )
    assert np.isclose(
        bottom_slab.lattice.c, test_interface.lattice.c, atol=FLOATING_POINT_TOLERANCE
    )
    assert interface_factory.bottom_slab.symbol_set == bottom_slab.symbol_set

    top_slab = test_interface.get_centered_top_slab(vacuum_in_angstrom=10)
    assert np.isclose(
        top_slab.center_of_mass[2], SLAB_CENTER_FRACTION, atol=FLOATING_POINT_TOLERANCE
    )
    assert not np.isclose(
        top_slab.lattice.c, test_interface.lattice.c, atol=FLOATING_POINT_TOLERANCE
    )
    assert np.isclose(
        top_slab.lattice.c,
        top_slab.height_in_angstrom_without_vacuum + 2 * 10,
        atol=FLOATING_POINT_TOLERANCE,
    )
    assert top_slab.miller_index == (0, 0, 1)
    assert interface_factory.top_slab.symbol_set == top_slab.symbol_set

    bottom_slab = test_interface.get_centered_bottom_slab(vacuum_in_angstrom=10)
    assert np.isclose(
        bottom_slab.center_of_mass[2],
        SLAB_CENTER_FRACTION,
        atol=FLOATING_POINT_TOLERANCE,
    )
    assert bottom_slab.miller_index == (0, 0, 1)
    assert not np.isclose(
        bottom_slab.lattice.c, test_interface.lattice.c, atol=FLOATING_POINT_TOLERANCE
    )
    assert np.isclose(
        bottom_slab.lattice.c,
        bottom_slab.height_in_angstrom_without_vacuum + 2 * 10,
        atol=FLOATING_POINT_TOLERANCE,
    )
    assert interface_factory.bottom_slab.symbol_set == bottom_slab.symbol_set


def test_centered_top_slab_preserves_distances() -> None:
    interface_factory = InterfaceFactory(
        top_slab=deepcopy(TEST_001_FE_SLAB),
        bottom_slab=deepcopy(TEST_LINBO3_SLAB),
        interface_match_policy=InterfaceMatchPolicy(
            maximum_area_in_angstrom_square=DEFAULT_INTERFACE_MATCH_MAX_AREA
        ),
    )
    [*_, test_interface] = list(interface_factory.generate_interfaces())
    original_distance_matrix = test_interface.film.distance_matrix.copy()
    centered_top_slab = test_interface.get_centered_top_slab()
    assert np.allclose(
        centered_top_slab.distance_matrix,
        original_distance_matrix,
        atol=FLOATING_POINT_TOLERANCE,
    )
    assert np.isclose(
        centered_top_slab.minimum_neighbor_distance_in_angstrom,
        np.min(
            original_distance_matrix + np.eye(original_distance_matrix.shape[0]) * 1e6
        ),
        atol=FLOATING_POINT_TOLERANCE,
    )


def test_interface_from_structure() -> None:
    [*_, test_interface] = list(
        deepcopy(TEST_INTERFACE_FACTORY).generate_interfaces(
            gap=3.0, in_plane_offset=(2.0, 3.0)
        )
    )
    assert isinstance(test_interface, Interface)
    interface_properties = test_interface.interface_properties
    structure_properties = test_interface.properties
    assert all(
        key in interface_properties
        for key in RequiredInterfaceFieldsInStructure.values()
    )
    assert all(
        key in structure_properties
        for key in RequiredInterfaceFieldsInStructure.values()
    )

    ase_atom_adaptor = AseAtomsAdaptor()
    ase_atoms: ase.Atoms = ase_atom_adaptor.get_atoms(test_interface)
    structure_from_ase_atoms: PMG_Structure = ase_atom_adaptor.get_structure(ase_atoms)
    recreated_interface_from_structure = Interface.from_structure(
        structure_from_ase_atoms
    )
    assert recreated_interface_from_structure.gap == pytest.approx(
        test_interface.gap, abs=FLOATING_POINT_TOLERANCE
    )
    assert recreated_interface_from_structure.in_plane_offset == pytest.approx(
        test_interface.in_plane_offset, abs=FLOATING_POINT_TOLERANCE
    )
    assert np.allclose(
        recreated_interface_from_structure.frac_coords,
        test_interface.frac_coords,
        atol=FLOATING_POINT_TOLERANCE,
    )
