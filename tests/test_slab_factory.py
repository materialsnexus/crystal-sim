from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Species
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.core.structure_matcher import StructureMatcher

from crystal.exceptions import UnsupportedTerminationLayerException
from crystal.helpers.json import write_data_to_json
from crystal.helpers.temp import run_in_temp_dir
from crystal.structure.slabs.builder import (
    Slab,
    SlabFactory,
    Termination,
    TerminationLayer,
    filter_slabs_by_termination,
    passivate_slab,
)
from tests.helpers.constants.structure import (
    FLOATING_POINT_TOLERANCE,
    SLAB_CENTER_FRACTION,
    TEST_SI_STRUCTURE,
)
from tests.helpers.test_paths import (
    TEST_FILE_HELPERS_DATA_DIRNAME,
    TEST_LINBO3_STRUCTURE_FILENAME,
    TEST_SLAB_NO_MONTY_JSON_FILENAME,
    TEST_STRUCTURE_JSON_FILENAME,
)

LINBO3_STRUCTURE_FILENAME = (
    Path(__file__).parent.resolve()
    / TEST_FILE_HELPERS_DATA_DIRNAME
    / TEST_LINBO3_STRUCTURE_FILENAME
)

TEST_BCC_FE_STRUCTURE = PMG_Structure(
    lattice=Lattice.cubic(2.86),
    species=["Fe", "Fe"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
TEST_B2_FE_NI_STRUCTURE = PMG_Structure(
    lattice=Lattice.cubic(2.86),
    species=["Fe", "Ni"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

TEST_LINBO3_STRUCTURE = PMG_Structure.from_file(filename=LINBO3_STRUCTURE_FILENAME)
TEST_LINBO3_CHARGED_STRUCTURE = deepcopy(
    TEST_LINBO3_STRUCTURE
).add_oxidation_state_by_element(oxidation_states={"Li": 1, "Nb": 5, "O": -2})


EXAMPLE_SLAB_FACTORY = SlabFactory(
    structure=TEST_BCC_FE_STRUCTURE,
    supercell=(1, 1, 5),
    vacuum_thickness_in_layers=2,
    surface_miller_indices=[1, 0, 0],
)

TEST_SLAB = EXAMPLE_SLAB_FACTORY.generate_slabs()[0]
H_SPECIES = Species(symbol="H", oxidation_state=1)


def _make_tilted_c_vector_test_slab() -> Slab:
    lattice = Lattice([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [6.0, 0.0, 8.0]])
    frac_coords = np.array([[0.0, 0.0, 0.2], [0.8, 0.6, 0.8]])
    oriented_unit_cell = PMG_Structure(
        lattice=lattice, species=["Fe", "Fe"], coords=frac_coords
    )
    return Slab(
        lattice=lattice,
        species=["Fe", "Fe"],
        coords=frac_coords,
        miller_index=(0, 0, 1),
        oriented_unit_cell=oriented_unit_cell,
        shift=0.0,
        scale_factor=np.eye(3),
        reorient_lattice=False,
    )


@pytest.mark.parametrize(
    argnames=("structure", "expected_number_of_slabs", "expected_terminations"),
    argvalues=[
        (TEST_BCC_FE_STRUCTURE, 1, (["Fe_P4/mmm_1", "Fe_P4/mmm_1"],)),
        (TEST_B2_FE_NI_STRUCTURE, 1, (["Fe_P4/mmm_1", "Ni_P4/mmm_1"],)),
        (
            TEST_LINBO3_STRUCTURE,
            5,
            (
                ["Nb_C2/m_1", "Li_C2/m_1"],
                ["Li_C2/m_1", "LiNbO6_P1_8"],
                ["NbO2_P1_3", "LiO4_P1_5"],
                ["NbO4_P1_5", "LiO2_P1_3"],
                ["LiNbO6_P1_8", "Nb_C2/m_1"],
            ),
        ),
    ],
)
def test_slab_generation(
    structure: PMG_Structure,
    expected_number_of_slabs: int,
    expected_terminations: tuple[Termination, Termination],
) -> None:
    slab_factory = SlabFactory(
        structure=structure,
        supercell=(1, 1, 5),
        vacuum_thickness_in_layers=2,
        surface_miller_indices=[1, 0, 0],
    )
    assert slab_factory.number_of_layers == 5
    slabs = slab_factory.generate_slabs()
    assert len(slabs) == expected_number_of_slabs
    assert all(
        len(slab) == slab_factory.number_of_layers * len(structure) for slab in slabs
    )

    all_slab_factory_terminations = slab_factory.get_all_terminations()
    assert len(all_slab_factory_terminations) == len(expected_terminations)

    for index, [top_termination, bottom_termination] in enumerate(
        all_slab_factory_terminations
    ):
        assert top_termination == expected_terminations[index][0]
        assert bottom_termination == expected_terminations[index][1]


def test_slab_factory_reduces_slabs_to_primitive() -> None:
    conventional_silicon_structure = deepcopy(TEST_SI_STRUCTURE).to_conventional()
    slabs = []
    for reduce_slabs_to_primitive in (True, False):
        slab_factory = SlabFactory(
            structure=conventional_silicon_structure,
            supercell=(1, 1, 2),
            vacuum_thickness_in_layers=1,
            surface_miller_indices=(1, 0, 1),
            reduce_slabs_to_primitive=reduce_slabs_to_primitive,
        )
        slab = slab_factory.generate_slabs()[0]
        slabs.append(slab)

    [primitive_slab, unreduced_slab] = slabs
    assert len(primitive_slab) == 4
    assert len(unreduced_slab) == 12


def test_slab() -> None:
    top_termination = TEST_SLAB.top_layer_termination
    bottom_termination = TEST_SLAB.bottom_layer_termination
    assert top_termination == bottom_termination == "Fe_P4/mmm_1"


def test_surface_structures_filtered_by_terminations() -> None:
    slab_factory = SlabFactory(
        structure=TEST_LINBO3_STRUCTURE,
        supercell=(1, 1, 5),
        vacuum_thickness_in_layers=2,
        surface_miller_indices=[0, 0, 1],  # Z-cut surface
    )
    all_slabs = slab_factory.generate_slabs()
    all_terminations = slab_factory.get_all_terminations()
    assert isinstance(all_terminations, tuple)
    assert len(all_terminations) == 5

    # top slab terminating with only Li atoms
    check_top_termination = "Li_"

    # filter by specific termination
    [filtered_slab] = filter_slabs_by_termination(
        slabs=all_slabs,
        termination=check_top_termination,
        at_layer=TerminationLayer.top,
    )
    assert isinstance(filtered_slab, Slab)
    assert check_top_termination in filtered_slab.top_layer_termination
    assert check_top_termination not in filtered_slab.bottom_layer_termination

    assert not filtered_slab.is_orthogonal_in_plane
    orthogonal_slab = filtered_slab.orthogonalize_in_xy_plane
    assert orthogonal_slab.is_orthogonal_in_plane


@run_in_temp_dir
def test_slab_io() -> None:
    json_file = Path(TEST_STRUCTURE_JSON_FILENAME)
    assert not json_file.is_file()

    example_slab = deepcopy(TEST_SLAB)
    example_slab.write_to_json(filepath=json_file)

    assert json_file.is_file()

    slab_from_file = Slab.read_from_json(json_file)
    assert StructureMatcher().fit(slab_from_file, example_slab)


def test_charged_slab_generation_from_charged_structure() -> None:
    slab_factory = SlabFactory(
        structure=TEST_LINBO3_CHARGED_STRUCTURE,
        supercell=(1, 1, 5),
        vacuum_thickness_in_layers=2,
        surface_miller_indices=[0, 0, 1],  # Z-cut surface
    )
    [*_, test_slab] = slab_factory.generate_slabs()
    net_charge = sum(site.specie.oxi_state for site in test_slab)
    assert net_charge == 0
    assert all(site.specie.oxi_state != 0 for site in test_slab)


def test_identify_surface_sites() -> None:
    depth_from_surface = 0.2
    slab = deepcopy(TEST_SLAB)
    surface_sites = slab.get_terminal_sites(depth_from_surface=depth_from_surface)
    top_surface_sites = surface_sites[TerminationLayer.top]
    assert all(
        site.coords[2] > slab.z_max - depth_from_surface for site in top_surface_sites
    )

    bottom_surface_sites = surface_sites[TerminationLayer.bottom]
    assert all(
        site.coords[2] < slab.z_min + depth_from_surface
        for site in bottom_surface_sites
    )


def test_passivate_slab() -> None:
    slab = deepcopy(TEST_SLAB)
    passivated_slab = passivate_slab(
        initial_slab=slab,
        passivant=H_SPECIES,
        bond_length_map={"Fe": 1},
        layer=TerminationLayer.top,
    )
    assert len(slab) + 1 == len(passivated_slab)
    [h_site] = [site for site in passivated_slab.sites if site.specie.symbol == "H"]
    assert np.allclose(h_site.coords, [0, 0, 17.445])

    top_and_bottom_passivated_slab = passivate_slab(
        initial_slab=slab,
        passivant=H_SPECIES,
        bond_length_map={"Fe": 1},
        layer=TerminationLayer.both,
    )
    assert len(slab) + 2 == len(top_and_bottom_passivated_slab)
    h_sites = [
        site
        for site in top_and_bottom_passivated_slab.sites
        if site.specie.symbol == "H"
    ]
    sorted_h_sites = sorted(h_sites, key=lambda x: x.coords[2])
    [h_site_bottom, h_site_top] = sorted_h_sites
    assert np.allclose(
        h_site_bottom.coords, [1.43, 1.43, 2.575], atol=FLOATING_POINT_TOLERANCE
    )
    assert np.allclose(h_site_top.coords, [0, 0, 17.445], atol=FLOATING_POINT_TOLERANCE)


def test_error_in_passivation() -> None:
    slab = deepcopy(TEST_SLAB)
    with pytest.raises(
        UnsupportedTerminationLayerException, match="Unsupported termination layer"
    ):
        _ = passivate_slab(
            initial_slab=slab,
            passivant=H_SPECIES,
            bond_length_map={"Fe": 1},
            layer="random",  # type: ignore [arg-type]
        )

    with pytest.raises(
        ValueError, match="Missing bond length entries for surface elements"
    ):
        _ = passivate_slab(
            initial_slab=slab,
            passivant=H_SPECIES,
            bond_length_map={"Li": 1},
            layer=TerminationLayer.top,
        )


def test_passivate_linbo3_slab() -> None:
    slab_factory = SlabFactory(
        structure=TEST_LINBO3_CHARGED_STRUCTURE,
        supercell=(1, 1, 5),
        vacuum_thickness_in_layers=2,
        surface_miller_indices=[0, 0, 1],  # Z-cut surface
    )
    [*_, test_slab] = slab_factory.generate_slabs()
    bond_length_map = {"Li": 1.1, "Nb": 1.4, "O": 0.8}
    depth_from_surface = 0.2
    passivated_slab = passivate_slab(
        initial_slab=test_slab,
        passivant=H_SPECIES,
        bond_length_map=bond_length_map,
        layer=TerminationLayer.top,
        depth_from_surface=depth_from_surface,
    )

    passivant_h_sites = [
        site for site in passivated_slab.sites if site.specie.symbol == "H"
    ]

    number_of_h_sites_in_passivated_slab = len(passivant_h_sites)
    top_surface_sites_in_slab = test_slab.get_terminal_sites(
        depth_from_surface=depth_from_surface
    )[TerminationLayer.top]
    assert len(top_surface_sites_in_slab) == number_of_h_sites_in_passivated_slab


def test_minimum_distance() -> None:
    slab = deepcopy(TEST_SLAB)
    assert np.isclose(slab.minimum_neighbor_distance_in_angstrom, 2.4768, atol=1e-3)


def test_height_without_vacuum() -> None:
    slab = deepcopy(TEST_SLAB)
    assert np.isclose(slab.height_in_angstrom_without_vacuum, 12.87, atol=1e-2)
    assert np.isclose(slab.lattice.c, 20.02, atol=1e-2)


def test_vacuum_uses_cell_and_slab_extents_along_c() -> None:
    slab = _make_tilted_c_vector_test_slab()

    assert slab.lattice.c == pytest.approx(10.0)
    assert (
        slab.height_in_angstrom_without_vacuum + slab.vacuum_thickness_in_angstrom
        == pytest.approx(slab.lattice.c)
    )
    assert slab.vacuum_thickness_in_angstrom == pytest.approx(4.0)


def test_strip_vacuum_uses_slab_extent_along_c() -> None:
    slab = _make_tilted_c_vector_test_slab()

    slab_without_vacuum = slab.strip_vacuum()

    assert slab_without_vacuum.lattice.c == pytest.approx(6.0)
    assert slab_without_vacuum.vacuum_thickness_in_angstrom == pytest.approx(0.0)


def test_mirrored_slab() -> None:
    slab = deepcopy(TEST_SLAB)
    assert isinstance(slab.mirrored, Slab)
    assert isinstance(slab, Slab)
    assert np.isclose(slab.mirrored[-1].coords[2], slab[0].coords[2])
    assert np.isclose(slab.mirrored[0].coords[2], slab[-1].coords[2])


@run_in_temp_dir
def test_read_from_file_without_monty_serialized_dict_items() -> None:
    slab = deepcopy(TEST_SLAB)
    slab_as_dict = slab.as_dict()
    slab_as_dict_without_monty_serialized_objects = {
        k: v for k, v in slab_as_dict.items() if "@" not in k
    }

    temp_file = Path(TEST_SLAB_NO_MONTY_JSON_FILENAME)
    write_data_to_json(temp_file, data=slab_as_dict_without_monty_serialized_objects)

    slab_from_file = Slab.read_from_json(temp_file)
    assert isinstance(slab_from_file, Slab)


def test_centered_slab() -> None:
    structure = TEST_LINBO3_STRUCTURE.copy()
    slab_factory = SlabFactory(
        structure=structure,
        supercell=(1, 1, 1),
        vacuum_thickness_in_layers=10,
        surface_miller_indices=[0, 0, 1],
        orthogonalize_slabs=False,
    )
    [*_, test_slab] = slab_factory.generate_slabs()
    shifted_slab = test_slab.copy()

    zmin = min(site.coords[2] for site in test_slab.sites)
    shifted_slab.translate_sites(
        range(len(shifted_slab)), [0, 0, -zmin // 2], frac_coords=False
    )
    assert not np.isclose(
        shifted_slab.center_of_mass[2],
        SLAB_CENTER_FRACTION,
        atol=FLOATING_POINT_TOLERANCE,
    )
    assert np.isclose(
        shifted_slab.lattice.c, test_slab.lattice.c, atol=FLOATING_POINT_TOLERANCE
    )
    centered_slab_without_modified_vacuum = shifted_slab.get_centered_slab()
    assert np.isclose(
        centered_slab_without_modified_vacuum.center_of_mass[2],
        SLAB_CENTER_FRACTION,
        atol=FLOATING_POINT_TOLERANCE,
    )
    assert np.isclose(
        centered_slab_without_modified_vacuum.lattice.c,
        test_slab.lattice.c,
        atol=FLOATING_POINT_TOLERANCE,
    )

    centered_slab_with_modified_vacuum = shifted_slab.get_centered_slab(
        vacuum_in_angstrom=20
    )
    total_vacuum = (
        centered_slab_with_modified_vacuum.lattice.c
        - centered_slab_with_modified_vacuum.height_in_angstrom_without_vacuum
    )
    assert np.isclose(
        centered_slab_with_modified_vacuum.center_of_mass[2],
        SLAB_CENTER_FRACTION,
        atol=FLOATING_POINT_TOLERANCE,
    )
    assert np.isclose(total_vacuum, 2 * 20, atol=FLOATING_POINT_TOLERANCE)


def test_centered_slab_preserves_distances_without_vacuum() -> None:
    slab = deepcopy(TEST_SLAB)
    original_distance_matrix = slab.distance_matrix.copy()
    centered_slab = slab.get_centered_slab()
    assert np.allclose(
        centered_slab.distance_matrix,
        original_distance_matrix,
        atol=FLOATING_POINT_TOLERANCE,
    )
    assert np.isclose(
        centered_slab.minimum_neighbor_distance_in_angstrom,
        slab.minimum_neighbor_distance_in_angstrom,
        atol=FLOATING_POINT_TOLERANCE,
    )


def test_center_slab_in_place() -> None:
    slab = deepcopy(TEST_SLAB)
    slab.center_slab_in_place()
    assert np.isclose(
        slab.center_of_mass[2], SLAB_CENTER_FRACTION, atol=FLOATING_POINT_TOLERANCE
    )
