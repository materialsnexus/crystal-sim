from copy import deepcopy

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.structure.interfaces.builder import InterfaceFactory
from crystal.structure.slabs.builder import SlabFactory
from tests.helpers.test_data_paths import TESTS_DATA_DIR
from tests.helpers.test_paths import (
    TEST_LINBO3_STRUCTURE_FILENAME,
    TEST_QE_SCHEMA_QE_DIRNAME,
    TEST_RELAXED_CIFS_CELL_BOTTOM_FILENAME,
    TEST_RELAXED_CIFS_CELL_DIRNAME,
    TEST_RELAXED_CIFS_CELL_TOP_FILENAME,
    TEST_SAMPLE_SCF_CONFIG_FILENAME,
    TEST_SI_PSEUDOPOTENTIAL_FILENAME,
)

TEST_QE_DATA_DIR = TESTS_DATA_DIR / TEST_QE_SCHEMA_QE_DIRNAME
SAMPLE_QE_SCF_CONFIG_PATH = TEST_QE_DATA_DIR / TEST_SAMPLE_SCF_CONFIG_FILENAME
TEST_SI_PSEUDOPOTENTIAL_PATH = TESTS_DATA_DIR / TEST_SI_PSEUDOPOTENTIAL_FILENAME
RELAXED_CIFS_CELL_DIR = TESTS_DATA_DIR / TEST_RELAXED_CIFS_CELL_DIRNAME
RELAXED_CIFS_CELL_TOP_PATH = RELAXED_CIFS_CELL_DIR / TEST_RELAXED_CIFS_CELL_TOP_FILENAME
RELAXED_CIFS_CELL_BOTTOM_PATH = (
    RELAXED_CIFS_CELL_DIR / TEST_RELAXED_CIFS_CELL_BOTTOM_FILENAME
)
SLAB_ASSEMBLY_TEST_DIR = (
    TESTS_DATA_DIR / "slabs_for_assembly" / "slabs_for_interface_small"
)
SLAB_ASSEMBLY_TEST_TOP_PATH = (
    SLAB_ASSEMBLY_TEST_DIR / "surface_mp3731_cell_hkl_001_term_Li_Cmmm_1__O2_Cmmm_1.cif"
)
SLAB_ASSEMBLY_TEST_BOTTOM_PATH = (
    SLAB_ASSEMBLY_TEST_DIR
    / "surface_mp8352_cell_hkl_001_term_Si_P4_mmm_1__O2_Pmmm_2.cif"
)
DEFAULT_INTERFACE_MATCH_MAX_AREA = 600
SLAB_CENTER_FRACTION = 0.5
DEFAULT_INTERFACE_OFFSETS = [(0.0, 0.0)]
DEFAULT_INTERFACE_GAPS = [1.0]
DEFAULT_INTERFACE_VACUUM = 12.0
TEST_INTERFACE_JOBS_PER_BATCH = 2
TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM = 1.0
TEST_RELAX_ENERGY_LOW = 1.0
TEST_RELAX_ENERGY_HIGH = 2.0
TEST_INTERFACE_ENERGY_A = 4.5
TEST_INTERFACE_ENERGY_B = 5.0
TEST_INTERFACE_ENERGY_SINGLE = 4.0
TEST_INTERFACE_OFFSETS_CUSTOM = [(0.1, 0.2), (0.3, 0.4)]
TEST_INTERFACE_GAPS_CUSTOM = [2.0]
TEST_INTERFACE_GAP_MULTIPLIERS_CUSTOM = [1.1]
TEST_INTERFACE_VACUUM_CUSTOM = 14.0
TEST_RELAX_LATTICE_SCALE_UP = 1.01
TEST_RELAX_LATTICE_SCALE_DOWN = 0.99
TEST_RELAX_MAX_STEPS = 2
TEST_RELAX_STEP_INDEX = 0
TEST_RELAX_ZERO_FORCE_VECTOR = [0.0, 0.0, 0.0]
TEST_RELAX_STRESS_VECTOR_ZERO = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TEST_INTERFACE_ASSEMBLER_TOP_LATTICE_KEY = "top_lattice"
TEST_INTERFACE_ASSEMBLER_BOTTOM_LATTICE_KEY = "bottom_lattice"
TEST_RELAX_ENERGY_PLACEHOLDER = 0.0
TEST_INTERFACE_RELAX_EXTERNAL_FACES_FREEZE_THICKNESS_IN_ANGSTROM = 2.0
TEST_INVALID_INTERFACE_RELAX_EXTERNAL_FACES_FREEZE_THICKNESS_IN_ANGSTROM = 0.0
TEST_MANUAL_FIXED_ATOM_INDEX = 1
TEST_INTERFACE_COUNT = 5
TEST_INTERFACE_BATCH_SIZES = [2, 2, 1]
TEST_SINGLE_RESULT_COUNT = 1
TEST_SLAB_TASK_DOC_COUNT = 2
TEST_INTERFACE_TASK_DOC_COUNT_SINGLE = 1
TEST_INTERFACE_TASK_DOC_COUNT_MULTI = 2
TEST_ADHESION_RECORD_COUNT_SINGLE = 1
TEST_SLAB_COUNT = 2
TEST_FIRST_INDEX = 0
TEST_C_AXIS_INDEX = 2
MOCK_TASK_DIR_NAME = "mock"
MOCK_ERROR_LOG_MESSAGE = "mock error"
TEST_SLAB_MILLER_INDICES = (0, 0, 1)
TEST_SLAB_SCALE_FACTOR = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
TEST_SLAB_SHIFT = 0.0
TEST_SLAB_REORIENT_LATTICE = False


TEST_BCC_FE_STRUCTURE = PMG_Structure(
    lattice=Lattice.cubic(2.86),
    species=["Fe", "Fe"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

LINBO3_STRUCTURE_FILENAME = TESTS_DATA_DIR / TEST_LINBO3_STRUCTURE_FILENAME
TEST_LINBO3_STRUCTURE = PMG_Structure.from_file(filename=LINBO3_STRUCTURE_FILENAME)
TEST_LINBO3_SLAB = (
    SlabFactory(
        structure=TEST_LINBO3_STRUCTURE,
        supercell=(1, 1, 1),
        vacuum_thickness_in_layers=2,
        surface_miller_indices=[0, 0, 1],
    )
    .generate_slabs()[0]
    .orthogonalize_in_xy_plane
)

TEST_001_FE_SLAB = (
    SlabFactory(
        structure=TEST_BCC_FE_STRUCTURE,
        supercell=(1, 1, 1),
        vacuum_thickness_in_layers=2,
        surface_miller_indices=[0, 0, 1],
    )
    .generate_slabs()[0]
    .orthogonalize_in_xy_plane
)

TEST_111_FE_SLAB = (
    SlabFactory(
        structure=TEST_BCC_FE_STRUCTURE,
        supercell=(1, 1, 1),
        vacuum_thickness_in_layers=5,
        surface_miller_indices=[1, 1, 1],
    )
    .generate_slabs()[0]
    .orthogonalize_in_xy_plane
)

TEST_SI_STRUCTURE = PMG_Structure(
    lattice=Lattice.cubic(5.431),
    species=["Si", "Si"],
    coords=np.array([[0, 0, 0], [0.25, 0.25, 0.25]]),
    coords_are_cartesian=False,
)
FLOATING_POINT_TOLERANCE = 1e-3
MOCK_ERROR_MESSAGE = "This is a mock error message"

TEST_INTERFACE_FACTORY = InterfaceFactory(
    top_slab=deepcopy(TEST_001_FE_SLAB), bottom_slab=deepcopy(TEST_001_FE_SLAB)
)

[TEST_INTERFACE] = list(deepcopy(TEST_INTERFACE_FACTORY).generate_interfaces())
TEST_TOP_SLAB = deepcopy(TEST_INTERFACE).get_centered_top_slab(vacuum_in_angstrom=10.0)
TEST_BOTTOM_SLAB = deepcopy(TEST_INTERFACE).get_centered_bottom_slab(
    vacuum_in_angstrom=10.0
)
