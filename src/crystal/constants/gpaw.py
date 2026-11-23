"""GPAW constants."""

from enum import StrEnum, auto
from pathlib import Path

from crystal.constants.chargemol import (
    CHARGEMOL_BOND_ORDERS_FILENAME,
    CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME,
    CHARGEMOL_OVERLAP_POPULATIONS_FILENAME,
    CHARGEMOL_TOTAL_CUBE_ANALYSIS_OUTPUT_FILENAME,
)

DEFAULT_GPAW_XC = "PBE"
DEFAULT_GPAW_SPINPOL = False
DEFAULT_GPAW_CUTOFF_EV = 850.0
DEFAULT_GPAW_KPOINT_DENSITY = 6.0
DEFAULT_GPAW_GAMMA_CENTERED = True
DEFAULT_GPAW_MAXITER = 2000
DEFAULT_GPAW_ISOLATED_ATOM_VACUUM_ANGSTROM = 10.0
DEFAULT_GPAW_ISOLATED_ATOM_NUMBER_OF_BANDS = -8
GPAW_ISOLATED_ATOM_MIXER_BACKEND = "no-mixing"
GPAW_ISOLATED_ATOM_EIGENSOLVER_NAME = "etdm-fdpw"
DEFAULT_GPAW_FERMI_DIRAC_WIDTH = 0.005

DEFAULT_GPAW_CONVERGENCE_DENSITY = 1e-4
DEFAULT_GPAW_CONVERGENCE_ENERGY = 5e-4
DEFAULT_GPAW_CONVERGENCE_FORCES = 0.01
DEFAULT_GPAW_CONVERGENCE_EIGENSTATES = 1e-5

DEFAULT_GPAW_PARALLEL_SL_AUTO = True
DEFAULT_GPAW_PARALLEL_USE_ELPA = True

DEFAULT_GPAW_SLAB_INTERFACE_PBC = (True, True, False)
DEFAULT_GPAW_SLAB_INTERFACE_DIPOLELAYER = "xy"
GPAW_PBC_VECTOR_COUNT = 3

GPAW_SCF_INPUT_STRUCTURE_FILENAME = "gpaw_input.extxyz"
GPAW_SCF_KWARGS_FILENAME = "gpaw_kwargs.json"
GPAW_SCF_OUTPUT_FILENAME = "gpaw_output.json"
GPAW_TEXT_LOG_FILENAME = "gpaw.txt"
GPAW_TOTAL_DENSITY_CUBE_FILENAME = "total_density.cube"
GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES = (
    CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME,
    CHARGEMOL_BOND_ORDERS_FILENAME,
    CHARGEMOL_OVERLAP_POPULATIONS_FILENAME,
    CHARGEMOL_TOTAL_CUBE_ANALYSIS_OUTPUT_FILENAME,
)
DEFAULT_GPAW_CHARGEMOL_DENSITY_GRIDREFINEMENT = 4


class GpawExternalScfScriptInputKey(StrEnum):
    """Keys used by the external GPAW SCF script input payload."""

    calculator_kwargs = auto()
    density_gridrefinement = auto()
    total_density_cube_filename = auto()


GPAW_EXTERNAL_SCF_SCRIPT_FILENAME = "gpaw_external_scf_script.py"
GPAW_RUNTIME_PROVENANCE_FILENAME = "gpaw_runtime_provenance.py"
GPAW_EXTERNAL_RELAX_SCRIPT_FILENAME = "gpaw_external_relax_script.py"
GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_FILENAME = "gpaw_external_isolated_atom_script.py"
GPAW_ISOLATED_ATOM_INPUT_STRUCTURE_FILENAME = "gpaw_isolated_atom_input.extxyz"
GPAW_ISOLATED_ATOM_KWARGS_FILENAME = "gpaw_isolated_atom_kwargs.json"
GPAW_ISOLATED_ATOM_OUTPUT_FILENAME = "gpaw_isolated_atom_output.json"
GPAW_ISOLATED_ATOM_REGISTRY_STORE_NAME = "gpaw_isolated_atom_registry"


class GpawExternalRelaxScriptInputKey(StrEnum):
    """Keys used by the external GPAW relax script input payload."""

    calculator_kwargs = auto()
    optimizer_type = auto()
    max_steps = auto()
    force_threshold_in_ev_per_angstrom = auto()
    store_interval = auto()
    fix_cell = auto()
    fixed_atom_indices = auto()


class GpawExternalRelaxScriptOutputKey(StrEnum):
    """Top-level keys in the external GPAW relax script output payload."""

    relax_steps = auto()


class GpawExternalRelaxScriptStepKey(StrEnum):
    """Per-step keys in the external GPAW relax script output payload."""

    step = auto()
    symbols = auto()
    positions = auto()
    cell = auto()
    pbc = auto()
    forces = auto()
    stress = auto()
    energy = auto()


GPAW_MPI_LAUNCHER = "mpirun"
GPAW_MPI_PROCESS_COUNT_FLAG = "-np"
GPAW_LAUNCH_PYTHON_SUBCOMMAND = "python"
GPAW_SETUP_PATH_ENVIRONMENT_VARIABLE = "GPAW_SETUP_PATH"

GPAW_ENV_PYTHON_RELATIVE_PATH = Path("bin") / "python"
GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH = Path("bin") / "gpaw"
EXTERNAL_GPAW_CONFIG_NOT_FOUND_MESSAGE = (
    "External GPAW runtime configuration not found. "
    "Set programs.gpaw_env_dir in crystal.toml."
)
EXTERNAL_GPAW_SETUPS_CONFIG_NOT_FOUND_MESSAGE = (
    "External GPAW setups configuration not found. "
    "Set programs.gpaw_setups_dir in crystal.toml."
)
EXTERNAL_GPAW_ENV_DIR_NOT_FOUND_MESSAGE_TEMPLATE = (
    "Configured external GPAW environment directory does not exist: {gpaw_env_root}"
)
EXTERNAL_GPAW_SETUPS_DIR_NOT_FOUND_MESSAGE_TEMPLATE = (
    "Configured external GPAW setups directory does not exist: {gpaw_setups_dir}"
)
EXTERNAL_GPAW_PYTHON_NOT_FOUND_MESSAGE_TEMPLATE = (
    "Configured external GPAW python executable does not exist: "
    "{python_executable_path}"
)
EXTERNAL_GPAW_EXECUTABLE_NOT_FOUND_MESSAGE_TEMPLATE = (
    "Configured external GPAW gpaw executable does not exist: {gpaw_executable_path}"
)
EXTERNAL_GPAW_IMPORT_CHECK_FAILED_MESSAGE_TEMPLATE = (
    "Configured external GPAW python failed to import gpaw: {python_executable_path}"
)
MIN_SUPPORTED_GPAW_VERSION = "26.7.0"
EXTERNAL_GPAW_VERSION_TOO_OLD_MESSAGE_TEMPLATE = (
    "Configured external GPAW version {installed_version} is below minimum "
    "supported {minimum_supported_version}: {python_executable_path}"
)
GPAW_BULK_UNSUPPORTED_SLAB_ONLY_FIELDS_MESSAGE = (
    "poissonsolver and pbc are supported only when bulk_or_slab='slab_interface'."
)
GPAW_SLAB_INTERFACE_PBC_REQUIRED_MESSAGE = (
    "pbc is required when bulk_or_slab='slab_interface'."
)
GPAW_SLAB_INTERFACE_POISSONSOLVER_REQUIRED_MESSAGE = (
    "poissonsolver is required when bulk_or_slab='slab_interface'."
)
GPAW_SLAB_INTERFACE_PBC_INVALID_MESSAGE = (
    "pbc must be (True, True, False) when bulk_or_slab='slab_interface'."
)
GPAW_EXTERNAL_SCF_RUN_FAILED_MESSAGE = "External GPAW SCF run failed."
GPAW_EXTERNAL_RELAX_RUN_FAILED_MESSAGE = "External GPAW relax run failed."
GPAW_EXTERNAL_ISOLATED_ATOM_RUN_FAILED_MESSAGE = (
    "External GPAW isolated atom run failed."
)
GPAW_INVALID_PBC_FLAGS_IN_OUTPUT_MESSAGE_TEMPLATE = (
    "Invalid PBC flags in GPAW output payload: {pbc}"
)
GPAW_TASK_DOCUMENT_OUTPUT_ERROR_MISMATCH_MESSAGE = (
    "Exactly one of properties and error log must exist"
)
GPAW_RELAX_OUTPUT_STEP_POSITIONS_COUNT_MISMATCH_MESSAGE_TEMPLATE = (
    "Invalid GPAW relax output step: positions count ({positions_count}) "
    "does not match symbols count ({symbols_count})."
)
GPAW_RELAX_OUTPUT_STEP_FORCES_COUNT_MISMATCH_MESSAGE_TEMPLATE = (
    "Invalid GPAW relax output step: forces count ({forces_count}) "
    "does not match symbols count ({symbols_count})."
)
GPAW_CHARGEMOL_TASK_DOCUMENT_RUN_INFO_REQUIRED_MESSAGE = (
    "Chargemol run info must exist when there are no errors."
)
GPAW_CHARGEMOL_TASK_DOCUMENT_RUN_INFO_FORBIDDEN_MESSAGE = (
    "Chargemol run info must be empty when errors are present."
)
