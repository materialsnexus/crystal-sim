from crystal import CRYSTAL_CONFIG
from crystal.schemas.qe import SerializableEspressoProfile

QE_NOT_CONFIGURED_MESSAGE = "Quantum Espresso is not configured in CRYSTAL_CONFIG."
QE_MISSING_PSEUDOS_MESSAGE = (
    "Pseudopotentials are missing for elements in the slab inputs."
)
DEGAUSS_FIELD_NAME = "degauss"
CONV_THR_FIELD_NAME = "conv_thr"
CUTOFF_SERIALIZATION_TOLERANCE = 1e-9
DEFAULT_DISK_IO = "nowf"
TEST_ESPRESSO_PROFILE = SerializableEspressoProfile(  # type: ignore[no-untyped-call]
    command="", pseudo_dir=""
)
DEFAULT_QE_PRE_ROCKET = f"source {CRYSTAL_CONFIG.programs.setvars}"
