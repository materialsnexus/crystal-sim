"""Pydantic models for Quantum ESPRESSO (QE) SCF configuration.

Designed to work with JSON configs that users create,
We implement settings we will use during CRYSTAL not a general calculator.
"""

import logging
from enum import IntEnum, StrEnum, auto
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from ase.calculators.espresso import EspressoProfile
from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pymatgen.core.structure import Structure as PMG_Structure

from crystal import CRYSTAL_CONFIG
from crystal.constants.qe import (
    DEFAULT_CHARGE_DENSITY_CUTOFF_ENERGY,
    DEFAULT_CONV_THR_ENERGY,
    DEFAULT_DEGAUSS_ENERGY,
    DEFAULT_QE_OUTPUT_FILE_NAMES,
    DEFAULT_WAVEFUNCTION_CUTOFF_ENERGY,
    QE_MPIRUN_COMMAND_TEMPLATE,
    QE_PW_EXECUTABLE_NAME,
    RYDBERG_TO_EV,
)
from crystal.helpers.dict import count_existing_keys, deep_merge
from crystal.helpers.file import read_file_lines
from crystal.helpers.json import read_json_to_dict
from crystal.helpers.parse import parse_text_in_between_headers
from crystal.schemas.backend import RelaxBackend
from crystal.schemas.base import BaseTaskDocument
from crystal.schemas.elements import ElementStr
from crystal.schemas.process import ProcessRunInfo
from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from crystal.schemas.properties_union import PropertiesUnion
from crystal.schemas.pseudopotential import (
    PseudopotentialConfig,
    PseudopotentialFileName,
)
from crystal.schemas.relax_settings import RelaxOptimizerSettings
from crystal.schemas.relaxation_constraints import RelaxationConstraints
from crystal.schemas.serialization import MSONABLE_JSON_MODE, MSONableBaseModel
from crystal.schemas.settings import MakerSettings, SimulationSettings

logger = logging.getLogger(__name__)

QE_JOB_COMPLETE = "JOB DONE"
QE_WARNING_TEXT = "WARNING"


class Occupations(StrEnum):
    """Electronic occupations type (restricted to smearing for QE usage)."""

    smearing = auto()


class Smearing(StrEnum):
    """Smearing type for metals."""

    gaussian = auto()
    mp = auto()
    mv = auto()
    cold = auto()
    fd = auto()


class Diagonalization(StrEnum):
    """Diagonalization algorithm."""

    david = auto()
    cg = auto()


class MixingMode(StrEnum):
    """Charge mixing mode."""

    plain = auto()
    tf = "TF"
    local_tf = "local-TF"


class IsolatedSystemPolicy(StrEnum):
    """Treatment of isolated systems
    (See https://www.quantum-espresso.org/Doc/INPUT_PW.html#id112 ).
    """

    none = auto()
    esm = auto()
    two_dimensions = "2D"
    mt = auto()


class EsmBoundaryCondition(StrEnum):
    """ESM boundary conditions."""

    bc1 = auto()  # Vacuum-slab-vacuum


class SpinPolarizationPolicy(IntEnum):
    """Spin polarization policy for QE calculations."""

    non_polarized = 1
    collinear = 2


class ScfCalculationType(StrEnum):
    """Supported calculation via ASE."""

    # NOTE: Can include scf constrained relax as another option
    scf = auto()


class DiskIO(StrEnum):
    """QE disk I/O policy."""

    high = auto()
    medium = auto()
    low = auto()
    nowf = auto()
    minimal = auto()
    none = auto()


class StartingWavefunction(StrEnum):
    """Starting wavefunction source for SCF."""

    atomic = auto()
    atomic_random = "atomic+random"
    random = auto()
    file = auto()


class StartingPotential(StrEnum):
    """Starting potential source for SCF."""

    atomic = auto()
    file = auto()


class QeSystemKeyword(StrEnum):
    """QE `SYSTEM` namelist keys."""

    wavefunction_cutoff = "ecutwfc"
    charge_density_cutoff = "ecutrho"


class ScfControlParameters(MSONableBaseModel):
    """Control parameters for SCF calculation."""

    calculation: ScfCalculationType = Field(
        default=ScfCalculationType.scf, description="QE calculation type (scf)"
    )
    tprnfor: bool = Field(
        default=True, description="Calculate forces (required for ASE optimization)"
    )
    tstress: bool = Field(
        default=True, description="Calculate stress tensor (required for MLIP training)"
    )
    disk_io: DiskIO = Field(default=DiskIO.nowf, description="QE disk I/O policy")


class ScfSystemParameters(MSONableBaseModel):
    """System parameters for SCF calculation."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    occupations: Occupations = Field(
        default=Occupations.smearing, description="Occupation type"
    )
    smearing: Smearing | None = Field(
        default=None, description="Smearing type (default: cold for metals)"
    )
    degauss: Annotated[float, Field(gt=0)] = Field(
        default=DEFAULT_DEGAUSS_ENERGY, description="Smearing width (eV)"
    )
    wavefunction_cutoff_energy: Annotated[float, Field(gt=0)] = Field(
        default=DEFAULT_WAVEFUNCTION_CUTOFF_ENERGY,
        description="Plane-wave kinetic energy cutoff for wavefunctions (eV)",
    )
    charge_density_cutoff_energy: Annotated[float, Field(gt=0)] = Field(
        default=DEFAULT_CHARGE_DENSITY_CUTOFF_ENERGY,
        description="Plane-wave kinetic energy cutoff for charge density (eV)",
    )

    assume_isolated: IsolatedSystemPolicy | None = Field(
        default=None, description="Isolated system treatment"
    )
    esm_bc: EsmBoundaryCondition | None = Field(
        default=None, description="ESM boundary condition"
    )
    spin_polarization_policy: SpinPolarizationPolicy = Field(
        default=SpinPolarizationPolicy.non_polarized,
        description=(
            "Spin polarization policy (non_polarized maps to nspin=1, "
            "collinear maps to nspin=2)"
        ),
        serialization_alias="nspin",
    )

    @field_validator("smearing")
    @classmethod
    def set_default_smearing(
        cls, variable: Smearing | None, info: ValidationInfo
    ) -> Smearing | None:
        """Normalize incoming smearing values."""
        if variable is not None:
            return variable
        if info.data.get("occupations") == Occupations.smearing:
            return Smearing.mv
        return variable

    @model_validator(mode="after")
    def ensure_smearing_when_needed(self) -> Self:
        """Guarantee smearing is defined for metallic occupations."""
        if self.occupations == Occupations.smearing and self.smearing is None:
            self.smearing = Smearing.mv
        return self

    def build_qe_system_namelist_with_cutoffs_in_rydberg(self) -> dict[str, Any]:
        """Build QE `SYSTEM` namelist payload with explicit eV -> Ry cutoffs."""
        system_dict = self.model_dump(mode=MSONABLE_JSON_MODE, exclude_none=True)
        system_dict[QeSystemKeyword.wavefunction_cutoff.value] = (
            self.wavefunction_cutoff_energy / RYDBERG_TO_EV
        )
        system_dict[QeSystemKeyword.charge_density_cutoff.value] = (
            self.charge_density_cutoff_energy / RYDBERG_TO_EV
        )
        system_dict["degauss"] = self.degauss / RYDBERG_TO_EV
        system_dict.pop("wavefunction_cutoff_energy", None)
        system_dict.pop("charge_density_cutoff_energy", None)
        return system_dict

    @model_validator(mode="after")
    def validate_esm_consistency(self) -> Self:
        """Ensure ESM boundary conditions require assume_isolated='esm'."""
        if (
            self.esm_bc is not None and self.assume_isolated != IsolatedSystemPolicy.esm
        ) or (self.assume_isolated == IsolatedSystemPolicy.esm and self.esm_bc is None):
            msg = (
                f"When using esm_bc ('{self.esm_bc}'), "
                f"assume_isolated must be set to 'esm' ( and vice versa). "
                f"Currently: assume_isolated='{self.assume_isolated}'"
            )
            raise ValueError(msg)
        return self


class ScfElectronsParameters(MSONableBaseModel):
    """Electron SCF parameters."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    conv_thr_energy: Annotated[float, Field(gt=0)] = Field(
        default=DEFAULT_CONV_THR_ENERGY,
        description="Electronic SCF convergence threshold (eV)",
    )
    mixing_beta: Annotated[float, Field(gt=0, le=1)] = Field(
        default=0.2, description="Mixing factor for SCF"
    )
    mixing_mode: MixingMode = Field(default=MixingMode.local_tf)
    mixing_ndim: Annotated[int, Field(gt=0)] = Field(
        default=12, description="Number of previous iterations used in mixing"
    )
    startingwfc: StartingWavefunction = Field(
        default=StartingWavefunction.atomic, description="Starting wavefunction source"
    )
    startingpot: StartingPotential = Field(
        default=StartingPotential.atomic, description="Starting potential source"
    )
    diagonalization: Diagonalization = Field(default=Diagonalization.david)
    electron_maxstep: Annotated[int, Field(gt=0)] = Field(
        default=800, description="Maximum SCF iterations"
    )

    def build_qe_electrons_namelist_in_rydberg(self) -> dict[str, Any]:
        """Build QE `ELECTRONS` namelist payload with explicit eV -> Ry conversion."""
        electrons_dict = self.model_dump(mode=MSONABLE_JSON_MODE, exclude_none=True)
        electrons_dict["conv_thr"] = self.conv_thr_energy / RYDBERG_TO_EV
        electrons_dict.pop("conv_thr_energy", None)
        return electrons_dict


class QeScfInputData(MSONableBaseModel):
    """Container for Quantum ESPRESSO (QE) namelists (SCF only)."""

    control: ScfControlParameters = Field(default_factory=ScfControlParameters)
    system: ScfSystemParameters = Field(default_factory=ScfSystemParameters)
    electrons: ScfElectronsParameters = Field(default_factory=ScfElectronsParameters)

    def build_qe_input_data_namelists_with_cutoffs_in_rydberg(self) -> dict[str, Any]:
        """Build QE input_data payload with explicit unit conversion."""
        input_data = self.model_dump(mode=MSONABLE_JSON_MODE, exclude_none=True)
        input_data["system"] = (
            self.system.build_qe_system_namelist_with_cutoffs_in_rydberg()
        )
        input_data["electrons"] = (
            self.electrons.build_qe_electrons_namelist_in_rydberg()
        )
        return input_data


class KPoints(MSONableBaseModel):
    """K-points configuration."""

    # Either spacing or mesh
    spacing_in_inverse_angstrom: Annotated[float, Field(gt=0)] | None = Field(
        default=None, description="K-point spacing (1/Angstrom)"
    )
    mesh: tuple[int, int, int] | None = Field(
        default=None, description="Explicit k-point mesh"
    )
    offset: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="K-point offset"
    )

    @model_validator(mode="after")
    def _validate_kpoints(self) -> Self:
        """Ensure exactly one k-point specification."""
        has_spacing = self.spacing_in_inverse_angstrom is not None
        has_mesh = self.mesh is not None
        if not has_spacing and not has_mesh:
            msg = "Must specify either 'spacing' or 'mesh'"
            raise ValueError(msg)
        if has_spacing and has_mesh:
            msg = "Cannot specify both 'spacing' and 'mesh'"
            raise ValueError(msg)
        return self


class SerializableEspressoProfile(EspressoProfile):  # noqa: D101
    def as_dict(self) -> dict[str, Any]:  # noqa: D102
        return self.__dict__

    @staticmethod
    def _command(number_of_processes: int) -> str:
        if CRYSTAL_CONFIG.programs.quantum_espresso_dir is None:
            err_msg = (
                "Provide the bin folder path for quantum espresso in toml file as "
                "`quantum_espresso_directory=/path/to/bin` in [programs] section"
            )
            raise RuntimeError(err_msg)

        executable = (
            CRYSTAL_CONFIG.programs.quantum_espresso_dir / QE_PW_EXECUTABLE_NAME
        )
        return QE_MPIRUN_COMMAND_TEMPLATE.format(
            number_of_processes=number_of_processes, executable=executable.as_posix()
        )

    @classmethod
    def from_number_of_processes_and_pseudo_dir(  # noqa: D102
        cls, number_of_processes: int, pseudopotential_dir: Path
    ) -> "SerializableEspressoProfile":
        command = cls._command(number_of_processes=number_of_processes)
        return cls(command=command, pseudo_dir=pseudopotential_dir.as_posix())


_MUTUALLY_EXCLUSIVE_KPOINT_ARGS: tuple[str, ...] = (
    "spacing_in_inverse_angstrom",
    "mesh",
)
DEFAULT_KPOINTS = KPoints(spacing_in_inverse_angstrom=0.25)


class QeConfig(MSONableBaseModel):
    """QE configuration: input/k-point data from nested dicts or files."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_data: QeScfInputData = Field(default_factory=QeScfInputData)
    kpoints: KPoints = Field(default=DEFAULT_KPOINTS)

    def get_modified_instance(self, overrides: dict[str, Any]) -> Self:
        """Return a new, validated instance with the values of one or more parameters changed."""
        obj_dict = self.model_dump()
        updated_obj_dict = deep_merge(obj_dict, overrides)
        return type(self).model_validate(updated_obj_dict)

    @classmethod
    def from_dict(  # noqa: D102
        cls, config_dict: dict[str, Any]
    ) -> "QeConfig":
        input_payload = config_dict.get("input_data")
        input_data = (
            QeScfInputData.model_validate(input_payload)
            if input_payload is not None
            else QeScfInputData()
        )

        kpoints_payload = cls._kpoint_config(config_dict)
        if kpoints_payload is None:
            msg = "kpoints configuration is required"
            raise ValueError(msg)
        kpoints = KPoints.model_validate(kpoints_payload)

        return cls(input_data=input_data, kpoints=kpoints)

    @classmethod
    def from_file(  # noqa: D102
        cls, config_path: Path | str | None = None
    ) -> "QeConfig":
        config = cls._load_json(config_path) if config_path else {}
        return cls.from_dict(config_dict=config)

    @staticmethod
    def _load_json(path: Path | str) -> dict[str, Any]:
        data = read_json_to_dict(Path(path).expanduser().resolve())
        if not isinstance(data, dict):
            msg = (
                "Configuration files must define a JSON object at the root. "
                f"Encountered {type(data).__name__} when reading '{path}'."
            )
            raise TypeError(msg)
        return data

    @staticmethod
    def _kpoint_config(config: dict[str, Any]) -> dict[str, Any] | None:
        kpoints_dict = config.get("kpoints", {})
        if not kpoints_dict:
            return None
        assert isinstance(kpoints_dict, dict)
        count = count_existing_keys(kpoints_dict, _MUTUALLY_EXCLUSIVE_KPOINT_ARGS)
        if count != 1:
            raise ValueError
        return kpoints_dict


class QeScfSimulationSettings(SimulationSettings):  # noqa: D101
    pseudopotentials: dict[ElementStr, PseudopotentialFileName]
    config: QeConfig = Field(default_factory=QeConfig)
    set_pseudopotential_directory: Path | None = Field(default=None)

    @property
    def pseudopotential_directory(self) -> Path:  # noqa: D102
        valid_dir = (
            self.set_pseudopotential_directory
            or CRYSTAL_CONFIG.programs.pseudopotentials_dir
        )
        if valid_dir:
            return valid_dir
        err_msg = "No Pseudopotential directory provided."
        raise TypeError(err_msg)

    @property
    def pseudopotential_config(self) -> PseudopotentialConfig:  # noqa: D102
        return PseudopotentialConfig(
            pseudopotential_dir=self.pseudopotential_directory,
            pseudopotentials_file_map=self.pseudopotentials,
        )


class QeRelaxSimulationSettings(QeScfSimulationSettings, RelaxOptimizerSettings):  # noqa: D101
    constraints: RelaxationConstraints | None = None


class QeRelaxMakerSettings(MakerSettings):  # noqa: D101
    backend: Literal[RelaxBackend.qe] = RelaxBackend.qe
    simulation_settings: QeRelaxSimulationSettings


class QeTaskDocument(BaseTaskDocument):  # noqa: D101
    dir_name: str = Field(description="The directory for this task")

    @property
    def converged(self) -> bool:  # noqa: D102
        return not bool(self.errors)

    @classmethod
    def from_directory_and_properties(  # noqa: D102
        cls, dir_path: Path, properties: PropertiesUnion | None
    ) -> Self:
        output_files = [
            dir_path / file_name for file_name in DEFAULT_QE_OUTPUT_FILE_NAMES
        ]
        errors = extract_error_messages(output_files=output_files)
        return cls(
            dir_name=dir_path.as_posix(), errors=",".join(errors), properties=properties
        )


class QeScfPpChargemolTaskDocument(BaseTaskDocument):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    properties: SinglePointProperties | None = Field(
        description="Computed SCF properties from QE"
    )
    structure: PMG_Structure | None = Field(
        description="Structure with DDEC6 charges in site_properties"
    )
    pp_run_info: ProcessRunInfo | None = Field(
        description="Process metadata for pp.x charge density run"
    )
    chargemol_run_info: ProcessRunInfo | None = Field(
        description="Process metadata for Chargemol run"
    )

    @model_validator(mode="after")
    def _validate_errors_and_outputs(self) -> Self:
        if self.is_success and (
            self.properties is None
            or self.structure is None
            or self.pp_run_info is None
            or self.chargemol_run_info is None
        ):
            error_msg = (
                "Output, structure, and run info must exist when there are no errors."
            )
            raise ValueError(error_msg)
        if not self.is_success and (
            self.properties is not None
            or self.structure is not None
            or self.pp_run_info is not None
            or self.chargemol_run_info is not None
        ):
            error_msg = "Outputs and run info must be empty when errors are present."
            raise ValueError(error_msg)
        return self

    @property
    def is_success(self) -> bool:  # noqa: D102
        return self.errors == ""

    @property
    def converged(self) -> bool:  # noqa: D102
        return self.is_success


class QeRelaxTaskDocument(QeTaskDocument):  # noqa: D101
    properties: RelaxProperties | None = Field(
        description="Computed relax properties from a QE task"
    )


class QeErrorDelimiters(StrEnum):  # noqa: D101
    percent = "%%%"
    dash = "--------------------------------"
    routine_error = "Error in routine"


class QeErrorMessages(StrEnum):  # noqa: D101
    not_orthogonal_operation = "not orthogonal operation"
    de0_is_positive = "dE0s is positive which should never happen"
    lone_vector = "lone vector"
    non_zero_exit_code = "non-zero exit code"
    mpi_abort = "errorcode 1"
    bus_error = "Program received signal SIGBUS"
    segmentation_fault = "Program received signal SIGSEGV"
    cutoff_energy_not_set = "ecutwfc not set"
    use_pencil_decomposition = "Use pencil decomposition"
    restart_with_large_cell_factor = "try restarting with a larger cell_factor"
    unconverged = "convergence NOT achieved"
    s_matrix_not_positive_definite = "S matrix not positive definite"
    cannot_bracket_ef = "cannot bracket Ef"
    bands_not_converged = "too many bands are not converged"
    fully_relatistiv_pps_need_spin_orbit_coupling = (
        "Fully relativistic PPs, need spin-orbit calc. (lspinorb=.true.)"
    )


def extract_errors_between_delimiters(
    error_file_lines: list[str],
) -> list[QeErrorMessages]:
    """Extract error messages between known QE error delimiters."""
    runtime_errors: list[QeErrorMessages] = []
    for delimiter in QeErrorDelimiters:
        # this will return the text between the first occurrence of a gap between delimiters ONLY
        all_error_lines = parse_text_in_between_headers(
            section=error_file_lines, start_header=delimiter, end_header=delimiter
        )
        # filter our warnings and job complete message
        error_lines = []
        for line in all_error_lines:
            if any(segment in line for segment in (QE_WARNING_TEXT, QE_JOB_COMPLETE)):
                continue
            error_lines.append(line)

        # concatenate to a single str
        error_text = "".join(line.strip() for line in error_lines)
        runtime_errors += [
            error_msg for error_msg in QeErrorMessages if error_msg in error_text
        ]

        unknown_error_detected = error_lines and not runtime_errors
        if unknown_error_detected:
            msg = f"Encountered new error: {error_text}"
            raise RuntimeError(msg)
    return runtime_errors


def extract_error_messages(output_files: list[Path]) -> list[str]:
    """Returns unique error messages from all output files.

    If an error is present which is not in `QeErrorMessages`, the function
    will throw a RuntimeError with the new error message.
    """
    existing_output_files = [f for f in output_files if f.is_file()]

    error_messages: list[str] = []
    for output_file in existing_output_files:
        try:
            # populate error messages with the known ones
            lines = read_file_lines(output_file)
            for line in lines:
                error_messages.extend(
                    error for error in QeErrorMessages if error in line
                )

            error_messages += extract_errors_between_delimiters(error_file_lines=lines)
        except UnicodeDecodeError:
            logger.warning("Could not read file %s for errors", output_file.name)

    unique_error_messages = []
    for error_message in error_messages:
        if error_message not in unique_error_messages:
            unique_error_messages.append(error_message)

    return unique_error_messages
