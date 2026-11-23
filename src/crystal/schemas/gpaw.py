"""Pydantic models for GPAW settings used in CRYSTAL."""

from __future__ import annotations

import subprocess
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

from packaging.version import InvalidVersion, Version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NegativeInt,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    StrictBool,
    model_validator,
)
from pymatgen.core.structure import Structure as PMG_Structure

from crystal import CRYSTAL_CONFIG, CRYSTAL_ROOT, CrystalConfig
from crystal.constants.gpaw import (
    DEFAULT_GPAW_CONVERGENCE_DENSITY,
    DEFAULT_GPAW_CONVERGENCE_EIGENSTATES,
    DEFAULT_GPAW_CONVERGENCE_ENERGY,
    DEFAULT_GPAW_CONVERGENCE_FORCES,
    DEFAULT_GPAW_CUTOFF_EV,
    DEFAULT_GPAW_FERMI_DIRAC_WIDTH,
    DEFAULT_GPAW_GAMMA_CENTERED,
    DEFAULT_GPAW_ISOLATED_ATOM_NUMBER_OF_BANDS,
    DEFAULT_GPAW_ISOLATED_ATOM_VACUUM_ANGSTROM,
    DEFAULT_GPAW_KPOINT_DENSITY,
    DEFAULT_GPAW_MAXITER,
    DEFAULT_GPAW_PARALLEL_SL_AUTO,
    DEFAULT_GPAW_PARALLEL_USE_ELPA,
    DEFAULT_GPAW_SLAB_INTERFACE_PBC,
    DEFAULT_GPAW_SPINPOL,
    DEFAULT_GPAW_XC,
    EXTERNAL_GPAW_CONFIG_NOT_FOUND_MESSAGE,
    EXTERNAL_GPAW_ENV_DIR_NOT_FOUND_MESSAGE_TEMPLATE,
    EXTERNAL_GPAW_EXECUTABLE_NOT_FOUND_MESSAGE_TEMPLATE,
    EXTERNAL_GPAW_IMPORT_CHECK_FAILED_MESSAGE_TEMPLATE,
    EXTERNAL_GPAW_PYTHON_NOT_FOUND_MESSAGE_TEMPLATE,
    EXTERNAL_GPAW_SETUPS_CONFIG_NOT_FOUND_MESSAGE,
    EXTERNAL_GPAW_SETUPS_DIR_NOT_FOUND_MESSAGE_TEMPLATE,
    EXTERNAL_GPAW_VERSION_TOO_OLD_MESSAGE_TEMPLATE,
    GPAW_BULK_UNSUPPORTED_SLAB_ONLY_FIELDS_MESSAGE,
    GPAW_CHARGEMOL_TASK_DOCUMENT_RUN_INFO_FORBIDDEN_MESSAGE,
    GPAW_CHARGEMOL_TASK_DOCUMENT_RUN_INFO_REQUIRED_MESSAGE,
    GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
    GPAW_ENV_PYTHON_RELATIVE_PATH,
    GPAW_RELAX_OUTPUT_STEP_FORCES_COUNT_MISMATCH_MESSAGE_TEMPLATE,
    GPAW_RELAX_OUTPUT_STEP_POSITIONS_COUNT_MISMATCH_MESSAGE_TEMPLATE,
    GPAW_SLAB_INTERFACE_PBC_INVALID_MESSAGE,
    GPAW_SLAB_INTERFACE_PBC_REQUIRED_MESSAGE,
    GPAW_SLAB_INTERFACE_POISSONSOLVER_REQUIRED_MESSAGE,
    GPAW_TASK_DOCUMENT_OUTPUT_ERROR_MISMATCH_MESSAGE,
    MIN_SUPPORTED_GPAW_VERSION,
)
from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.backend import RelaxBackend, SinglePointBackend
from crystal.schemas.base import BaseTaskDocument
from crystal.schemas.elements import ElementStr
from crystal.schemas.process import ProcessRunInfo
from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from crystal.schemas.relax_settings import RelaxOptimizerSettings
from crystal.schemas.relaxation_constraints import RelaxationConstraints
from crystal.schemas.serialization import MSONABLE_JSON_MODE, MSONableBaseModel
from crystal.schemas.settings import MakerSettings, SimulationSettings


class GpawBulkOrSlab(StrEnum):
    """Physical system branch controlling bulk vs slab/interface behavior."""

    bulk = auto()
    slab_interface = auto()


class GpawOccupationsName(StrEnum):
    """Supported GPAW occupation models in CRYSTAL."""

    fixed_uniform = "fixed-uniform"
    fermi_dirac = "fermi-dirac"


class GpawEigensolverName(StrEnum):
    """Supported GPAW eigensolver models in CRYSTAL."""

    ppcg = auto()
    davidson = auto()
    etdm_fdpw = "etdm-fdpw"


class GpawDensityMixerBackend(StrEnum):
    """Supported GPAW density-mixer backends."""

    msr1 = auto()
    pulay = auto()
    no_mixing = "no-mixing"


class GpawDensityMixerSpinChannelMethod(StrEnum):
    """Supported GPAW density-mixer spin-channel methods."""

    fullspin = auto()


class GpawDipoleLayer(StrEnum):
    """Supported dipole-layer planes for slab/interface simulations."""

    xy = auto()


class GpawTaskStatus(StrEnum):
    """Status of a GPAW task document."""

    success = auto()
    failed = auto()


GpawVector3: TypeAlias = tuple[float, float, float]
GpawStressVector6: TypeAlias = tuple[float, float, float, float, float, float]
GpawPbc3: TypeAlias = tuple[StrictBool, StrictBool, StrictBool]
GpawCell3x3: TypeAlias = tuple[GpawVector3, GpawVector3, GpawVector3]


class GpawRuntimeProvenance(MSONableBaseModel):
    """Runtime facts resolved by GPAW after calculator construction."""

    gpaw_version: str = Field(min_length=1)
    atomic_data: tuple[GpawAtomicDataInfo, ...] = Field(min_length=1)

    @property
    def atomic_data_by_element(self) -> dict[ElementStr, GpawAtomicDataInfo]:
        """Return resolved PAW data indexed by element."""
        return {
            atomic_data.element_symbol: atomic_data for atomic_data in self.atomic_data
        }

    @property
    def elements(self) -> frozenset[ElementStr]:
        """Return elements covered by resolved PAW data."""
        return frozenset(self.atomic_data_by_element)

    @model_validator(mode="after")
    def _validate_unique_atomic_data_elements(self) -> Self:
        if len(self.atomic_data_by_element) != len(self.atomic_data):
            msg = "atomic_data must contain at most one entry per element."
            raise ValueError(msg)
        return self


class GpawExternalScfOutputPayload(MSONableBaseModel):
    """Validated output payload produced by the external GPAW SCF script."""

    energy: float
    forces: list[GpawVector3]
    stress: GpawStressVector6
    pbc: GpawPbc3
    runtime_provenance: GpawRuntimeProvenance


class GpawExternalIsolatedAtomOutputPayload(MSONableBaseModel):
    """Validated output payload from the external GPAW isolated-atom script."""

    energy: float
    magnetic_moment: float
    eigensolver: GpawEigensolverName
    runtime_provenance: GpawRuntimeProvenance


class GpawExternalRelaxStepOutputPayload(MSONableBaseModel):
    """Validated per-step output payload produced by the external GPAW relax script."""

    step: int
    symbols: list[str]
    positions: list[GpawVector3]
    cell: GpawCell3x3
    pbc: GpawPbc3
    forces: list[GpawVector3]
    stress: GpawStressVector6
    energy: float

    @model_validator(mode="after")
    def validate_step_atom_counts(self) -> Self:
        """Ensure all atom-resolved arrays match the symbol count."""
        atom_count = len(self.symbols)
        if len(self.positions) != atom_count:
            raise ValueError(
                GPAW_RELAX_OUTPUT_STEP_POSITIONS_COUNT_MISMATCH_MESSAGE_TEMPLATE.format(
                    positions_count=len(self.positions), symbols_count=atom_count
                )
            )
        if len(self.forces) != atom_count:
            raise ValueError(
                GPAW_RELAX_OUTPUT_STEP_FORCES_COUNT_MISMATCH_MESSAGE_TEMPLATE.format(
                    forces_count=len(self.forces), symbols_count=atom_count
                )
            )
        return self


class GpawExternalRelaxOutputPayload(MSONableBaseModel):
    """Validated output payload produced by the external GPAW relax script."""

    relax_steps: list[GpawExternalRelaxStepOutputPayload]


class GpawConvergenceSettings(MSONableBaseModel):
    """Convergence criteria for GPAW SCF."""

    density: PositiveFloat = Field(
        default=DEFAULT_GPAW_CONVERGENCE_DENSITY,
        description=(
            "Density convergence threshold. Tighten for higher accuracy; loosen for "
            "faster exploratory runs."
        ),
    )
    energy: PositiveFloat = Field(
        default=DEFAULT_GPAW_CONVERGENCE_ENERGY,
        description=(
            "Energy convergence threshold in eV per valence electron over recent SCF "
            "steps."
        ),
    )
    forces: PositiveFloat | None = Field(
        default=DEFAULT_GPAW_CONVERGENCE_FORCES,
        description=(
            "Force convergence threshold in eV/Angstrom. Usually relevant for relax "
            "workflows."
        ),
    )
    eigenstates: PositiveFloat = Field(
        default=DEFAULT_GPAW_CONVERGENCE_EIGENSTATES,
        description=(
            "Eigenstate convergence threshold. Tighten for difficult metallic or "
            "near-degenerate systems."
        ),
    )


class GpawOccupationsSettings(MSONableBaseModel):
    """Occupation settings for GPAW."""

    name: GpawOccupationsName = Field(
        default=GpawOccupationsName.fixed_uniform,
        description=(
            "Occupation model. Change from fixed-uniform to a smearing model for "
            "metallic systems."
        ),
    )
    width: NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Fermi-Dirac smearing width in eV. Defaults to 0.005 eV for "
            "Fermi-Dirac occupations."
        ),
    )

    @model_validator(mode="after")
    def _validate_and_default_width(self) -> Self:
        if self.name is GpawOccupationsName.fermi_dirac:
            if self.width is None:
                self.width = DEFAULT_GPAW_FERMI_DIRAC_WIDTH
            return self
        if self.width is not None:
            msg = "Occupation width is supported only for fermi-dirac occupations."
            raise ValueError(msg)
        return self


class GpawEigensolverSettings(MSONableBaseModel):
    """Eigensolver settings for GPAW."""

    name: GpawEigensolverName = Field(
        default=GpawEigensolverName.ppcg,
        description=(
            "SCF eigensolver selection. Change only when troubleshooting convergence "
            "or comparing solver behavior."
        ),
    )


class GpawMixerSettings(MSONableBaseModel):
    """Mixer settings for GPAW."""

    model_config = ConfigDict(extra="forbid")

    backend: GpawDensityMixerBackend = Field(
        default=GpawDensityMixerBackend.msr1,
        description=(
            "Density-mixer backend passed to GPAW's mixer['backend'] setting."
        ),
    )
    method: GpawDensityMixerSpinChannelMethod = Field(
        default=GpawDensityMixerSpinChannelMethod.fullspin,
        description=(
            "Density-mixer spin-channel method passed to GPAW's mixer['method'] "
            "setting."
        ),
    )
    beta: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description=(
            "MSR1 or Pulay density-mixer beta passed to GPAW's mixer['beta'] setting. "
            "Leave unset to use the selected backend's default."
        ),
    )
    nmaxold: PositiveInt | None = Field(
        default=None,
        description=(
            "MSR1 or Pulay density-history length passed to GPAW's "
            "mixer['nmaxold'] setting."
        ),
    )
    weight: PositiveFloat | None = Field(
        default=None,
        description=(
            "MSR1 or Pulay density-mixer metric weight passed to GPAW's "
            "mixer['weight'] setting."
        ),
    )

    @model_validator(mode="after")
    def _validate_no_mixing_excludes_parameters(self) -> Self:
        if self.backend is not GpawDensityMixerBackend.no_mixing:
            return self
        if any(
            parameter is not None
            for parameter in (self.beta, self.nmaxold, self.weight)
        ):
            msg = (
                "The no-mixing density mixer backend cannot be combined with "
                "density-mixer parameters."
            )
            raise ValueError(msg)
        return self

    def build_gpaw_mixer_kwargs(self) -> dict[str, Any]:
        """Build the mixer mapping accepted by the public GPAW calculator."""
        return self.model_dump(mode=MSONABLE_JSON_MODE, exclude_none=True)


class GpawParallelSettings(MSONableBaseModel):
    """Parallelization settings for GPAW."""

    auto_scalapack_parallelization: bool = Field(
        default=DEFAULT_GPAW_PARALLEL_SL_AUTO,
        serialization_alias="sl_auto",
        description=(
            "Enable automatic ScaLAPACK parallel layout selection. Maps to GPAW "
            "parallel['sl_auto']."
        ),
    )
    use_elpa: bool = Field(
        default=DEFAULT_GPAW_PARALLEL_USE_ELPA,
        description=(
            "Use ELPA-backed diagonalization when available in the target runtime "
            "environment."
        ),
    )


class GpawPoissonSolverSettings(MSONableBaseModel):
    """Poisson solver settings for slab/interface simulations."""

    dipolelayer: GpawDipoleLayer = Field(
        default=GpawDipoleLayer.xy,
        description=(
            "Dipole correction plane for slab/interface calculations with vacuum "
            "along c."
        ),
    )


class GpawCalculatorSettings(MSONableBaseModel):
    """Reusable GPAW calculator settings independent of system periodicity."""

    model_config = ConfigDict(extra="forbid")

    xc: str = Field(
        default=DEFAULT_GPAW_XC, description="Exchange-correlation functional label."
    )
    spin_polarisation: bool = Field(
        default=DEFAULT_GPAW_SPINPOL,
        description=(
            "Enable spin-polarized calculations. Turn on for magnetic or open-shell "
            "systems."
        ),
    )
    cutoff_ev: PositiveFloat = Field(
        default=DEFAULT_GPAW_CUTOFF_EV,
        description=(
            "Plane-wave cutoff in eV. Increase for stricter basis convergence checks."
        ),
    )
    kpoint_density: PositiveFloat | None = Field(
        default=DEFAULT_GPAW_KPOINT_DENSITY,
        description=(
            "Reciprocal-space sampling density. Increase for periodic bulk accuracy."
        ),
    )
    gamma_centered: bool = Field(
        default=DEFAULT_GPAW_GAMMA_CENTERED,
        description="Use Gamma-centered k-point grids.",
    )
    maxiter: PositiveInt = Field(
        default=DEFAULT_GPAW_MAXITER,
        description=(
            "Maximum SCF iterations before termination. Increase for hard-to-converge "
            "cases."
        ),
    )
    convergence: GpawConvergenceSettings = Field(
        default_factory=GpawConvergenceSettings
    )
    occupations: GpawOccupationsSettings = Field(
        default_factory=GpawOccupationsSettings
    )
    eigensolver: GpawEigensolverSettings = Field(
        default_factory=GpawEigensolverSettings
    )
    mixer: GpawMixerSettings = Field(default_factory=GpawMixerSettings)
    parallel: GpawParallelSettings | None = Field(default_factory=GpawParallelSettings)
    kpoint_size: tuple[PositiveInt, PositiveInt, PositiveInt] | None = Field(
        default=None,
        description="Explicit k-point grid dimensions, when density sampling is unsuitable.",
    )
    charge: float | None = Field(
        default=None, description="Net electronic charge passed to GPAW."
    )
    enable_hund_rule_initialisation: bool | None = Field(
        default=None,
        description="Enable Hund-rule initialisation for open-shell atoms.",
    )
    symmetry: str | None = Field(default=None, description="GPAW symmetry setting.")
    number_of_bands: PositiveInt | NegativeInt | None = Field(
        default=None, description="GPAW band-count setting or adjustment."
    )

    @model_validator(mode="after")
    def _validate_kpoint_sampling(self) -> Self:
        if (self.kpoint_density is None) == (self.kpoint_size is None):
            msg = "Set exactly one of kpoint_density or kpoint_size."
            raise ValueError(msg)
        return self

    @model_validator(mode="before")
    @classmethod
    def _default_density_to_none_for_explicit_kpoint_size(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and data.get("kpoint_size") is not None
            and "kpoint_density" not in data
        ):
            normalized_data = dict(data)
            normalized_data["kpoint_density"] = None
            return normalized_data
        return data

    def build_gpaw_calculator_kwargs(self) -> dict[str, Any]:
        """Build kwargs compatible with the public gpaw.GPAW calculator."""
        kpoints: dict[str, Any]
        if self.kpoint_size is not None:
            kpoints = {"size": self.kpoint_size, "gamma": self.gamma_centered}
        else:
            kpoints = {"density": self.kpoint_density, "gamma": self.gamma_centered}
        mixer_kwargs = self.mixer.build_gpaw_mixer_kwargs()
        kwargs: dict[str, Any] = {
            "xc": self.xc,
            "spinpol": self.spin_polarisation,
            "mode": {"name": "pw", "ecut": self.cutoff_ev},
            "kpts": kpoints,
            "maxiter": self.maxiter,
            "convergence": self.convergence.model_dump(
                mode=MSONABLE_JSON_MODE, exclude_none=True
            ),
            "occupations": self.occupations.model_dump(
                mode=MSONABLE_JSON_MODE, exclude_none=True
            ),
            "eigensolver": self.eigensolver.model_dump(
                mode=MSONABLE_JSON_MODE, exclude_none=True
            ),
            "mixer": mixer_kwargs,
        }
        if self.parallel is not None:
            kwargs["parallel"] = self.parallel.model_dump(
                mode=MSONABLE_JSON_MODE, exclude_none=True, by_alias=True
            )
        optional_calculator_fields = {
            "charge": self.charge,
            "hund": self.enable_hund_rule_initialisation,
            "symmetry": self.symmetry,
            "nbands": self.number_of_bands,
        }
        kwargs.update(
            {
                key: value
                for key, value in optional_calculator_fields.items()
                if value is not None
            }
        )
        return kwargs


class GpawSettings(GpawCalculatorSettings):
    """GPAW settings for periodic bulk and slab/interface systems."""

    bulk_or_slab: GpawBulkOrSlab = Field(
        default=GpawBulkOrSlab.bulk,
        description=(
            "Select bulk or slab/interface behavior. Slab/interface enables additional "
            "electrostatics settings."
        ),
    )
    poissonsolver: GpawPoissonSolverSettings | None = Field(
        default=None,
        description=(
            "Poisson solver overrides for slab/interface systems. Not used for bulk."
        ),
    )
    pbc: tuple[bool, bool, bool] | None = Field(
        default=None,
        description=(
            "Periodic boundary condition flags. For slab/interface this is fixed to "
            "(True, True, False)."
        ),
    )

    @model_validator(mode="after")
    def validate_bulk_or_slab_specific_fields(self) -> Self:
        """Validate slab/interface-only fields based on system kind."""
        if self.bulk_or_slab == GpawBulkOrSlab.bulk:
            if self.poissonsolver is not None or self.pbc is not None:
                raise ValueError(GPAW_BULK_UNSUPPORTED_SLAB_ONLY_FIELDS_MESSAGE)
            return self

        if self.pbc is None:
            raise ValueError(GPAW_SLAB_INTERFACE_PBC_REQUIRED_MESSAGE)
        if self.poissonsolver is None:
            raise ValueError(GPAW_SLAB_INTERFACE_POISSONSOLVER_REQUIRED_MESSAGE)
        if self.pbc != DEFAULT_GPAW_SLAB_INTERFACE_PBC:
            raise ValueError(GPAW_SLAB_INTERFACE_PBC_INVALID_MESSAGE)
        return self

    def build_gpaw_calculator_kwargs(self) -> dict[str, Any]:
        """Build GPAW kwargs and add periodic electrostatics settings."""
        kwargs = super().build_gpaw_calculator_kwargs()
        if self.poissonsolver is not None:
            kwargs["poissonsolver"] = self.poissonsolver.model_dump(
                mode=MSONABLE_JSON_MODE, exclude_none=True
            )
        return kwargs


class GpawScfSimulationSettings(SimulationSettings):  # noqa: D101
    gpaw_settings: GpawSettings = Field(default_factory=GpawSettings)


class GpawIsolatedAtomConvergenceSettings(GpawConvergenceSettings):
    """Convergence settings for isolated atoms without force convergence."""

    model_config = ConfigDict(extra="forbid")

    forces: None = None


class GpawIsolatedAtomOccupationsSettings(GpawOccupationsSettings):
    """Fixed occupations required by the isolated-atom energy recipe."""

    model_config = ConfigDict(extra="forbid")

    name: GpawOccupationsName = GpawOccupationsName.fixed_uniform
    width: None = None

    @model_validator(mode="after")
    def _validate_fixed_uniform_occupations(self) -> Self:
        if self.name is not GpawOccupationsName.fixed_uniform:
            msg = "Isolated atoms require fixed-uniform occupations."
            raise ValueError(msg)
        return self


class GpawIsolatedAtomEigensolverSettings(GpawEigensolverSettings):
    """Direct-minimization eigensolver required by the isolated-atom recipe."""

    model_config = ConfigDict(extra="forbid")

    name: GpawEigensolverName = GpawEigensolverName.etdm_fdpw

    @model_validator(mode="after")
    def _validate_direct_minimization_eigensolver(self) -> Self:
        if self.name is not GpawEigensolverName.etdm_fdpw:
            msg = "Isolated atoms require the etdm-fdpw eigensolver."
            raise ValueError(msg)
        return self


class GpawIsolatedAtomMixerSettings(GpawMixerSettings):
    """No-mixing policy required by the isolated-atom energy recipe."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal[GpawDensityMixerBackend.no_mixing] = (
        GpawDensityMixerBackend.no_mixing
    )
    method: Literal[GpawDensityMixerSpinChannelMethod.fullspin] = (
        GpawDensityMixerSpinChannelMethod.fullspin
    )
    beta: None = None
    nmaxold: None = None
    weight: None = None


class GpawIsolatedAtomCalculatorSettings(GpawCalculatorSettings):
    """GPAW calculator settings constrained to the isolated-atom energy recipe."""

    spin_polarisation: Literal[True] = True
    kpoint_density: None = None
    kpoint_size: tuple[Literal[1], Literal[1], Literal[1]] = (1, 1, 1)
    gamma_centered: Literal[True] = True
    parallel: None = None
    convergence: GpawIsolatedAtomConvergenceSettings = Field(
        default_factory=GpawIsolatedAtomConvergenceSettings
    )
    occupations: GpawIsolatedAtomOccupationsSettings = Field(
        default_factory=GpawIsolatedAtomOccupationsSettings
    )
    eigensolver: GpawIsolatedAtomEigensolverSettings = Field(
        default_factory=GpawIsolatedAtomEigensolverSettings
    )
    mixer: GpawIsolatedAtomMixerSettings = Field(
        default_factory=GpawIsolatedAtomMixerSettings
    )
    charge: float = Field(default=0.0, ge=0.0, le=0.0)
    enable_hund_rule_initialisation: Literal[True] = True
    symmetry: Literal["off"] = "off"
    number_of_bands: int = Field(
        default=DEFAULT_GPAW_ISOLATED_ATOM_NUMBER_OF_BANDS,
        ge=DEFAULT_GPAW_ISOLATED_ATOM_NUMBER_OF_BANDS,
        le=DEFAULT_GPAW_ISOLATED_ATOM_NUMBER_OF_BANDS,
    )


class GpawIsolatedAtomSimulationSettings(SimulationSettings):
    """Scientific settings for one neutral GPAW isolated-atom calculation."""

    model_config = ConfigDict(extra="forbid")

    calculator_settings: GpawIsolatedAtomCalculatorSettings = Field(
        default_factory=GpawIsolatedAtomCalculatorSettings
    )
    vacuum_angstrom: PositiveFloat = Field(
        default=DEFAULT_GPAW_ISOLATED_ATOM_VACUUM_ANGSTROM,
        description="Vacuum padding around the centred atom in Angstrom.",
    )

    def build_gpaw_calculator_kwargs(self) -> dict[str, Any]:
        """Build calculator kwargs from validated isolated-atom settings."""
        return self.calculator_settings.build_gpaw_calculator_kwargs()

    def build_gpaw_ppcg_calculator_kwargs(self) -> dict[str, Any]:
        """Build isolated-atom kwargs with the PPCG eigensolver."""
        calculator_kwargs = self.build_gpaw_calculator_kwargs()
        calculator_kwargs["eigensolver"] = {"name": GpawEigensolverName.ppcg}
        return calculator_kwargs


class GpawRelaxSimulationSettings(GpawScfSimulationSettings, RelaxOptimizerSettings):
    """Settings for GPAW relaxation simulations."""

    constraints: RelaxationConstraints | None = None


class GpawChargemolSimulationSettings(SimulationSettings):
    """GPAW-to-Chargemol settings.

    Note: this workflow currently assumes neutral systems. `job_control.txt`
    is written with net charge fixed to `0.0`.
    """


class GpawScfChargemolSimulationSettings(SimulationSettings):  # noqa: D101
    gpaw_scf_simulation_settings: GpawScfSimulationSettings
    gpaw_chargemol_simulation_settings: GpawChargemolSimulationSettings


class GpawScfChargemolMakerSettings(MakerSettings):  # noqa: D101
    backend: Literal[SinglePointBackend.gpaw] = SinglePointBackend.gpaw
    simulation_settings: GpawScfChargemolSimulationSettings


class GpawRelaxMakerSettings(MakerSettings):  # noqa: D101
    backend: Literal[RelaxBackend.gpaw] = RelaxBackend.gpaw
    simulation_settings: GpawRelaxSimulationSettings


class GpawTaskDocument(BaseTaskDocument):  # noqa: D101
    dir_name: str = Field(description="The directory for this task")
    status: GpawTaskStatus = Field(description="Task status")

    @property
    def converged(self) -> bool:  # noqa: D102
        return self.status is GpawTaskStatus.success

    @staticmethod
    def _validate_status_errors_and_properties(
        *, status: GpawTaskStatus, errors: str, properties: object | None
    ) -> None:
        is_success = status is GpawTaskStatus.success
        has_errors = errors != ""
        has_properties = properties is not None
        if is_success and (has_errors or not has_properties):
            raise ValueError(GPAW_TASK_DOCUMENT_OUTPUT_ERROR_MISMATCH_MESSAGE)
        if (not is_success) and ((not has_errors) or has_properties):
            raise ValueError(GPAW_TASK_DOCUMENT_OUTPUT_ERROR_MISMATCH_MESSAGE)

    @staticmethod
    def _status_and_error_log_from_error_log(
        error_log: str,
    ) -> tuple[GpawTaskStatus, str]:
        if error_log == "":
            return GpawTaskStatus.success, ""
        return GpawTaskStatus.failed, error_log


class GpawChargemolTaskDocument(BaseTaskDocument):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    properties: SinglePointProperties | None = Field(
        description="Computed single-point properties from GPAW SCF."
    )
    structure: PMG_Structure | None = Field(
        description="Structure with DDEC6 charges in site_properties"
    )
    chargemol_run_info: ProcessRunInfo | None = Field(
        description="Process metadata for Chargemol run"
    )

    @property
    def is_success(self) -> bool:  # noqa: D102
        return self.errors == ""

    @property
    def converged(self) -> bool:  # noqa: D102
        return self.is_success

    @model_validator(mode="after")
    def _validate_errors_and_outputs(self) -> Self:
        if self.is_success and (
            self.properties is None
            or self.structure is None
            or self.chargemol_run_info is None
        ):
            raise ValueError(GPAW_CHARGEMOL_TASK_DOCUMENT_RUN_INFO_REQUIRED_MESSAGE)
        if not self.is_success and (
            self.properties is not None
            or self.structure is not None
            or self.chargemol_run_info is not None
        ):
            raise ValueError(GPAW_CHARGEMOL_TASK_DOCUMENT_RUN_INFO_FORBIDDEN_MESSAGE)
        return self


class GpawScfChargemolTaskDocument(GpawChargemolTaskDocument):
    """Combined GPAW SCF and Chargemol result with SCF audit metadata."""

    gpaw_scf_simulation_settings: GpawScfSimulationSettings | None = None
    runtime_provenance: GpawRuntimeProvenance | None = None

    @model_validator(mode="after")
    def _validate_scf_audit_metadata(self) -> Self:
        has_scf_audit_metadata = (
            self.gpaw_scf_simulation_settings is not None
            and self.runtime_provenance is not None
        )
        if self.is_success and not has_scf_audit_metadata:
            msg = "Successful GPAW SCF plus Chargemol tasks require runtime provenance."
            raise ValueError(msg)
        if not self.is_success and has_scf_audit_metadata:
            msg = "Failed GPAW SCF plus Chargemol tasks cannot include runtime provenance."
            raise ValueError(msg)
        if (
            self.is_success
            and self.structure is not None
            and self.runtime_provenance is not None
        ):
            missing_elements = (
                frozenset(self.structure.symbol_set) - self.runtime_provenance.elements
            )
            if missing_elements:
                msg = "GPAW label runtime PAW data must cover every label element."
                raise ValueError(msg)
        return self


class GpawRelaxTaskDocument(GpawTaskDocument):  # noqa: D101
    properties: RelaxProperties | None = Field(
        description="Computed relax properties from a GPAW task"
    )

    @model_validator(mode="after")
    def _validate_status_error_log_and_properties(self) -> Self:
        self._validate_status_errors_and_properties(
            status=self.status, errors=self.errors, properties=self.properties
        )
        return self

    @classmethod
    def from_directory_and_properties(  # noqa: D102
        cls, dir_path: Path, properties: RelaxProperties | None, error_log: str
    ) -> Self:
        status, normalized_error_log = cls._status_and_error_log_from_error_log(
            error_log
        )
        return cls(
            dir_name=dir_path.as_posix(),
            status=status,
            errors=normalized_error_log,
            properties=properties,
        )


class GpawExternalInstallPathsConfig(BaseModel):
    """Paths needed to run GPAW from a dedicated external environment."""

    gpaw_env_root: Path = Field(...)
    python_executable_path: Path = Field(...)
    gpaw_executable_path: Path = Field(...)
    gpaw_setups_dir: Path = Field(...)


class GpawExternalInstallPathsConfigLoader:
    """Load and validate external GPAW runtime paths from crystal.toml."""

    @classmethod
    def from_crystal_config(
        cls, crystal_config: CrystalConfig
    ) -> GpawExternalInstallPathsConfig:
        """Load external GPAW runtime paths from CrystalConfig."""
        gpaw_env_dir = crystal_config.programs.gpaw_env_dir
        if gpaw_env_dir is None:
            raise FileNotFoundError(EXTERNAL_GPAW_CONFIG_NOT_FOUND_MESSAGE)
        gpaw_setups_dir = crystal_config.programs.gpaw_setups_dir
        if gpaw_setups_dir is None:
            raise FileNotFoundError(EXTERNAL_GPAW_SETUPS_CONFIG_NOT_FOUND_MESSAGE)
        resolved_root = cls._resolve_config_path(gpaw_env_dir)
        resolved_setups_dir = cls._resolve_config_path(gpaw_setups_dir)
        return GpawExternalInstallPathsConfig(
            gpaw_env_root=resolved_root,
            python_executable_path=resolved_root / GPAW_ENV_PYTHON_RELATIVE_PATH,
            gpaw_executable_path=resolved_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
            gpaw_setups_dir=resolved_setups_dir,
        )

    @classmethod
    def load_and_validate_external_runtime(
        cls, crystal_config: CrystalConfig | None = None
    ) -> GpawExternalInstallPathsConfig:
        """Load external GPAW paths and validate runtime usability."""
        config = cls.from_crystal_config(crystal_config or CRYSTAL_CONFIG)
        cls._validate_gpaw_env_root_exists(config.gpaw_env_root)
        cls._validate_gpaw_setups_dir_exists(config.gpaw_setups_dir)
        cls._validate_python_executable_exists(config.python_executable_path)
        cls._validate_gpaw_executable_exists(config.gpaw_executable_path)
        installed_gpaw_version = cls._validate_python_can_import_gpaw(
            config.python_executable_path
        )
        cls._validate_minimum_supported_gpaw_version(
            installed_gpaw_version=installed_gpaw_version,
            python_executable_path=config.python_executable_path,
        )
        return config

    @staticmethod
    def _resolve_config_path(path_value: Path) -> Path:
        if path_value.is_absolute():
            return path_value
        return (CRYSTAL_ROOT / path_value).resolve()

    @staticmethod
    def _validate_gpaw_env_root_exists(gpaw_env_root: Path) -> None:
        if gpaw_env_root.is_dir():
            return
        error_msg = EXTERNAL_GPAW_ENV_DIR_NOT_FOUND_MESSAGE_TEMPLATE.format(
            gpaw_env_root=gpaw_env_root
        )
        raise FileNotFoundError(error_msg)

    @staticmethod
    def _validate_gpaw_setups_dir_exists(gpaw_setups_dir: Path) -> None:
        if gpaw_setups_dir.is_dir():
            return
        error_msg = EXTERNAL_GPAW_SETUPS_DIR_NOT_FOUND_MESSAGE_TEMPLATE.format(
            gpaw_setups_dir=gpaw_setups_dir
        )
        raise FileNotFoundError(error_msg)

    @staticmethod
    def _validate_python_executable_exists(python_executable_path: Path) -> None:
        if python_executable_path.is_file():
            return
        error_msg = EXTERNAL_GPAW_PYTHON_NOT_FOUND_MESSAGE_TEMPLATE.format(
            python_executable_path=python_executable_path
        )
        raise FileNotFoundError(error_msg)

    @staticmethod
    def _validate_gpaw_executable_exists(gpaw_executable_path: Path) -> None:
        if gpaw_executable_path.is_file():
            return
        error_msg = EXTERNAL_GPAW_EXECUTABLE_NOT_FOUND_MESSAGE_TEMPLATE.format(
            gpaw_executable_path=gpaw_executable_path
        )
        raise FileNotFoundError(error_msg)

    @staticmethod
    def _validate_python_can_import_gpaw(python_executable_path: Path) -> str:
        command_arguments = [
            str(python_executable_path),
            "-c",
            "import gpaw; print(gpaw.__version__)",
        ]
        try:
            completed_process = subprocess.run(
                command_arguments, check=True, text=True, capture_output=True
            )
            installed_gpaw_version = completed_process.stdout.strip()
            if installed_gpaw_version:
                return installed_gpaw_version
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            error_msg = EXTERNAL_GPAW_IMPORT_CHECK_FAILED_MESSAGE_TEMPLATE.format(
                python_executable_path=python_executable_path
            )
            raise RuntimeError(error_msg) from exc
        error_msg = EXTERNAL_GPAW_IMPORT_CHECK_FAILED_MESSAGE_TEMPLATE.format(
            python_executable_path=python_executable_path
        )
        raise RuntimeError(error_msg)

    @staticmethod
    def _validate_minimum_supported_gpaw_version(
        installed_gpaw_version: str, python_executable_path: Path
    ) -> None:
        try:
            parsed_installed_version = Version(installed_gpaw_version)
            minimum_supported_version = Version(MIN_SUPPORTED_GPAW_VERSION)
        except InvalidVersion as exc:
            error_msg = EXTERNAL_GPAW_IMPORT_CHECK_FAILED_MESSAGE_TEMPLATE.format(
                python_executable_path=python_executable_path
            )
            raise RuntimeError(error_msg) from exc

        if parsed_installed_version >= minimum_supported_version:
            return

        error_msg = EXTERNAL_GPAW_VERSION_TOO_OLD_MESSAGE_TEMPLATE.format(
            installed_version=installed_gpaw_version,
            minimum_supported_version=MIN_SUPPORTED_GPAW_VERSION,
            python_executable_path=python_executable_path,
        )
        raise RuntimeError(error_msg)
