"""Pydantic models for Chargemol configuration and inputs."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from crystal import CRYSTAL_CONFIG, CRYSTAL_ROOT, CrystalConfig
from crystal.constants.chargemol import (
    CHARGEMOL_ATOMIC_DENSITIES_SUBDIR,
    CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE,
    CHARGEMOL_EXECUTABLE_RELATIVE_PATH,
)
from crystal.helpers.chargemol.core_electrons import (
    build_chargemol_core_electrons_by_atomic_number_from_pseudopotential_files,
)
from crystal.schemas.pseudopotential import PseudopotentialConfig
from crystal.schemas.qe_pp import (
    QePpChargeDensityCubeInputBuilder,
    QePpInputSettings,
    QePpPlotSettings,
)

FORTRAN_TRUE = ".true."
FORTRAN_FALSE = ".false."
CHARGEMOL_TAG_OPEN = "<"
CHARGEMOL_TAG_CLOSE = ">"
CHARGEMOL_TAG_CLOSING_PREFIX = "/"
CHARGEMOL_NEUTRAL_NET_CHARGE_VALUE = "0.0"


class ChargemolChargeType(StrEnum):
    """Charge partitioning method for Chargemol."""

    ddec6 = "ddec6"


class ChargemolJobControlTag(StrEnum):
    """Tag labels used in Chargemol job_control.txt."""

    net_charge = "net charge"
    periodicity_vectors = "periodicity along A, B, and C vectors"
    atomic_densities_directory = "atomic densities directory complete path"
    input_filename = "input filename"
    charge_type = "charge type"
    compute_bond_orders = "compute bos"
    print_atomic_densities = "print atomic densities"
    number_of_core_electrons = "number of core electrons"


CHARGEMOL_JOB_CONTROL_TAG_ORDER: tuple[ChargemolJobControlTag, ...] = (
    ChargemolJobControlTag.net_charge,
    ChargemolJobControlTag.periodicity_vectors,
    ChargemolJobControlTag.atomic_densities_directory,
    ChargemolJobControlTag.input_filename,
    ChargemolJobControlTag.charge_type,
    ChargemolJobControlTag.compute_bond_orders,
    ChargemolJobControlTag.print_atomic_densities,
    ChargemolJobControlTag.number_of_core_electrons,
)


class ChargemolJobControlSettings(BaseModel):
    """Inputs required to build a Chargemol job_control.txt file."""

    input_filename: Path = Field(...)
    atomic_densities_directory: Path = Field(...)
    periodic_a: bool
    periodic_b: bool
    periodic_c: bool
    charge_type: ChargemolChargeType = Field(default=ChargemolChargeType.ddec6)
    compute_bond_orders: bool = Field(default=True)
    print_atomic_densities: bool = Field(default=False)
    core_electrons_by_atomic_number: dict[int, int] = Field(...)

    @classmethod
    def from_qe_pseudopotentials(
        cls,
        *,
        input_cube_path: Path,
        atomic_densities_directory: Path,
        pseudopotential_config: PseudopotentialConfig,
    ) -> ChargemolJobControlSettings:
        """Build Chargemol settings from QE pseudopotentials and cube path."""
        core_electrons_by_atomic_number = (
            build_chargemol_core_electrons_by_atomic_number_from_pseudopotential_files(
                pseudopotential_config=pseudopotential_config
            )
        )
        return cls(
            input_filename=input_cube_path,
            atomic_densities_directory=atomic_densities_directory,
            periodic_a=True,
            periodic_b=True,
            periodic_c=True,
            core_electrons_by_atomic_number=core_electrons_by_atomic_number,
        )

    @classmethod
    def from_qe_pp_inputs(
        cls,
        *,
        cube_output_directory: Path,
        input_settings: QePpInputSettings,
        plot_settings: QePpPlotSettings,
        atomic_densities_directory: Path,
        pseudopotential_config: PseudopotentialConfig,
    ) -> ChargemolJobControlSettings:
        """Build Chargemol settings from QE pp.x inputs."""
        cube_input_builder = QePpChargeDensityCubeInputBuilder(
            input_settings=input_settings, plot_settings=plot_settings
        )
        input_cube_path = cube_output_directory / cube_input_builder.cube_filename
        return cls.from_qe_pseudopotentials(
            input_cube_path=input_cube_path,
            atomic_densities_directory=atomic_densities_directory,
            pseudopotential_config=pseudopotential_config,
        )


class QePpToChargemolWorkflowInputs(BaseModel):
    """Inputs required to hand off QE pp.x data to Chargemol."""

    cube_output_directory: Path
    input_settings: QePpInputSettings
    plot_settings: QePpPlotSettings
    pseudopotential_config: PseudopotentialConfig


class ChargemolJobControlInputBuilder(BaseModel):
    """Build Chargemol job_control.txt inputs."""

    settings: ChargemolJobControlSettings

    @property
    def job_control_text(self) -> str:
        """Return job_control.txt contents using closing-tag format."""
        return "\n".join(self.job_control_lines)

    @property
    def job_control_lines(self) -> list[str]:
        """Return job_control.txt lines in canonical order."""
        formatter = ChargemolJobControlTagBlockFormatter(settings=self.settings)
        return formatter.job_control_lines

    @staticmethod
    def format_chargemol_opening_tag(tag: ChargemolJobControlTag) -> str:
        """Return an opening tag for job_control.txt."""
        return f"{CHARGEMOL_TAG_OPEN}{tag.value}{CHARGEMOL_TAG_CLOSE}"

    @staticmethod
    def format_chargemol_closing_tag(tag: ChargemolJobControlTag) -> str:
        """Return a closing tag for job_control.txt."""
        return (
            f"{CHARGEMOL_TAG_OPEN}{CHARGEMOL_TAG_CLOSING_PREFIX}"
            f"{tag.value}{CHARGEMOL_TAG_CLOSE}"
        )

    @classmethod
    def build_chargemol_tag_block_lines(
        cls, tag: ChargemolJobControlTag, values: list[str]
    ) -> list[str]:
        """Return a full tag block for job_control.txt."""
        return [
            cls.format_chargemol_opening_tag(tag),
            *values,
            cls.format_chargemol_closing_tag(tag),
        ]

    @staticmethod
    def format_chargemol_fortran_bool(*, fortran_bool_value: bool) -> str:
        """Return a Fortran-style boolean string for job_control.txt."""
        return FORTRAN_TRUE if fortran_bool_value else FORTRAN_FALSE

    @staticmethod
    def format_chargemol_atomic_densities_directory(path: Path) -> str:
        """Return an atomic densities directory path with a trailing separator."""
        path_str = str(path)
        if path_str.endswith(("/", "\\")):
            return path_str
        return f"{path_str}{os.sep}"


class ChargemolJobControlTagBlockFormatter(BaseModel):
    """Format Chargemol job_control.txt lines from settings."""

    settings: ChargemolJobControlSettings

    @property
    def job_control_lines(self) -> list[str]:
        """Return job_control.txt lines in canonical order."""
        tag_values = self._job_control_tag_values
        ordered_lines: list[str] = []
        for tag in CHARGEMOL_JOB_CONTROL_TAG_ORDER:
            values = tag_values.get(tag)
            if values is None:
                continue
            ordered_lines.extend(
                ChargemolJobControlInputBuilder.build_chargemol_tag_block_lines(
                    tag=tag, values=values
                )
            )
        return ordered_lines

    @property
    def _job_control_tag_values(self) -> dict[ChargemolJobControlTag, list[str]]:
        tag_values: dict[ChargemolJobControlTag, list[str]] = {
            ChargemolJobControlTag.net_charge: [CHARGEMOL_NEUTRAL_NET_CHARGE_VALUE],
            ChargemolJobControlTag.periodicity_vectors: [
                ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                    fortran_bool_value=self.settings.periodic_a
                ),
                ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                    fortran_bool_value=self.settings.periodic_b
                ),
                ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                    fortran_bool_value=self.settings.periodic_c
                ),
            ],
            ChargemolJobControlTag.atomic_densities_directory: [
                ChargemolJobControlInputBuilder.format_chargemol_atomic_densities_directory(
                    self.settings.atomic_densities_directory
                )
            ],
            ChargemolJobControlTag.input_filename: [str(self.settings.input_filename)],
            ChargemolJobControlTag.charge_type: [self.settings.charge_type],
            ChargemolJobControlTag.compute_bond_orders: [
                ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                    fortran_bool_value=self.settings.compute_bond_orders
                )
            ],
            ChargemolJobControlTag.print_atomic_densities: [
                ChargemolJobControlInputBuilder.format_chargemol_fortran_bool(
                    fortran_bool_value=self.settings.print_atomic_densities
                )
            ],
            ChargemolJobControlTag.number_of_core_electrons: self._core_electron_entry_lines,
        }
        return tag_values

    @property
    def _core_electron_entry_lines(self) -> list[str]:
        return [
            f"{atomic_number} {core_electrons}"
            for atomic_number, core_electrons in sorted(
                self.settings.core_electrons_by_atomic_number.items()
            )
        ]


class ChargemolInstallPathsConfig(BaseModel):
    """Paths needed to run Chargemol on the local machine."""

    chargemol_binary_path: Path = Field(...)
    atomic_densities_directory: Path = Field(...)


class ChargemolInstallPathsConfigLoader:
    """Load Chargemol installation paths from crystal.toml."""

    @classmethod
    def from_crystal_config(
        cls, crystal_config: CrystalConfig
    ) -> ChargemolInstallPathsConfig:
        """Load Chargemol installation paths from CrystalConfig."""
        chargemol_dir = crystal_config.programs.chargemol_dir
        if chargemol_dir is not None:
            resolved_root = cls._resolve_config_path(chargemol_dir)
            return ChargemolInstallPathsConfig(
                chargemol_binary_path=resolved_root
                / CHARGEMOL_EXECUTABLE_RELATIVE_PATH,
                atomic_densities_directory=(
                    resolved_root / CHARGEMOL_ATOMIC_DENSITIES_SUBDIR
                ),
            )
        raise FileNotFoundError(CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE)

    @classmethod
    def discover(cls) -> ChargemolInstallPathsConfig:
        """Discover Chargemol installation paths from crystal.toml."""
        return cls.from_crystal_config(CRYSTAL_CONFIG)

    @staticmethod
    def _resolve_config_path(path_value: Path) -> Path:
        if path_value.is_absolute():
            return path_value
        return (CRYSTAL_ROOT / path_value).resolve()
