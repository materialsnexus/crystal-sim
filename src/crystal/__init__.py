# noqa: D104
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

import tomli
from pydantic import BaseModel, Field, field_validator

from crystal.constants.fireworks import (
    FIREWORKS_DIRNAME,
    JOBSTORE_DIRNAME,
    LOCAL_DIRNAME,
    LOCAL_FWORKER_YAML_NAME,
    LOCAL_JOBSTORE_YAML_NAME,
    LOCAL_LAUNCHPAD_YAML_NAME,
    SLURM_DIRNAME,
    SLURM_FWORKER_YAML_NAME,
    SLURM_JOBSTORE_YAML_NAME,
    SLURM_LAUNCHPAD_YAML_NAME,
    SLURM_QADAPTER_YAML_NAME,
)
from crystal.constants.project import (
    CRYSTAL_TOML_NAME,
    GIT_DIRNAME,
    PYPROJECT_TOML_NAME,
    SAMPLE_CRYSTAL_TOML_NAME,
)


def find_project_root(start: Path) -> Path:  # noqa: D103
    for p in [start, *start.parents]:
        if (p / PYPROJECT_TOML_NAME).exists() or (p / GIT_DIRNAME).exists():
            return p
    return start


CRYSTAL_ROOT = find_project_root(Path(__file__).resolve().parent)

FIREWORKS_CONFIG_DIR = CRYSTAL_ROOT / FIREWORKS_DIRNAME
JOBSTORE_CONFIG_DIR = CRYSTAL_ROOT / JOBSTORE_DIRNAME

LOCAL_FIREWORKER_DIR = FIREWORKS_CONFIG_DIR / LOCAL_DIRNAME
LOCAL_JOBSTORE_DIR = JOBSTORE_CONFIG_DIR / LOCAL_DIRNAME

LOCAL_LAUNCHPAD_YAML = LOCAL_FIREWORKER_DIR / LOCAL_LAUNCHPAD_YAML_NAME
LOCAL_FWORKER_YAML = LOCAL_FIREWORKER_DIR / LOCAL_FWORKER_YAML_NAME
LOCAL_JOBSTORE_YAML = LOCAL_JOBSTORE_DIR / LOCAL_JOBSTORE_YAML_NAME

SLURM_FIREWORKER_DIR = FIREWORKS_CONFIG_DIR / SLURM_DIRNAME
SLURM_JOBSTORE_DIR = JOBSTORE_CONFIG_DIR / SLURM_DIRNAME

SLURM_LAUNCHPAD_YAML = SLURM_FIREWORKER_DIR / SLURM_LAUNCHPAD_YAML_NAME
SLURM_FWORKER_YAML = SLURM_FIREWORKER_DIR / SLURM_FWORKER_YAML_NAME
SLURM_QADAPTER_YAML = SLURM_FIREWORKER_DIR / SLURM_QADAPTER_YAML_NAME
SLURM_JOBSTORE_YAML = SLURM_JOBSTORE_DIR / SLURM_JOBSTORE_YAML_NAME


EMPTY_STRING = ""
CRYSTAL_TOML_PATH: Final[Path] = CRYSTAL_ROOT / CRYSTAL_TOML_NAME
SAMPLE_CRYSTAL_TOML_PATH: Final[Path] = CRYSTAL_ROOT / SAMPLE_CRYSTAL_TOML_NAME


def _expand_optional_path(value: Any, base_dir: Path | None = None) -> Any:
    if value in (None, EMPTY_STRING):
        return None
    raw = os.path.expandvars(Path(str(value)).expanduser().as_posix())
    path = Path(raw)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    return path


class Programs(BaseModel):
    """Scientific program settings."""

    quantum_espresso_dir: Path | None = Field(default=None)
    gpaw_env_dir: Path | None = Field(default=None)
    gpaw_setups_dir: Path | None = Field(default=None)
    pseudopotentials_dir: Path | None = Field(default=CRYSTAL_ROOT)
    mlip_models_dir: Path | None = Field(default=CRYSTAL_ROOT / "mlip_models")
    isolated_atom_energies_dir: Path | None = Field(
        default=CRYSTAL_ROOT / "isolated_atom_energies"
    )
    chargemol_dir: Path | None = Field(default=None)
    setvars: Path | None = Field(default=None)
    mpi_environment_setup_script: Path | None = Field(default=None)

    @field_validator("*", mode="before")
    @classmethod
    def expand_paths(cls, v: Any) -> Any:
        """Expand Path-typed fields."""
        return _expand_optional_path(v)


class CrystalConfig(BaseModel):
    """Aggregated configuration for all crystal services and libraries."""

    programs: Programs = Field(default_factory=Programs)

    @classmethod
    def from_toml(cls, toml_path: Path) -> CrystalConfig:
        """Load configuration from a TOML file."""
        if not toml_path.is_file():
            msg = "Please include a toml file at the repo root"
            raise FileNotFoundError(msg)

        with toml_path.open("rb") as toml_file:
            toml_as_dict = tomli.load(toml_file)

        programs = toml_as_dict.get("programs")
        if isinstance(programs, dict):
            for field in Programs.model_fields:
                if field in programs:
                    programs[field] = _expand_optional_path(
                        programs[field], base_dir=toml_path.parent
                    )

        return CrystalConfig.model_validate(toml_as_dict)


def setup_configuration(toml_path: Path = CRYSTAL_TOML_PATH) -> CrystalConfig:
    """Setup CrystalConfig."""
    if toml_path.is_file():
        return CrystalConfig.from_toml(toml_path)
    if SAMPLE_CRYSTAL_TOML_PATH.is_file():
        return CrystalConfig.from_toml(SAMPLE_CRYSTAL_TOML_PATH)
    error_msg = f"{toml_path.name} and/or {SAMPLE_CRYSTAL_TOML_PATH.name} must exist in the root directory!"
    raise FileNotFoundError(error_msg)


CRYSTAL_CONFIG = setup_configuration()
