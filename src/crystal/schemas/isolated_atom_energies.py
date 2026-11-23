"""Pydantic models for isolated atom energy references."""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

import yaml
from ase.io import read
from pydantic import TypeAdapter, model_validator

from crystal import CRYSTAL_CONFIG
from crystal.helpers.file import count_args_not_none, validate_dir_exists
from crystal.schemas.elements import ElementStr
from crystal.schemas.serialization import MSONableBaseModel

ISOLATED_ATOM_ENERGIES_FILENAME = "isolated_atom_energies.extxyz"
ISOLATED_ATOM_ENERGIES_METADATA_FILENAME = "metadata.yaml"

IsolatedAtomEnergies: TypeAlias = dict[ElementStr, float]
ISOLATED_ATOM_ENERGIES_ADAPTER = TypeAdapter(IsolatedAtomEnergies)
ISOLATED_ATOM_ENERGY_METADATA_ADAPTER = TypeAdapter(dict[str, object])


class IsolatedAtomEnergiesRegistry(StrEnum):
    """Known isolated atom energy artifact names."""

    gpaw = auto()


class IsolatedAtomEnergiesRef(MSONableBaseModel):
    """Reference to an isolated atom energy artifact directory."""

    _REGISTRY_MODEL_FIELD: ClassVar[str] = "registry_energies"
    _PATH_FIELD: ClassVar[str] = "path"

    registry_energies: IsolatedAtomEnergiesRegistry = IsolatedAtomEnergiesRegistry.gpaw
    path: Path | None = None

    def _validate_isolated_atom_energies_dir_is_defined(self) -> None:
        if CRYSTAL_CONFIG.programs.isolated_atom_energies_dir is None:
            error_msg = (
                "Isolated atom energies dir is not defined in crystal.toml file!"
            )
            raise ValueError(error_msg)

    @model_validator(mode="before")
    @classmethod
    def _ensure_path_or_registry(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        path_or_registry_count = count_args_not_none(
            data.get(cls._REGISTRY_MODEL_FIELD), data.get(cls._PATH_FIELD)
        )

        if path_or_registry_count > 1:
            error_msg = "Provide only one of path or registry_energies, not both."
            raise ValueError(error_msg)

        return data

    @property
    def artifact_path(self) -> Path:
        """Resolved artifact directory path."""
        if self.path is not None:
            path = self.path
        else:
            self._validate_isolated_atom_energies_dir_is_defined()
            assert CRYSTAL_CONFIG.programs.isolated_atom_energies_dir is not None
            assert self.registry_energies is not None
            path = (
                CRYSTAL_CONFIG.programs.isolated_atom_energies_dir
                / self.registry_energies.value
            )
        validate_dir_exists(path)
        return path

    @property
    def energies(self) -> IsolatedAtomEnergies:
        """Loaded isolated atom energies."""
        return self.load_energies_from_extxyz(
            self.artifact_path / ISOLATED_ATOM_ENERGIES_FILENAME
        )

    @property
    def metadata(self) -> dict[str, object]:
        """Loaded isolated atom energy metadata."""
        metadata_payload = yaml.safe_load(
            (self.artifact_path / ISOLATED_ATOM_ENERGIES_METADATA_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        return ISOLATED_ATOM_ENERGY_METADATA_ADAPTER.validate_python(metadata_payload)

    @staticmethod
    def load_energies_from_extxyz(path: Path) -> IsolatedAtomEnergies:
        """Load isolated atom energies from an extxyz file."""
        structures = read(path, index=":")

        energies: dict[str, float] = {}
        for structure in structures:
            if len(structure) > 1:
                error_msg = (
                    "Isolated atom energy file must contain only single atom "
                    "structures."
                )
                raise ValueError(error_msg)
            element = structure.get_chemical_symbols()[0]
            if element in energies:
                error_msg = f"Duplicate isolated atom energy reference for {element}."
                raise ValueError(error_msg)
            energies[element] = structure.get_potential_energy()
        return ISOLATED_ATOM_ENERGIES_ADAPTER.validate_python(energies)
