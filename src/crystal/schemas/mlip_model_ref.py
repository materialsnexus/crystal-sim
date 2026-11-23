"""Shared MLIP model reference schemas."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import model_validator

from crystal import CRYSTAL_CONFIG
from crystal.helpers.file import count_args_not_none, validate_dir_exists
from crystal.schemas.serialization import MSONableBaseModel


class QetModelRegistry(StrEnum):  # noqa: D101
    # Models must exist in the mlip directory provided in mlip_model_dir in crystal.toml
    foundational_matpes_pbe = "QET-PES-MatPES-PBE-2025.2"
    foundational_matq = "QET-PES-MatQ"
    finetuned_linbosi = "LiNbOSi-finetuned"
    finetuned_litaosi = "LiTaOSi-finetuned"


class MlipModelRef(MSONableBaseModel):  # noqa: D101
    registry_model: QetModelRegistry | None = None
    path: Path | None = None

    @model_validator(mode="after")
    def _validate_mlip_model_dir_is_defined(self) -> Self:
        if (
            self.registry_model is not None
            and CRYSTAL_CONFIG.programs.mlip_models_dir is None
        ):
            error_msg = "MLIP Model dir is not defined in crystal.toml file!"
            raise ValueError(error_msg)
        return self

    @property
    def model_path(self) -> Path:  # noqa: D102
        if self.path is not None:
            validate_dir_exists(self.path)
            return self.path
        assert isinstance(self.registry_model, QetModelRegistry)
        assert CRYSTAL_CONFIG.programs.mlip_models_dir is not None
        path = CRYSTAL_CONFIG.programs.mlip_models_dir / self.registry_model.value
        validate_dir_exists(path)
        return path

    @model_validator(mode="after")
    def _ensure_path_or_registry(self) -> Self:
        if count_args_not_none(self.registry_model, self.path) != 1:
            error_msg = "Provide only one of path or registry_model, not both."
            raise ValueError(error_msg)
        return self
