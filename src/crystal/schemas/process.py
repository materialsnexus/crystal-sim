"""Shared process run metadata schemas."""

from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict

from crystal.constants.process import PROCESS_EXIT_CODE_SUCCESS
from crystal.schemas.serialization import MSONableBaseModel


class ProcessRunInfo(MSONableBaseModel):  # noqa: D101
    model_config = ConfigDict(frozen=True)

    command_arguments: list[str]
    exit_code: int
    standard_output_file: Path | None = None
    standard_error_file: Path | None = None
    error_message: str | None = None

    @property
    def exit_code_ok(self) -> bool:  # noqa: D102
        return self.exit_code == PROCESS_EXIT_CODE_SUCCESS
