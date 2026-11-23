"""Shared relaxation schemas across supported backends."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from crystal.schemas.gpaw import GpawRelaxMakerSettings, GpawRelaxTaskDocument
from crystal.schemas.mlip import MlipRelaxMakerSettings, MlipRelaxTaskDocument
from crystal.schemas.qe import QeRelaxMakerSettings, QeRelaxTaskDocument

RelaxMakerSettings = Annotated[
    QeRelaxMakerSettings | MlipRelaxMakerSettings | GpawRelaxMakerSettings,
    Field(discriminator="backend"),
]

RelaxTaskDocument = QeRelaxTaskDocument | MlipRelaxTaskDocument | GpawRelaxTaskDocument
