"""Shared single-point schemas across supported backends."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from crystal.schemas.gpaw import (
    GpawChargemolTaskDocument,
    GpawScfChargemolMakerSettings,
)
from crystal.schemas.mlip import (
    MlipSinglePointMakerSettings,
    MlipSinglePointTaskDocument,
)
from crystal.schemas.qe import QeScfPpChargemolTaskDocument
from crystal.schemas.qe_pp import QeScfPpChargemolMakerSettings

SinglePointMakerSettings = Annotated[
    QeScfPpChargemolMakerSettings
    | MlipSinglePointMakerSettings
    | GpawScfChargemolMakerSettings,
    Field(discriminator="backend"),
]

SinglePointWithChargesMakerSettings = Annotated[
    QeScfPpChargemolMakerSettings | GpawScfChargemolMakerSettings,
    Field(discriminator="backend"),
]

SinglePointWithChargesTaskDocument = (
    QeScfPpChargemolTaskDocument | GpawChargemolTaskDocument
)

SinglePointTaskDocument = (
    SinglePointWithChargesTaskDocument | MlipSinglePointTaskDocument
)
