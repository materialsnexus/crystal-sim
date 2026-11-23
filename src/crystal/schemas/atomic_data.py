"""Domain contracts for resolved electronic-core data."""

from __future__ import annotations

from re import fullmatch
from typing import Self

from pydantic import Field, NonNegativeInt, model_validator

from crystal.schemas.elements import ElementStr
from crystal.schemas.serialization import MSONableBaseModel

SHA256_DIGEST_PATTERN = r"[0-9a-f]{64}"


class GpawAtomicDataInfo(MSONableBaseModel):
    """Provenance for GPAW-resolved PAW data for one chemical element."""

    element_symbol: ElementStr = Field(description="Chemical element symbol.")
    paw_setup_sha256_digest: str = Field(
        description="SHA256 digest of the resolved PAW data file."
    )
    paw_setup_filename: str | None = Field(
        default=None, description="Filename of the resolved PAW data file."
    )
    paw_setup_version: str | None = Field(
        default=None, description="Version of the resolved PAW data, when available."
    )
    paw_setup_kind: str | None = Field(
        default=None, description="Kind of resolved electronic-core data, such as PAW."
    )
    valence_electron_count: NonNegativeInt | None = Field(
        default=None,
        description="Number of valence electrons represented by the resolved PAW data.",
    )

    @model_validator(mode="after")
    def _validate_paw_setup_sha256_digest(self) -> Self:
        normalized_digest = self.paw_setup_sha256_digest.lower()
        if fullmatch(SHA256_DIGEST_PATTERN, normalized_digest) is None:
            msg = (
                "paw_setup_sha256_digest must be a 64-character hexadecimal "
                "SHA256 digest."
            )
            raise ValueError(msg)
        self.paw_setup_sha256_digest = normalized_digest
        return self
