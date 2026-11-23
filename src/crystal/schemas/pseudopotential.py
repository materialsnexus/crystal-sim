"""Pydantic models for pseudopotential configuration."""

from enum import StrEnum, auto
from pathlib import Path
from typing import Any, TypeAlias

from pydantic import Field, field_validator
from pymatgen.core.periodic_table import Element

from crystal.schemas.elements import ElementStr
from crystal.schemas.serialization import MSONableBaseModel

PseudopotentialFileName: TypeAlias = str
Z_VALENCE_INTEGER_TOLERANCE = 1e-6


class PseudoPotentialType(StrEnum):
    """Pseudopotential types."""

    norm_conserving = "nc"
    ultrasoft = "usp"
    paw = "paw"


class Functional(StrEnum):
    """Exchange-correlation functionals."""

    pbe = auto()
    pbesol = auto()
    lda = auto()
    pw91 = auto()
    blyp = auto()
    tpss = auto()
    scan = auto()
    hse = auto()
    pbe0 = auto()


class PseudopotentialConfig(MSONableBaseModel):
    """Pseudopotential directory paired with its element→filename map."""

    pseudopotential_dir: Path = Field(
        ..., description="Directory containing pseudopotential files"
    )
    pseudopotentials_file_map: dict[ElementStr, PseudopotentialFileName] = Field(
        ..., description="Mapping of element symbols to pseudopotential filenames"
    )

    @field_validator("pseudopotentials_file_map", mode="before")
    @classmethod
    def _normalize_elements(cls, v: dict[str, Any]) -> dict[str, Any]:
        return {element.capitalize(): filename for element, filename in v.items()}

    def __getitem__(self, element: ElementStr) -> ElementStr:
        """Get pseudopotential filename for element."""
        return self.pseudopotentials_file_map[element.capitalize()]

    def validate_elements(self, elements: set[ElementStr]) -> None:
        """Validate all elements have pseudopotentials."""
        missing = {el.capitalize() for el in elements} - set(
            self.pseudopotentials_file_map.keys()
        )
        if missing:
            msg = f"Missing pseudopotentials for: {', '.join(sorted(missing))}"
            raise ValueError(msg)


class PseudopotentialInfo(MSONableBaseModel):
    """Pseudopotential metadata extracted from UPF file."""

    element: ElementStr = Field(..., description="Element symbol (e.g., 'Si', 'O')")
    functional: Functional = Field(..., description="Exchange-correlation functional")
    type: PseudoPotentialType = Field(..., description="Pseudopotential type")
    filename: PseudopotentialFileName = Field(..., description="Original UPF filename")
    z_valence: float | None = Field(
        default=None, description="Number of valence electrons"
    )
    model_config = {"use_enum_values": False}  # noqa: RUF012

    @property
    def key(self) -> str:
        """Standard key: element.functional.type."""
        return f"{self.element}.{self.functional}.{self.type}"

    @property
    def atomic_number(self) -> int:
        """Atomic number derived from the element symbol."""
        return int(Element(self.element).Z)

    @property
    def validated_integer_z_valence(self) -> int:
        """Return z_valence coerced to an int after integer-like validation."""
        z_valence = self.z_valence
        if z_valence is None:
            error_msg = "z_valence must be set to compute core electrons."
            raise ValueError(error_msg)
        if abs(z_valence - round(z_valence)) > Z_VALENCE_INTEGER_TOLERANCE:
            error_msg = (
                "z_valence must be integer-like to compute core electrons "
                f"(received {z_valence})."
            )
            raise ValueError(error_msg)
        return int(round(z_valence))  # noqa: RUF046

    @property
    def core_electron_count(self) -> int:
        """Return the number of core electrons derived from metadata."""
        atomic_number = self.atomic_number
        z_valence = self.validated_integer_z_valence
        core_electrons = atomic_number - z_valence
        if core_electrons < 0:
            error_msg = (
                "Core electrons cannot be negative when derived from "
                f"element {self.element} and z_valence {z_valence}."
            )
            raise ValueError(error_msg)
        return core_electrons
