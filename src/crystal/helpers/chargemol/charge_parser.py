from __future__ import annotations  # noqa: D100

from functools import cached_property
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.schemas.properties import SinglePointProperties

CHARGEMOL_NET_ATOMIC_CHARGES_PROPERTY = "ddec6_charges"
NET_ATOMIC_CHARGES_ATOM_COUNT_LINE_INDEX = 0
NET_ATOMIC_CHARGES_HEADER_LINE_COUNT = 2
NET_ATOMIC_CHARGES_FIELDS_REQUIRED = 5


class ChargemolNetAtomicChargeRecord(BaseModel):
    """Single DDEC6 net atomic charge record from Chargemol output."""

    model_config = ConfigDict(frozen=True)

    atom_index: int
    atom_symbol: str
    net_atomic_charge: float


class ChargemolNetAtomicCharges(BaseModel):
    """Parsed DDEC6 net atomic charges from Chargemol output."""

    model_config = ConfigDict(frozen=True)

    charge_records: list[ChargemolNetAtomicChargeRecord]

    @cached_property
    def net_atomic_charges(self) -> list[float]:  # noqa: D102
        return [record.net_atomic_charge for record in self.charge_records]

    @cached_property
    def atom_symbols(self) -> list[str]:  # noqa: D102
        return [record.atom_symbol for record in self.charge_records]

    @cached_property
    def atom_count(self) -> int:  # noqa: D102
        return len(self.charge_records)

    @cached_property
    def total_charge(self) -> float:  # noqa: D102
        return sum(record.net_atomic_charge for record in self.charge_records)


class ChargemolNetAtomicChargesParser(BaseModel):
    """Parse DDEC6 net atomic charges from Chargemol output."""

    model_config = ConfigDict(frozen=True)

    chargemol_net_atomic_charges_path: Path

    @model_validator(mode="after")
    def _validate_chargemol_net_atomic_charges_path_exists(self) -> Self:
        if not self.chargemol_net_atomic_charges_path.is_file():
            msg = (
                "Missing Chargemol DDEC6 net atomic charges file at "
                f"{self.chargemol_net_atomic_charges_path}"
            )
            raise ValueError(msg)
        return self

    def get_chargemol_atomic_charges(self) -> ChargemolNetAtomicCharges:  # noqa: D102
        return ChargemolNetAtomicCharges(charge_records=self.charge_records)

    @cached_property
    def file_lines(self) -> list[str]:  # noqa: D102
        file_lines = self.chargemol_net_atomic_charges_path.read_text().splitlines()
        if len(file_lines) < NET_ATOMIC_CHARGES_HEADER_LINE_COUNT + 1:
            msg = "DDEC6 net atomic charges file is missing atom entries."
            raise ValueError(msg)
        return file_lines

    @cached_property
    def atom_count(self) -> int:  # noqa: D102
        try:
            atom_count = int(
                self.file_lines[NET_ATOMIC_CHARGES_ATOM_COUNT_LINE_INDEX].strip()
            )
        except ValueError as exc:
            msg = "DDEC6 net atomic charges file is missing the atom count header."
            raise ValueError(msg) from exc
        if atom_count <= 0:
            msg = "DDEC6 net atomic charges file has invalid atom count."
            raise ValueError(msg)
        return atom_count

    @cached_property
    def atom_lines(self) -> list[str]:  # noqa: D102
        first_atom_line_index = NET_ATOMIC_CHARGES_HEADER_LINE_COUNT
        last_atom_line_index = first_atom_line_index + self.atom_count
        if len(self.file_lines) < last_atom_line_index:
            msg = "DDEC6 net atomic charges file is missing atom entries."
            raise ValueError(msg)
        return self.file_lines[first_atom_line_index:last_atom_line_index]

    @cached_property
    def charge_records(self) -> list[ChargemolNetAtomicChargeRecord]:  # noqa: D102
        records: list[ChargemolNetAtomicChargeRecord] = []
        for atom_index, line in enumerate(self.atom_lines, start=1):
            fields = line.split()
            if len(fields) < NET_ATOMIC_CHARGES_FIELDS_REQUIRED:
                msg = f"DDEC6 net atomic charge entry {atom_index} is incomplete."
                raise ValueError(msg)
            atom_symbol = fields[0]
            try:
                net_atomic_charge = float(fields[4])
            except ValueError as exc:
                msg = f"DDEC6 net atomic charge value missing for atom {atom_index}."
                raise ValueError(msg) from exc
            records.append(
                ChargemolNetAtomicChargeRecord(
                    atom_index=atom_index,
                    atom_symbol=atom_symbol,
                    net_atomic_charge=net_atomic_charge,
                )
            )
        return records

    @cached_property
    def total_charge(self) -> float:  # noqa: D102
        return sum(record.net_atomic_charge for record in self.charge_records)


def attach_chargemol_net_atomic_charges_to_structure(
    structure: PMG_Structure, net_atomic_charges: ChargemolNetAtomicCharges
) -> PMG_Structure:
    """Attach DDEC6 net atomic charges to a structure site property."""
    if len(structure) != net_atomic_charges.atom_count:
        msg = "Structure atom count does not match DDEC6 atom count"
        raise ValueError(msg)
    structure_symbols = [str(specie) for specie in structure.species]
    if structure_symbols != net_atomic_charges.atom_symbols:
        msg = "Structure species do not match DDEC6 atom symbols"
        raise ValueError(msg)
    updated_structure = structure.copy()
    updated_structure.add_site_property(
        CHARGEMOL_NET_ATOMIC_CHARGES_PROPERTY, net_atomic_charges.net_atomic_charges
    )
    return updated_structure


def attach_chargemol_net_atomic_charges_to_single_point_properties(
    properties: SinglePointProperties, net_atomic_charges: ChargemolNetAtomicCharges
) -> SinglePointProperties:
    """Return revalidated single-point properties with Chargemol charges."""
    structure_with_charges = attach_chargemol_net_atomic_charges_to_structure(
        structure=properties.structure, net_atomic_charges=net_atomic_charges
    )
    property_values = properties.model_dump()
    property_values.update(
        structure=structure_with_charges, charges=net_atomic_charges.net_atomic_charges
    )
    return SinglePointProperties.model_validate(property_values)
