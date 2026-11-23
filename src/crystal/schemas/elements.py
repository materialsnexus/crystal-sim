"""Shared element symbol types and validation."""

from __future__ import annotations

from typing import Annotated

from ase.data import atomic_numbers, chemical_symbols
from pydantic import AfterValidator, TypeAdapter

VALID_ELEMENT_SYMBOLS = frozenset(chemical_symbols) - {"X"}
VALID_ATOMIC_NUMBERS = frozenset(atomic_numbers.values()) - {atomic_numbers["X"]}


def _validate_element_symbol(symbol: str) -> str:
    normalized = symbol.capitalize()
    if normalized not in VALID_ELEMENT_SYMBOLS:
        error_msg = f"Invalid element symbol: {symbol}"
        raise ValueError(error_msg)
    return normalized


def _validate_atomic_number(atomic_number: int) -> int:
    if atomic_number not in VALID_ATOMIC_NUMBERS:
        error_msg = f"Invalid atomic number: {atomic_number}"
        raise ValueError(error_msg)
    return atomic_number


ElementStr = Annotated[str, AfterValidator(_validate_element_symbol)]
AtomicNumber = Annotated[int, AfterValidator(_validate_atomic_number)]
ELEMENT_STR_ADAPTER: TypeAdapter[ElementStr] = TypeAdapter(ElementStr)
ATOMIC_NUMBER_ADAPTER: TypeAdapter[AtomicNumber] = TypeAdapter(AtomicNumber)


def atomic_number_to_element_str(atomic_number: AtomicNumber) -> ElementStr:
    """Validate an atomic number and return its normalized element symbol."""
    return ELEMENT_STR_ADAPTER.validate_python(
        chemical_symbols[ATOMIC_NUMBER_ADAPTER.validate_python(atomic_number)]
    )


def element_str_to_atomic_number(element: ElementStr) -> AtomicNumber:
    """Validate an element symbol and return its atomic number."""
    return ATOMIC_NUMBER_ADAPTER.validate_python(
        atomic_numbers[ELEMENT_STR_ADAPTER.validate_python(element)]
    )
