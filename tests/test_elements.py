"""Unit tests for shared element symbol schema validation."""

import pytest
from pydantic import TypeAdapter, ValidationError

from crystal.schemas.elements import (
    AtomicNumber,
    ElementStr,
    atomic_number_to_element_str,
    element_str_to_atomic_number,
)

ELEMENT_SYMBOL_ADAPTER: TypeAdapter[ElementStr] = TypeAdapter(ElementStr)
ATOMIC_NUMBER_ADAPTER: TypeAdapter[AtomicNumber] = TypeAdapter(AtomicNumber)


@pytest.mark.parametrize(
    ("raw_symbol", "expected_symbol"),
    [("si", "Si"), ("SI", "Si"), ("sI", "Si"), ("o", "O"), ("C", "C"), ("na", "Na")],
)
def test_element_symbol_normalizes_case(raw_symbol: str, expected_symbol: str) -> None:
    assert ELEMENT_SYMBOL_ADAPTER.validate_python(raw_symbol) == expected_symbol


@pytest.mark.parametrize("raw_symbol", ["X", "5", "?", "Aa", ""])
def test_element_symbol_rejects_invalid_values(raw_symbol: str) -> None:
    with pytest.raises(ValidationError, match="Invalid element symbol"):
        ELEMENT_SYMBOL_ADAPTER.validate_python(raw_symbol)


@pytest.mark.parametrize("raw_atomic_number", [1, 2, 8, 14, 118])
def test_atomic_number_accepts_valid_values(raw_atomic_number: int) -> None:
    assert ATOMIC_NUMBER_ADAPTER.validate_python(raw_atomic_number) == raw_atomic_number


@pytest.mark.parametrize("raw_atomic_number", [0, -1, 119])
def test_atomic_number_rejects_invalid_values(raw_atomic_number: int) -> None:
    with pytest.raises(ValidationError, match="Invalid atomic number"):
        ATOMIC_NUMBER_ADAPTER.validate_python(raw_atomic_number)


def test_atomic_number_and_element_symbol_conversion_helpers() -> None:
    assert atomic_number_to_element_str(14) == "Si"
    assert element_str_to_atomic_number("si") == 14

    with pytest.raises(ValueError, match="Invalid atomic number"):
        atomic_number_to_element_str(-1)
