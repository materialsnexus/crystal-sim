"""Unit tests for serialized chemical formula schema validation."""

import pytest
from pydantic import TypeAdapter, ValidationError

from crystal.schemas.formula import FormulaStr

FORMULA_STR_ADAPTER: TypeAdapter[FormulaStr] = TypeAdapter(FormulaStr)


@pytest.mark.parametrize(
    ("raw_formula", "expected_formula"),
    [
        ("Na1Cl1", "Cl1Na1"),
        ("NaCl", "Cl1Na1"),
        ("na1cl1", "Cl1Na1"),
        ("H2O", "H2O1"),
        ("SI2o4", "O4Si2"),
        ("c3o1", "C3O1"),
        ("O2Si", "O2Si1"),
        ("SiO2", "O2Si1"),
    ],
)
def test_formula_str_normalizes_tokens(raw_formula: str, expected_formula: str) -> None:
    assert FORMULA_STR_ADAPTER.validate_python(raw_formula) == expected_formula


def test_formula_str_rejects_empty_string() -> None:
    with pytest.raises(ValidationError, match="Chemical formula cannot be empty"):
        FORMULA_STR_ADAPTER.validate_python("")


@pytest.mark.parametrize(
    "raw_formula", ["Na0Cl1", "Na-1Cl1", "Na1 Cl1", "Na1-Cl1", "1Na1Cl1O"]
)
def test_formula_str_rejects_invalid_token_sequences(raw_formula: str) -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "Chemical formula must be a sequence of "
            "<ElementStr><PositiveInteger> tokens with no separators"
        ),
    ):
        FORMULA_STR_ADAPTER.validate_python(raw_formula)


@pytest.mark.parametrize("raw_formula", ["Na1Na1", "na1NA2", "C1c2"])
def test_formula_str_rejects_repeated_elements(raw_formula: str) -> None:
    with pytest.raises(
        ValidationError, match="Chemical formula must not repeat elements"
    ):
        FORMULA_STR_ADAPTER.validate_python(raw_formula)


def test_formula_str_rejects_invalid_element_symbols() -> None:
    with pytest.raises(ValidationError, match="Invalid element symbol"):
        FORMULA_STR_ADAPTER.validate_python("Xx2O1")
