"""Schema-layer helpers for serialized chemical formula strings."""

from __future__ import annotations

import re
from typing import Annotated, TypeAlias

from pydantic import AfterValidator, TypeAdapter

from crystal.schemas.elements import ElementStr

FORMULA_TOKEN_PATTERN = re.compile(r"([A-Za-z]{1,2})([1-9]\d*)?")
ELEMENT_STR_ADAPTER: TypeAdapter[ElementStr] = TypeAdapter(ElementStr)


def _validate_formula_string(formula: str) -> str:
    if formula == "":
        error_msg = "Chemical formula cannot be empty."
        raise ValueError(error_msg)

    normalized_counts: dict[str, str] = {}
    current_position = 0

    for match in FORMULA_TOKEN_PATTERN.finditer(formula):
        if match.start() != current_position:
            error_msg = (
                "Chemical formula must be a sequence of <ElementStr><PositiveInteger> "
                f"tokens with no separators: {formula}"
            )
            raise ValueError(error_msg)

        raw_element, raw_count = match.groups()
        element = ELEMENT_STR_ADAPTER.validate_python(raw_element)
        if element in normalized_counts:
            error_msg = f"Chemical formula must not repeat elements: {formula}"
            raise ValueError(error_msg)

        normalized_counts[element] = raw_count or "1"
        current_position = match.end()

    if current_position != len(formula):
        error_msg = (
            "Chemical formula must be a sequence of <ElementStr><PositiveInteger> "
            f"tokens with no separators: {formula}"
        )
        raise ValueError(error_msg)

    return "".join(
        f"{element}{count}" for element, count in sorted(normalized_counts.items())
    )


FormulaStr: TypeAlias = Annotated[str, AfterValidator(_validate_formula_string)]
