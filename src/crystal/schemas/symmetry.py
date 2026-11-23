"""Shared symmetry annotated types."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator

MIN_SPACE_GROUP_NUM = 1
MAX_SPACE_GROUP_NUM = 230

ALL_SPACE_GROUPS = tuple(range(MIN_SPACE_GROUP_NUM, MAX_SPACE_GROUP_NUM + 1))


def all_space_groups_list() -> list[int]:  # noqa: D103
    return list(ALL_SPACE_GROUPS)


def all_space_groups_set() -> set[int]:  # noqa: D103
    return set(ALL_SPACE_GROUPS)


def _validate_space_group(number: int) -> int:
    if not MIN_SPACE_GROUP_NUM <= number <= MAX_SPACE_GROUP_NUM:
        error_msg = f"Space group number must be 1-230, {number} is not."
        raise ValueError(error_msg)
    return number


SpaceGroupNumber = Annotated[int, AfterValidator(_validate_space_group)]
