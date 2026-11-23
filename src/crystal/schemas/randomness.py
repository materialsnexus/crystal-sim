"""Shared random-seed types and helpers."""

from __future__ import annotations

import secrets
from typing import Annotated, Any

from pydantic import BeforeValidator, Field


def generate_random_seed() -> int:
    """Generate a 32-bit seed from OS entropy."""
    return secrets.randbits(32)


def _generate_random_seed_if_none(value: int | None) -> int:
    if value is None:
        return generate_random_seed()
    return value


RandomSeed = Annotated[
    int,
    BeforeValidator(_generate_random_seed_if_none, json_schema_input_type=int | None),
]


def random_seed_field() -> Any:
    """Reusable Field config for random-seed inputs."""
    return Field(
        default_factory=generate_random_seed,
        description="Random seed. Pass null/None to auto-generate a fresh 32-bit seed.",
    )
