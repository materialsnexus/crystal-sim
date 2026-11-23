"""Utility helpers for dictionary manipulations.

These helpers are focused on the patterns we use in CRYSTAL
when building nested configuration dictionaries.
"""

from __future__ import annotations

from typing import Any


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where ``overrides`` are applied on top of ``base``.

    Meant for configuration files where user configuration overrides global ones.
    """
    result: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def count_existing_keys(data: dict[str, Any], keys: tuple[str, ...]) -> int:
    """Returns the number of keys which exist in data dictionary."""
    return sum(key in data for key in set(keys))
