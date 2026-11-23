"""Tests for Chargemol core-electron helpers."""

from __future__ import annotations

from pathlib import Path

from crystal.helpers.chargemol.core_electrons import (
    build_chargemol_core_electrons_by_atomic_number_from_pseudopotential_files,
)
from tests.helpers.test_chargemol_factories import build_test_pseudopotential_config


def test_core_electron_mapping_from_pseudopotential_files() -> None:
    """Core electrons should be derived from UPF z_valence metadata."""
    pseudopotential_config = build_test_pseudopotential_config(Path(__file__))
    core_electrons = (
        build_chargemol_core_electrons_by_atomic_number_from_pseudopotential_files(
            pseudopotential_config=pseudopotential_config
        )
    )
    assert core_electrons == {3: 0, 8: 2, 41: 28}
