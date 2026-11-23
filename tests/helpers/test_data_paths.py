"""Shared path utilities for test data assets."""

from __future__ import annotations

from pathlib import Path

from crystal import CRYSTAL_ROOT
from tests.helpers.test_paths import (
    TEST_PSEUDOPOTENTIALS_DIRNAME,
    TESTS_DATA_RELATIVE_DIR,
)

TESTS_DATA_DIR = (CRYSTAL_ROOT / TESTS_DATA_RELATIVE_DIR).resolve()
TESTS_PSEUDOPOTENTIALS_DIR = TESTS_DATA_DIR / TEST_PSEUDOPOTENTIALS_DIRNAME


def resolve_tests_data_dir(_test_file_path: Path) -> Path:
    """Return the test data directory."""
    return TESTS_DATA_DIR


def resolve_tests_pseudopotentials_dir(_test_file_path: Path) -> Path:
    """Return the pseudopotentials directory under test data."""
    return TESTS_PSEUDOPOTENTIALS_DIR
