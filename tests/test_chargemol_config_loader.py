"""Tests for Chargemol installation config discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from crystal import CRYSTAL_ROOT, CrystalConfig
from crystal.constants.chargemol import (
    CHARGEMOL_ATOMIC_DENSITIES_SUBDIR,
    CHARGEMOL_EXECUTABLE_RELATIVE_PATH,
)
from crystal.schemas.chargemol import ChargemolInstallPathsConfigLoader
from tests.helpers.test_paths import (
    TEST_CHARGEMOL_ROOT_DIRNAME,
    TEST_RELATIVE_CHARGEMOL_DIRNAME,
)


def test_config_loader_uses_chargemol_dir_from_programs(tmp_path: Path) -> None:
    """Programs chargemol_dir should derive chargemol paths."""
    chargemol_root = tmp_path / TEST_CHARGEMOL_ROOT_DIRNAME
    config = CrystalConfig.model_validate(
        {"programs": {"chargemol_dir": chargemol_root.as_posix()}}
    )
    loaded = ChargemolInstallPathsConfigLoader.from_crystal_config(config)
    assert (
        loaded.chargemol_binary_path
        == chargemol_root / CHARGEMOL_EXECUTABLE_RELATIVE_PATH
    )
    assert (
        loaded.atomic_densities_directory
        == chargemol_root / CHARGEMOL_ATOMIC_DENSITIES_SUBDIR
    )


def test_config_loader_resolves_relative_program_dir() -> None:
    """Relative programs.chargemol_dir should resolve from CRYSTAL_ROOT."""
    config = CrystalConfig.model_validate(
        {"programs": {"chargemol_dir": TEST_RELATIVE_CHARGEMOL_DIRNAME}}
    )
    loaded = ChargemolInstallPathsConfigLoader.from_crystal_config(config)
    expected_root = (CRYSTAL_ROOT / TEST_RELATIVE_CHARGEMOL_DIRNAME).resolve()
    assert (
        loaded.chargemol_binary_path
        == expected_root / CHARGEMOL_EXECUTABLE_RELATIVE_PATH
    )
    assert (
        loaded.atomic_densities_directory
        == expected_root / CHARGEMOL_ATOMIC_DENSITIES_SUBDIR
    )


def test_config_loader_raises_when_missing() -> None:
    """Missing chargemol config should raise a clear error."""
    config = CrystalConfig.model_validate({})
    with pytest.raises(FileNotFoundError, match="chargemol"):
        _ = ChargemolInstallPathsConfigLoader.from_crystal_config(config)
