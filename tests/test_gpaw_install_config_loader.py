"""Tests for external GPAW runtime config discovery and validation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from crystal import CRYSTAL_ROOT, CrystalConfig
from crystal.constants.gpaw import (
    GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
    GPAW_ENV_PYTHON_RELATIVE_PATH,
)
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.schemas.gpaw import GpawExternalInstallPathsConfigLoader
from tests.helpers.constants.gpaw import (
    GPAW_MIN_SUPPORTED_VERSION,
    GPAW_VERSION_BELOW_MINIMUM,
)
from tests.helpers.test_paths import (
    TEST_GPAW_ENV_ROOT_DIRNAME,
    TEST_GPAW_SETUPS_DIRNAME,
    TEST_RELATIVE_GPAW_ENV_DIRNAME,
)

TEST_EXTERNAL_GPAW_PYTHON_EXECUTABLE_MODE = 0o755


def _create_external_gpaw_runtime_bin_file(
    executable_path: Path, script_text: str = "#!/usr/bin/env bash\nexit 0\n"
) -> None:
    executable_path.parent.mkdir(parents=True, exist_ok=True)
    executable_path.write_text(script_text, encoding=DEFAULT_TEXT_ENCODING)
    executable_path.chmod(TEST_EXTERNAL_GPAW_PYTHON_EXECUTABLE_MODE)


def test_external_config_loader_uses_gpaw_env_dir_from_programs(tmp_path: Path) -> None:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    gpaw_setups_dir = tmp_path / TEST_GPAW_SETUPS_DIRNAME
    config = CrystalConfig.model_validate(
        {
            "programs": {
                "gpaw_env_dir": gpaw_env_root.as_posix(),
                "gpaw_setups_dir": gpaw_setups_dir.as_posix(),
            }
        }
    )
    loaded = GpawExternalInstallPathsConfigLoader.from_crystal_config(config)
    assert loaded.gpaw_env_root == gpaw_env_root
    assert loaded.python_executable_path == (
        gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH
    )
    assert loaded.gpaw_executable_path == (
        gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH
    )
    assert loaded.gpaw_setups_dir == gpaw_setups_dir


def test_external_config_loader_resolves_relative_program_dir() -> None:
    config = CrystalConfig.model_validate(
        {
            "programs": {
                "gpaw_env_dir": TEST_RELATIVE_GPAW_ENV_DIRNAME,
                "gpaw_setups_dir": TEST_GPAW_SETUPS_DIRNAME,
            }
        }
    )
    loaded = GpawExternalInstallPathsConfigLoader.from_crystal_config(config)
    expected_root = (CRYSTAL_ROOT / TEST_RELATIVE_GPAW_ENV_DIRNAME).resolve()
    expected_setups_dir = (CRYSTAL_ROOT / TEST_GPAW_SETUPS_DIRNAME).resolve()
    assert loaded.gpaw_env_root == expected_root
    assert loaded.python_executable_path == (
        expected_root / GPAW_ENV_PYTHON_RELATIVE_PATH
    )
    assert loaded.gpaw_executable_path == (
        expected_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH
    )
    assert loaded.gpaw_setups_dir == expected_setups_dir


def test_external_config_loader_raises_when_missing() -> None:
    config = CrystalConfig.model_validate({})
    with pytest.raises(FileNotFoundError, match="gpaw_env_dir"):
        _ = GpawExternalInstallPathsConfigLoader.from_crystal_config(config)


def test_external_config_loader_raises_when_setups_dir_missing() -> None:
    config = CrystalConfig.model_validate(
        {"programs": {"gpaw_env_dir": TEST_RELATIVE_GPAW_ENV_DIRNAME}}
    )
    with pytest.raises(FileNotFoundError, match="gpaw_setups_dir"):
        _ = GpawExternalInstallPathsConfigLoader.from_crystal_config(config)


def test_loader_raises_when_env_dir_does_not_exist(tmp_path: Path) -> None:
    config = CrystalConfig.model_validate(
        {
            "programs": {
                "gpaw_env_dir": (tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME).as_posix(),
                "gpaw_setups_dir": (tmp_path / TEST_GPAW_SETUPS_DIRNAME).as_posix(),
            }
        }
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _ = GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime(
            config
        )


def test_loader_raises_when_python_executable_is_missing(tmp_path: Path) -> None:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    gpaw_env_root.mkdir()
    gpaw_setups_dir = tmp_path / TEST_GPAW_SETUPS_DIRNAME
    gpaw_setups_dir.mkdir()
    _create_external_gpaw_runtime_bin_file(
        gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH
    )
    config = CrystalConfig.model_validate(
        {
            "programs": {
                "gpaw_env_dir": gpaw_env_root.as_posix(),
                "gpaw_setups_dir": gpaw_setups_dir.as_posix(),
            }
        }
    )
    with pytest.raises(FileNotFoundError, match="python"):
        _ = GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime(
            config
        )


def test_loader_raises_when_gpaw_executable_is_missing(tmp_path: Path) -> None:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    gpaw_setups_dir = tmp_path / TEST_GPAW_SETUPS_DIRNAME
    gpaw_setups_dir.mkdir()
    _create_external_gpaw_runtime_bin_file(
        gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH
    )
    config = CrystalConfig.model_validate(
        {
            "programs": {
                "gpaw_env_dir": gpaw_env_root.as_posix(),
                "gpaw_setups_dir": gpaw_setups_dir.as_posix(),
            }
        }
    )
    with pytest.raises(FileNotFoundError, match="gpaw executable"):
        _ = GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime(
            config
        )


def test_loader_raises_when_setups_dir_does_not_exist(tmp_path: Path) -> None:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    _create_external_gpaw_runtime_bin_file(
        gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH
    )
    _create_external_gpaw_runtime_bin_file(
        gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH
    )
    config = CrystalConfig.model_validate(
        {
            "programs": {
                "gpaw_env_dir": gpaw_env_root.as_posix(),
                "gpaw_setups_dir": (tmp_path / TEST_GPAW_SETUPS_DIRNAME).as_posix(),
            }
        }
    )
    with pytest.raises(FileNotFoundError, match="setups"):
        _ = GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime(
            config
        )


def test_loader_raises_when_python_cannot_import_gpaw(tmp_path: Path) -> None:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    gpaw_setups_dir = tmp_path / TEST_GPAW_SETUPS_DIRNAME
    gpaw_setups_dir.mkdir()
    _create_external_gpaw_runtime_bin_file(
        gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH
    )
    _create_external_gpaw_runtime_bin_file(
        gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH
    )
    config = CrystalConfig.model_validate(
        {
            "programs": {
                "gpaw_env_dir": gpaw_env_root.as_posix(),
                "gpaw_setups_dir": gpaw_setups_dir.as_posix(),
            }
        }
    )

    with pytest.raises(RuntimeError, match="import gpaw"):
        _ = GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime(
            config
        )


def test_loader_raises_when_gpaw_version_is_below_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    gpaw_setups_dir = tmp_path / TEST_GPAW_SETUPS_DIRNAME
    gpaw_setups_dir.mkdir()
    _create_external_gpaw_runtime_bin_file(
        gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH
    )
    _create_external_gpaw_runtime_bin_file(
        gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH
    )
    config = CrystalConfig.model_validate(
        {
            "programs": {
                "gpaw_env_dir": gpaw_env_root.as_posix(),
                "gpaw_setups_dir": gpaw_setups_dir.as_posix(),
            }
        }
    )

    def _mock_subprocess_run_for_external_gpaw_version_below_minimum(
        command_arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command_arguments,
            returncode=0,
            stdout=f"{GPAW_VERSION_BELOW_MINIMUM}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "crystal.schemas.gpaw.subprocess.run",
        _mock_subprocess_run_for_external_gpaw_version_below_minimum,
    )

    with pytest.raises(RuntimeError, match=GPAW_MIN_SUPPORTED_VERSION):
        _ = GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime(
            config
        )


def test_loader_accepts_gpaw_version_26_7_or_higher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    gpaw_setups_dir = tmp_path / TEST_GPAW_SETUPS_DIRNAME
    gpaw_setups_dir.mkdir()
    python_executable_path = gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH
    _create_external_gpaw_runtime_bin_file(python_executable_path)
    _create_external_gpaw_runtime_bin_file(
        gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH
    )
    config = CrystalConfig.model_validate(
        {
            "programs": {
                "gpaw_env_dir": gpaw_env_root.as_posix(),
                "gpaw_setups_dir": gpaw_setups_dir.as_posix(),
            }
        }
    )

    def _mock_subprocess_run_for_external_gpaw_min_supported_version(
        command_arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command_arguments,
            returncode=0,
            stdout=f"{GPAW_MIN_SUPPORTED_VERSION}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "crystal.schemas.gpaw.subprocess.run",
        _mock_subprocess_run_for_external_gpaw_min_supported_version,
    )

    loaded = GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime(
        config
    )
    assert loaded.python_executable_path == python_executable_path
