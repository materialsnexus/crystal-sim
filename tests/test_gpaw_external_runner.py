"""Tests for shared external GPAW launch mechanics."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from crystal.calculators.gpaw.external_runner import (
    build_external_gpaw_command_arguments,
    build_external_gpaw_environment,
    run_external_gpaw_script,
)
from crystal.constants.gpaw import GPAW_SETUP_PATH_ENVIRONMENT_VARIABLE
from crystal.schemas.gpaw import GpawExternalInstallPathsConfig
from tests.helpers.test_paths import (
    TEST_GPAW_ENV_ROOT_DIRNAME,
    TEST_GPAW_SETUPS_DIRNAME,
)

TEST_GPAW_PROCESS_COUNT = 4


def _external_runtime_config(tmp_path: Path) -> GpawExternalInstallPathsConfig:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    return GpawExternalInstallPathsConfig(
        gpaw_env_root=gpaw_env_root,
        python_executable_path=gpaw_env_root / "bin" / "python",
        gpaw_executable_path=gpaw_env_root / "bin" / "gpaw",
        gpaw_setups_dir=tmp_path / TEST_GPAW_SETUPS_DIRNAME,
    )


def test_build_external_gpaw_environment_preserves_environment_and_sets_setups_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CRYSTAL_TEST_KEEP", "present")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    external_runtime_config = _external_runtime_config(tmp_path)
    additional_pythonpath = tmp_path / "additional-pythonpath"

    environment = build_external_gpaw_environment(
        external_runtime_config,
        paths_to_prepend_to_pythonpath=(tmp_path, additional_pythonpath),
    )

    assert environment["CRYSTAL_TEST_KEEP"] == "present"
    assert environment[GPAW_SETUP_PATH_ENVIRONMENT_VARIABLE] == (
        external_runtime_config.gpaw_setups_dir.as_posix()
    )
    assert environment["PYTHONPATH"] == os.pathsep.join(
        (tmp_path.as_posix(), additional_pythonpath.as_posix())
    )


def test_build_external_gpaw_command_arguments_uses_mpirun_launcher(
    tmp_path: Path,
) -> None:
    external_runtime_config = _external_runtime_config(tmp_path)
    script_path = tmp_path / "external_script.py"
    structure_path = tmp_path / "input.extxyz"
    kwargs_path = tmp_path / "kwargs.json"
    output_path = tmp_path / "output.json"

    command_arguments = build_external_gpaw_command_arguments(
        external_runtime_config=external_runtime_config,
        number_of_processes=TEST_GPAW_PROCESS_COUNT,
        script_path=script_path,
        structure_path=structure_path,
        kwargs_path=kwargs_path,
        output_path=output_path,
    )

    assert command_arguments == [
        "mpirun",
        "-np",
        str(TEST_GPAW_PROCESS_COUNT),
        external_runtime_config.gpaw_executable_path.as_posix(),
        "python",
        script_path.as_posix(),
        structure_path.as_posix(),
        kwargs_path.as_posix(),
        output_path.as_posix(),
    ]


def test_run_external_gpaw_script_returns_output_path_without_processing_json(
    tmp_path: Path,
) -> None:
    external_runtime_config = _external_runtime_config(tmp_path)
    output_path = tmp_path / "output.json"
    output_path.write_text("not json", encoding="utf-8")
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        ["gpaw"], 0
    )

    with patch(
        "crystal.calculators.gpaw.external_runner.subprocess.run",
        return_value=completed_process,
    ):
        result = run_external_gpaw_script(
            external_runtime_config=external_runtime_config,
            number_of_processes=TEST_GPAW_PROCESS_COUNT,
            script_path=tmp_path / "external_script.py",
            structure_path=tmp_path / "input.extxyz",
            kwargs_path=tmp_path / "kwargs.json",
            output_path=output_path,
            run_directory=tmp_path,
            failure_message="External GPAW run failed.",
        )

    assert result == output_path


def test_run_external_gpaw_script_wraps_external_process_os_error(
    tmp_path: Path,
) -> None:
    external_runtime_config = _external_runtime_config(tmp_path)
    process_error = NotADirectoryError("external runtime working directory")

    with (
        patch(
            "crystal.calculators.gpaw.external_runner.subprocess.run",
            side_effect=process_error,
        ),
        pytest.raises(RuntimeError, match="External GPAW run failed"),
    ):
        run_external_gpaw_script(
            external_runtime_config=external_runtime_config,
            number_of_processes=TEST_GPAW_PROCESS_COUNT,
            script_path=tmp_path / "external_script.py",
            structure_path=tmp_path / "input.extxyz",
            kwargs_path=tmp_path / "kwargs.json",
            output_path=tmp_path / "output.json",
            run_directory=tmp_path,
            failure_message="External GPAW run failed.",
        )
