"""Shared mechanics for launching scripts in the external GPAW runtime."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from crystal.constants.gpaw import (
    GPAW_LAUNCH_PYTHON_SUBCOMMAND,
    GPAW_MPI_LAUNCHER,
    GPAW_MPI_PROCESS_COUNT_FLAG,
    GPAW_RUNTIME_PROVENANCE_FILENAME,
    GPAW_SETUP_PATH_ENVIRONMENT_VARIABLE,
)
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.schemas.gpaw import GpawExternalInstallPathsConfig

_GPAW_RUNTIME_PROVENANCE_SOURCE_PATH = (
    Path(__file__).resolve().parent
    / "external_runtime"
    / GPAW_RUNTIME_PROVENANCE_FILENAME
)


def build_external_gpaw_environment(
    external_runtime_config: GpawExternalInstallPathsConfig,
    paths_to_prepend_to_pythonpath: Sequence[Path],
) -> dict[str, str]:
    """Build the environment that selects the configured GPAW PAW setups."""
    environment_variables = dict(os.environ)
    environment_variables[GPAW_SETUP_PATH_ENVIRONMENT_VARIABLE] = (
        external_runtime_config.gpaw_setups_dir.as_posix()
    )
    inherited_pythonpath = environment_variables.get("PYTHONPATH")
    pythonpath_entries = [path.as_posix() for path in paths_to_prepend_to_pythonpath]
    if inherited_pythonpath:
        pythonpath_entries.append(inherited_pythonpath)
    environment_variables["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return environment_variables


def build_external_gpaw_command_arguments(
    *,
    external_runtime_config: GpawExternalInstallPathsConfig,
    number_of_processes: int,
    script_path: Path,
    structure_path: Path,
    kwargs_path: Path,
    output_path: Path,
) -> list[str]:
    """Build the explicit MPI GPAW command for one external script invocation."""
    return [
        GPAW_MPI_LAUNCHER,
        GPAW_MPI_PROCESS_COUNT_FLAG,
        str(number_of_processes),
        external_runtime_config.gpaw_executable_path.as_posix(),
        GPAW_LAUNCH_PYTHON_SUBCOMMAND,
        script_path.as_posix(),
        structure_path.as_posix(),
        kwargs_path.as_posix(),
        output_path.as_posix(),
    ]


def prepare_external_gpaw_script(
    *,
    script_path: Path,
    script_source_path: Path,
    runtime_provenance_module_path: Path | None = None,
) -> None:
    """Stage an external GPAW script and optional runtime support module."""
    script_path.write_text(
        script_source_path.read_text(encoding=DEFAULT_TEXT_ENCODING),
        encoding=DEFAULT_TEXT_ENCODING,
    )
    if runtime_provenance_module_path is not None:
        runtime_provenance_module_path.write_text(
            _GPAW_RUNTIME_PROVENANCE_SOURCE_PATH.read_text(
                encoding=DEFAULT_TEXT_ENCODING
            ),
            encoding=DEFAULT_TEXT_ENCODING,
        )


def run_external_gpaw_script(
    *,
    external_runtime_config: GpawExternalInstallPathsConfig,
    number_of_processes: int,
    script_path: Path,
    structure_path: Path,
    kwargs_path: Path,
    output_path: Path,
    run_directory: Path,
    failure_message: str,
) -> Path:
    """Run an external GPAW script and return its JSON output path."""
    command_arguments = build_external_gpaw_command_arguments(
        external_runtime_config=external_runtime_config,
        number_of_processes=number_of_processes,
        script_path=script_path,
        structure_path=structure_path,
        kwargs_path=kwargs_path,
        output_path=output_path,
    )
    try:
        subprocess.run(
            command_arguments,
            cwd=run_directory,
            check=True,
            text=True,
            capture_output=True,
            env=build_external_gpaw_environment(
                external_runtime_config,
                paths_to_prepend_to_pythonpath=(script_path.parent,),
            ),
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise RuntimeError(failure_message) from exc
    return output_path
