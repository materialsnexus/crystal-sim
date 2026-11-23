# noqa: D100
from __future__ import annotations

import shlex
from pathlib import Path

import yaml
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from fireworks.utilities.fw_serializers import load_object_from_file

from crystal.constants.fireworks_runtime import (
    RUNTIME_BASH_BINARY,
    RUNTIME_BASH_LOGIN_COMMAND_FLAG,
    RUNTIME_QADAPTER_PRE_ROCKET_KEY,
    RUNTIME_SOURCE_SHELL_COMMAND,
)


def ensure_qadapter_log_directory(qadapter_path: Path) -> None:
    """Create the optional queue-adapter log directory before queue launch."""
    queue_adapter: CommonAdapter = load_object_from_file(qadapter_path.as_posix())
    qadapter_log_directory = queue_adapter.get("logdir")
    if qadapter_log_directory:
        Path(str(qadapter_log_directory)).mkdir(parents=True, exist_ok=True)


def write_mpi_environment_qadapter(
    qadapter_template_path: Path,
    mpi_environment_setup_script_path: Path,
    output_path: Path,
) -> Path:
    """Write a queue adapter that sources the configured MPI environment."""
    queue_adapter: CommonAdapter = load_object_from_file(
        qadapter_template_path.as_posix()
    )
    existing_pre_rocket = queue_adapter.get(RUNTIME_QADAPTER_PRE_ROCKET_KEY)
    pre_rocket_commands = [
        build_mpi_environment_setup_command(mpi_environment_setup_script_path)
    ]
    if existing_pre_rocket:
        pre_rocket_commands.append(str(existing_pre_rocket))
    queue_adapter[RUNTIME_QADAPTER_PRE_ROCKET_KEY] = "\n".join(pre_rocket_commands)
    output_path.write_text(
        yaml.safe_dump(queue_adapter.to_dict(), sort_keys=False), encoding="utf-8"
    )
    return output_path


def build_mpi_environment_setup_command(mpi_environment_setup_script_path: Path) -> str:
    """Build the shell command that sources the configured MPI environment."""
    return (
        f"{RUNTIME_SOURCE_SHELL_COMMAND} "
        f"{shlex.quote(mpi_environment_setup_script_path.as_posix())}"
    )


def build_local_launch_command_with_mpi_environment(
    launch_arguments: list[str], mpi_environment_setup_script_path: Path
) -> list[str]:
    """Prefix a local FireWorks launch with the configured MPI environment."""
    shell_command = (
        f"{build_mpi_environment_setup_command(mpi_environment_setup_script_path)} "
        f"&& exec {shlex.join(launch_arguments)}"
    )
    return [RUNTIME_BASH_BINARY, RUNTIME_BASH_LOGIN_COMMAND_FLAG, shell_command]
