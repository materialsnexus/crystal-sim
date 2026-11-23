from __future__ import annotations

from pathlib import Path

from crystal.helpers.fireworks.runtime_config import FireWorksRuntimePaths
from crystal.schemas.fireworks_runtime import (
    FireWorksRuntimeProfile,
    FireWorksRuntimeProfilePaths,
)
from tests.helpers.constants.fireworks_runtime import (
    TEST_RUNTIME_FW_CONFIG_FILENAME,
    TEST_RUNTIME_FWORKER_FILENAME,
    TEST_RUNTIME_JOBSTORE_FILENAME,
    TEST_RUNTIME_LAUNCHPAD_FILENAME,
    TEST_RUNTIME_QADAPTER_TEMPLATE_FILENAME,
    TEST_RUNTIME_QLAUNCH_STUB_FILENAME,
    TEST_RUNTIME_QLAUNCH_STUB_SCRIPT,
    TEST_RUNTIME_QLAUNCH_STUB_SLEEP_SECONDS,
)


def create_fireworks_runtime_fixture_file(path: Path, content: str = "test\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def build_fireworks_runtime_paths_fixture_for_profile(
    tmp_path: Path,
    profile: FireWorksRuntimeProfile = FireWorksRuntimeProfile.SLURM,
    mpi_environment_setup_script_path: Path | None = None,
) -> FireWorksRuntimePaths:
    runtime_paths = FireWorksRuntimeProfilePaths(
        launchpad_path=create_fireworks_runtime_fixture_file(
            tmp_path / TEST_RUNTIME_LAUNCHPAD_FILENAME
        ),
        jobstore_path=create_fireworks_runtime_fixture_file(
            tmp_path / TEST_RUNTIME_JOBSTORE_FILENAME
        ),
        fworker_path=create_fireworks_runtime_fixture_file(
            tmp_path / TEST_RUNTIME_FWORKER_FILENAME
        ),
        qadapter_template_path=(
            create_fireworks_runtime_fixture_file(
                tmp_path / TEST_RUNTIME_QADAPTER_TEMPLATE_FILENAME
            )
            if profile is not FireWorksRuntimeProfile.LOCAL
            else None
        ),
    )

    fw_config_path = create_fireworks_runtime_fixture_file(
        tmp_path / TEST_RUNTIME_FW_CONFIG_FILENAME
    )

    return FireWorksRuntimePaths.from_paths(
        profile=profile,
        profile_paths=runtime_paths,
        fw_config_path=fw_config_path,
        mpi_environment_setup_script_path=mpi_environment_setup_script_path,
    )


def create_fireworks_runtime_launch_stub_executable(
    binary_directory: Path,
    executable_name: str = TEST_RUNTIME_QLAUNCH_STUB_FILENAME,
    sleep_seconds: int = TEST_RUNTIME_QLAUNCH_STUB_SLEEP_SECONDS,
) -> Path:
    binary_directory.mkdir(parents=True, exist_ok=True)
    qlaunch_stub_path = binary_directory / executable_name
    script_content = TEST_RUNTIME_QLAUNCH_STUB_SCRIPT.format(
        sleep_seconds=sleep_seconds
    )
    qlaunch_stub_path.write_text(script_content, encoding="utf-8")
    qlaunch_stub_path.chmod(0o755)
    return qlaunch_stub_path
