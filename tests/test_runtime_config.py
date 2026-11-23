from __future__ import annotations

from pathlib import Path

import pytest

from crystal import (
    LOCAL_FWORKER_YAML,
    LOCAL_JOBSTORE_YAML,
    LOCAL_LAUNCHPAD_YAML,
    SLURM_FWORKER_YAML,
    SLURM_JOBSTORE_YAML,
    SLURM_LAUNCHPAD_YAML,
    SLURM_QADAPTER_YAML,
    CrystalConfig,
)
from crystal.helpers.fireworks.runtime_config import FireWorksRuntimePaths
from crystal.schemas.fireworks_runtime import FireWorksRuntimeProfile


def test_crystal_config_resolves_mpi_environment_setup_script_path(
    tmp_path: Path,
) -> None:
    setup_script_path = tmp_path / "scripts" / "load-mpi.sh"
    setup_script_path.parent.mkdir()
    setup_script_path.write_text("#!/bin/bash\n", encoding="utf-8")
    toml_path = tmp_path / "crystal.toml"
    toml_path.write_text(
        "[programs]\nmpi_environment_setup_script = 'scripts/load-mpi.sh'\n",
        encoding="utf-8",
    )

    config = CrystalConfig.from_toml(toml_path)

    assert config.programs.mpi_environment_setup_script == setup_script_path.resolve()


@pytest.mark.parametrize(
    (
        "profile",
        "expected_launchpad",
        "expected_jobstore",
        "expected_fworker",
        "expected_qadapter_template_path",
    ),
    [
        (
            FireWorksRuntimeProfile.LOCAL,
            LOCAL_LAUNCHPAD_YAML,
            LOCAL_JOBSTORE_YAML,
            LOCAL_FWORKER_YAML,
            None,
        ),
        (
            FireWorksRuntimeProfile.SLURM,
            SLURM_LAUNCHPAD_YAML,
            SLURM_JOBSTORE_YAML,
            SLURM_FWORKER_YAML,
            SLURM_QADAPTER_YAML,
        ),
    ],
)
def test_runtime_paths_from_profile_resolves_repo_default_paths(
    profile: FireWorksRuntimeProfile,
    expected_launchpad: Path,
    expected_jobstore: Path,
    expected_fworker: Path,
    expected_qadapter_template_path: Path | None,
) -> None:
    runtime = FireWorksRuntimePaths.from_profile(
        profile=profile, crystal_config=CrystalConfig()
    )

    assert runtime.profile_paths.launchpad_path == expected_launchpad
    assert runtime.profile_paths.jobstore_path == expected_jobstore
    assert runtime.profile_paths.fworker_path == expected_fworker
    assert (
        runtime.profile_paths.qadapter_template_path == expected_qadapter_template_path
    )
