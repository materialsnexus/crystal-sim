"""Runtime profile defaults for FireWorks runtime configuration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, FilePath

from crystal import (
    LOCAL_FWORKER_YAML,
    LOCAL_JOBSTORE_YAML,
    LOCAL_LAUNCHPAD_YAML,
    SLURM_FWORKER_YAML,
    SLURM_JOBSTORE_YAML,
    SLURM_LAUNCHPAD_YAML,
    SLURM_QADAPTER_YAML,
)


class FireWorksRuntimeProfile(StrEnum):  # noqa: D101
    LOCAL = "local"
    SLURM = "slurm"


class FireWorksRuntimeProfilePaths(BaseModel):  # noqa: D101
    model_config = ConfigDict(frozen=True)

    launchpad_path: FilePath
    jobstore_path: FilePath
    fworker_path: FilePath
    qadapter_template_path: FilePath | None = None


FIREWORKS_RUNTIME_PROFILE_DEFAULT_PATHS: dict[
    FireWorksRuntimeProfile, FireWorksRuntimeProfilePaths
] = {
    FireWorksRuntimeProfile.LOCAL: FireWorksRuntimeProfilePaths(
        launchpad_path=LOCAL_LAUNCHPAD_YAML,
        jobstore_path=LOCAL_JOBSTORE_YAML,
        fworker_path=LOCAL_FWORKER_YAML,
        qadapter_template_path=None,
    ),
    FireWorksRuntimeProfile.SLURM: FireWorksRuntimeProfilePaths(
        launchpad_path=SLURM_LAUNCHPAD_YAML,
        jobstore_path=SLURM_JOBSTORE_YAML,
        fworker_path=SLURM_FWORKER_YAML,
        qadapter_template_path=SLURM_QADAPTER_YAML,
    ),
}


def resolve_fireworks_runtime_profile_paths(
    profile: FireWorksRuntimeProfile,
) -> FireWorksRuntimeProfilePaths:
    """Resolve profile-backed default paths for FireWorks runtime usage."""
    profile_paths = FIREWORKS_RUNTIME_PROFILE_DEFAULT_PATHS.get(profile)
    if profile_paths is not None:
        return profile_paths
    error_message = f"Unsupported runtime profile {profile!s}"
    raise ValueError(error_message)
