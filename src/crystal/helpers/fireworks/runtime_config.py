# noqa: D100
from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from crystal import CRYSTAL_CONFIG, CRYSTAL_ROOT, CrystalConfig
from crystal.constants.fireworks_runtime import (
    FW_CONFIG_ENVIRONMENT_VARIABLE_NAME,
    RUNTIME_FW_CONFIG_FILENAME,
    RUNTIME_LOCAL_LAUNCH_BINARY,
    RUNTIME_QUEUE_LAUNCH_BINARY,
    FireWorksRuntimeCommandFlag,
)
from crystal.helpers.fireworks.connections import validate_config_file_exists
from crystal.schemas.fireworks_runtime import (
    FireWorksRuntimeProfile,
    FireWorksRuntimeProfilePaths,
    resolve_fireworks_runtime_profile_paths,
)


class FireWorksRuntimePathsError(RuntimeError):  # noqa: D101
    pass


class FireWorksRuntimePaths(BaseModel):  # noqa: D101
    model_config = ConfigDict(frozen=True)
    profile_paths: FireWorksRuntimeProfilePaths
    fw_config_path: Path
    mpi_environment_setup_script_path: Path | None = None

    @model_validator(mode="after")
    def _validate_required_static_files(self) -> Self:
        for config_file in (
            self.profile_paths.launchpad_path,
            self.profile_paths.jobstore_path,
            self.profile_paths.fworker_path,
            self.fw_config_path,
        ):
            validate_config_file_exists(config_file)
        if self.mpi_environment_setup_script_path is not None:
            validate_config_file_exists(self.mpi_environment_setup_script_path)
        _ = self._validated_qadapter_template_path()
        return self

    @classmethod
    def from_profile(
        cls,
        profile: FireWorksRuntimeProfile = FireWorksRuntimeProfile.LOCAL,
        crystal_config: CrystalConfig | None = None,
    ) -> FireWorksRuntimePaths:
        """Build runtime config from standard repository profile paths."""
        resolved_crystal_config = (
            crystal_config if crystal_config is not None else CRYSTAL_CONFIG
        )
        try:
            profile_defaults = resolve_fireworks_runtime_profile_paths(profile)
        except ValueError as error:
            raise FireWorksRuntimePathsError(str(error)) from error
        mpi_environment_setup_script_path = (
            resolved_crystal_config.programs.mpi_environment_setup_script
        )
        return cls.from_paths(
            profile=profile,
            profile_paths=profile_defaults,
            fw_config_path=CRYSTAL_ROOT / RUNTIME_FW_CONFIG_FILENAME,
            mpi_environment_setup_script_path=mpi_environment_setup_script_path,
        )

    @classmethod
    def from_paths(
        cls,
        profile: FireWorksRuntimeProfile,
        profile_paths: FireWorksRuntimeProfilePaths,
        fw_config_path: Path,
        mpi_environment_setup_script_path: Path | None = None,
    ) -> FireWorksRuntimePaths:
        """Build runtime config from explicit config file paths."""
        run_paths = FireWorksRuntimeProfilePaths(
            launchpad_path=profile_paths.launchpad_path.resolve(),
            jobstore_path=profile_paths.jobstore_path.resolve(),
            fworker_path=profile_paths.fworker_path.resolve(),
            qadapter_template_path=(
                profile_paths.qadapter_template_path.resolve()
                if profile_paths.qadapter_template_path is not None
                else None
            ),
        )
        return cls(
            profile=profile,
            profile_paths=run_paths,
            fw_config_path=fw_config_path.resolve(),
            mpi_environment_setup_script_path=(
                mpi_environment_setup_script_path.resolve()
                if mpi_environment_setup_script_path is not None
                else None
            ),
        )

    def _validated_qadapter_template_path(self) -> Path | None:
        qadapter_template_path = self.profile_paths.qadapter_template_path
        if qadapter_template_path is None:
            return None
        validate_config_file_exists(qadapter_template_path)
        return qadapter_template_path

    def environment(self) -> dict[str, str]:  # noqa: D102
        fw_config_path = self.fw_config_path
        return {FW_CONFIG_ENVIRONMENT_VARIABLE_NAME: fw_config_path.as_posix()}

    @property
    def is_queue_launch(self) -> bool:  # noqa: D102
        return self.profile_paths.qadapter_template_path is not None

    @property
    def launch_binary(self) -> str:  # noqa: D102
        return (
            RUNTIME_QUEUE_LAUNCH_BINARY
            if self.is_queue_launch
            else RUNTIME_LOCAL_LAUNCH_BINARY
        )

    def launch_args(
        self,
        mode: str = "rapidfire",
        nlaunches: int | str = "infinite",
        qadapter_override_path: Path | None = None,
    ) -> list[str]:
        """Build subprocess-safe launch CLI arguments for the runtime profile."""
        arguments = [self.launch_binary]
        if self.is_queue_launch:
            arguments.append(FireWorksRuntimeCommandFlag.RESERVE)
        arguments.extend(
            [
                FireWorksRuntimeCommandFlag.FWORKER,
                self.profile_paths.fworker_path.as_posix(),
                FireWorksRuntimeCommandFlag.LAUNCHPAD,
                self.profile_paths.launchpad_path.as_posix(),
            ]
        )
        resolved_qadapter_path = (
            qadapter_override_path
            if qadapter_override_path is not None
            else self.profile_paths.qadapter_template_path
        )
        if resolved_qadapter_path is not None:
            arguments.extend(
                [
                    FireWorksRuntimeCommandFlag.QADAPTER,
                    resolved_qadapter_path.as_posix(),
                ]
            )
        arguments.extend([mode, FireWorksRuntimeCommandFlag.NLAUNCHES, str(nlaunches)])
        return arguments
