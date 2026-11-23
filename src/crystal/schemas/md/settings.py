"""Serializable schema models for MD stability workflows."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt, model_validator

from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.serialization import MSONableBaseModel

DEFAULT_ASE_STRESS_UNITS = "eV/A3"


class MdEnsemble(StrEnum):  # noqa: D101
    nvt = "nvt"
    nve = "nve"
    npt = "npt"


class MdTimeSettings(MSONableBaseModel):  # noqa: D101
    time_step_fs: PositiveFloat
    simulation_time_ps: PositiveFloat
    trajectory_interval: PositiveInt = 1
    property_log_interval: PositiveInt = 1


class MdThermostatSettings(MSONableBaseModel):  # noqa: D101
    temperature_k: NonNegativeFloat
    time_constant_fs: PositiveFloat


class MdBarostatSettings(MSONableBaseModel):  # noqa: D101
    pressure_gpa: float
    temperature_damping_fs: PositiveFloat
    pressure_damping_fs: PositiveFloat


class MdDynamicsSettings(MSONableBaseModel):  # noqa: D101
    ensemble: MdEnsemble
    thermostat: MdThermostatSettings | None = None
    barostat: MdBarostatSettings | None = None

    @model_validator(mode="after")
    def _validate_relevant_settings_are_provided(self) -> Self:
        required_settings_by_ensemble: dict[
            MdEnsemble, dict[str, MSONableBaseModel | None]
        ] = {
            MdEnsemble.npt: {"barostat": self.barostat, "thermostat": self.thermostat},
            MdEnsemble.nvt: {"thermostat": self.thermostat},
        }
        required_settings = required_settings_by_ensemble.get(self.ensemble, {})
        missing_settings = [
            setting_name
            for setting_name, setting_value in required_settings.items()
            if setting_value is None
        ]
        if missing_settings:
            error_msg = (
                "Must provide valid settings for required settings: "
                f"{', '.join(missing_settings)}."
            )
            raise ValueError(error_msg)
        return self


class MdPotentialSettings(MSONableBaseModel):  # noqa: D101
    model: MlipModelRef
    stress_unit: str = DEFAULT_ASE_STRESS_UNITS


class MdRunSettings(MSONableBaseModel):  # noqa: D101
    time: MdTimeSettings
    dynamics: MdDynamicsSettings
    potential: MdPotentialSettings
    pressure_mask: tuple[bool, bool, bool] = (True, True, False)
    pbc: tuple[bool, bool, bool] = (True, True, True)
    output_directory: Path | None = None
