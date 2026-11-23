# noqa: D100
from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, NonNegativeFloat

from crystal.schemas.md.properties import MdStageResult
from crystal.schemas.md.utils import compute_running_average_series
from crystal.schemas.serialization import MSONableBaseModel


class BaseMdEquilibrationCheck(MSONableBaseModel, ABC):  # noqa: D101
    check_type: str

    @property
    @abstractmethod
    def threshold(self) -> float:
        """Get the threshold value for this check."""
        error_msg = (
            "threshold must be implemented in subclasses of BaseMdEquilibrationCheck."
        )
        raise NotImplementedError(error_msg)

    @abstractmethod
    def get_comparison_value(
        self, md_stage_result: MdStageResult, time_averaging_window: float
    ) -> float:
        """Get the value to compare against the check threshold."""
        error_msg = "get_comparison_value must be implemented in subclasses of BaseMdEquilibrationCheck."
        raise NotImplementedError(error_msg)


class AverageEnergyDriftCheck(BaseMdEquilibrationCheck):  # noqa: D101
    check_type: Literal["AverageEnergyDriftCheck"] = "AverageEnergyDriftCheck"
    max_absolute_drift_slope: NonNegativeFloat

    def get_comparison_value(  # noqa: D102
        self, md_stage_result: MdStageResult, time_averaging_window: float
    ) -> float:
        averaged_energy_series = compute_running_average_series(
            series=md_stage_result.energy_series,
            time_averaging_window=time_averaging_window,
        )
        return abs(averaged_energy_series.statistics.drift_slope)

    @property
    def threshold(self) -> float:  # noqa: D102
        return self.max_absolute_drift_slope


MdEquilibrationCheck: TypeAlias = Annotated[
    AverageEnergyDriftCheck, Field(discriminator="check_type")
]
