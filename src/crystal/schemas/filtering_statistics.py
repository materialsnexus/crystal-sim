"""Statistics models for structure filtering stages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator


class CohesiveEnergyFilteringStatistics(BaseModel):
    """Counts produced by one cohesive-energy filtering stage."""

    model_config = ConfigDict(extra="forbid")

    considered: NonNegativeInt = 0
    passed: NonNegativeInt = 0
    filtered: NonNegativeInt = 0
    failed: NonNegativeInt = 0

    @model_validator(mode="after")
    def _validate_count_partition(self) -> CohesiveEnergyFilteringStatistics:
        if self.passed + self.filtered + self.failed != self.considered:
            msg = "passed plus filtered plus failed must equal considered."
            raise ValueError(msg)
        return self
