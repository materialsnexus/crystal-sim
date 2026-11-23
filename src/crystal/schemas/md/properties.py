# noqa: D100
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import PositiveFloat
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.schemas.base_properties import Properties
from crystal.schemas.md.data import DefaultMdLogData, MdSeries, UserDefinedMdLogData
from crystal.schemas.md.settings import MdEnsemble
from crystal.schemas.md.utils import (
    compute_running_average_series,
    parse_default_md_log,
    parse_user_defined_md_log,
    read_last_trajectory_frame,
)
from crystal.schemas.serialization import MSONableBaseModel


class MdStageResult(MSONableBaseModel):  # noqa: D101
    ensemble: MdEnsemble
    trajectory_path: Path
    default_log_path: Path
    user_defined_log_path: Path

    @cached_property
    def default_log_data(self) -> DefaultMdLogData:  # noqa: D102
        return parse_default_md_log(self.default_log_path)

    @cached_property
    def user_defined_log_data(self) -> UserDefinedMdLogData:  # noqa: D102
        return parse_user_defined_md_log(self.user_defined_log_path)

    @property
    def energy_series(self) -> MdSeries:  # noqa: D102
        return MdSeries(
            time_ps=self.default_log_data.time_ps,
            values=self.default_log_data.total_energy_ev,
        )

    @property
    def temperature_series(self) -> MdSeries:  # noqa: D102
        return MdSeries(
            time_ps=self.default_log_data.time_ps,
            values=self.default_log_data.temperature_k,
        )

    @property
    def density_series(self) -> MdSeries:  # noqa: D102
        return MdSeries(
            time_ps=self.user_defined_log_data.time_ps,
            values=self.user_defined_log_data.density,
        )


class MdCheckResult(MSONableBaseModel):  # noqa: D101
    check_type: Literal["AverageEnergyDriftCheck"]
    passed: bool
    threshold: float
    observed_value: float


class MdEquilibrationPlots(MSONableBaseModel):  # noqa: D101
    energy_plot_path: str | None = None
    temperature_plot_path: str | None = None
    density_plot_path: str | None = None


class MdEquilibrationProperties(Properties):  # noqa: D101
    type: Literal["MdEquilibrationProperties"] = "MdEquilibrationProperties"
    ensemble: MdEnsemble
    md_stage_result: MdStageResult
    check_results: list[MdCheckResult]
    time_averaging_window: PositiveFloat
    plots: MdEquilibrationPlots

    @property
    def all_checks_passed(self) -> bool:  # noqa: D102
        return all(check.passed for check in self.check_results)

    @property
    def average_energy(self) -> float:  # noqa: D102
        averaged_energy_series = compute_running_average_series(
            series=self.md_stage_result.energy_series,
            time_averaging_window=float(self.time_averaging_window),
        )
        return averaged_energy_series.statistics.mean

    @property
    def last_frame_structure(self) -> PMG_Structure:  # noqa: D102
        last_ase_structure = read_last_trajectory_frame(
            trajectory_path=self.md_stage_result.trajectory_path
        )
        return AseAtomsAdaptor.get_structure(last_ase_structure)
