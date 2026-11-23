"""Unit tests for MLIP MD equilibration schemas."""

from __future__ import annotations

import pytest

from crystal.schemas.md.checks import AverageEnergyDriftCheck
from crystal.schemas.md.data import MdSeries
from crystal.schemas.md.mlip_md_equilibration import (
    MlipMdEquilibrationSimulationSettings,
)
from crystal.schemas.md.properties import MdCheckResult
from tests.helpers.constants.md import MOCK_MD_PROPERTIES, MOCK_MD_RUN_SETTINGS


def test_mlip_md_equilibration_simulation_settings_accepts_average_energy_check() -> (
    None
):
    simulation_settings = MlipMdEquilibrationSimulationSettings(
        run_settings=MOCK_MD_RUN_SETTINGS,
        checks=[AverageEnergyDriftCheck(max_absolute_drift_slope=0.2)],
        time_averaging_window=0.003,
    )

    assert len(simulation_settings.checks) == 1
    assert simulation_settings.time_averaging_window == pytest.approx(0.003)


def test_md_series_statistics_computes_average_and_end_value() -> None:
    series = MdSeries(time_ps=[0.1, 0.2, 0.3], values=[-5.2, -5.0, -4.8])

    assert series.statistics.mean == pytest.approx(-5.0)
    assert series.statistics.end_value == pytest.approx(-4.8)


def test_md_series_get_final_window_energy_series_selects_window_samples() -> None:
    series = MdSeries(time_ps=[0.0, 0.5, 1.0, 1.5], values=[-5.3, -5.2, -5.1, -5.0])

    final_window_series = series.get_final_window_series(final_window_ps=0.5)

    assert final_window_series == MdSeries(time_ps=[1.0, 1.5], values=[-5.1, -5.0])


@pytest.mark.parametrize("final_window_ps", [0.0, -0.1])
def test_md_series_get_final_window_energy_series_rejects_non_positive_window(
    final_window_ps: float,
) -> None:
    series = MdSeries(time_ps=[0.0, 0.5], values=[-5.1, -5.0])

    with pytest.raises(ValueError, match="final_window_ps must be positive"):
        series.get_final_window_series(final_window_ps=final_window_ps)


def test_mlip_md_equilibration_properties_all_checks_passed_uses_results() -> None:
    properties = MOCK_MD_PROPERTIES.model_copy(
        update={
            "check_results": [
                MdCheckResult(
                    check_type="AverageEnergyDriftCheck",
                    passed=True,
                    threshold=0.2,
                    observed_value=0.1,
                )
            ]
        }
    )

    assert properties.all_checks_passed
