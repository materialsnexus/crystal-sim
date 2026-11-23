"""Unit tests for MD schema models."""

from __future__ import annotations

import pytest

from crystal.schemas.md.settings import (
    MdDynamicsSettings,
    MdEnsemble,
    MdPotentialSettings,
    MdRunSettings,
    MdTimeSettings,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry


def test_md_dynamics_settings_raises_for_required_settings_for_ensembles() -> None:
    with pytest.raises(
        ValueError, match="Must provide valid settings for required settings"
    ):
        _ = MdRunSettings(
            time=MdTimeSettings(time_step_fs=1.0, simulation_time_ps=1.0),
            dynamics=MdDynamicsSettings(ensemble=MdEnsemble.nvt),
            potential=MdPotentialSettings(
                model=MlipModelRef(
                    registry_model=QetModelRegistry.foundational_matpes_pbe
                )
            ),
        )
