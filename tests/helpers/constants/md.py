"""Unit tests for MLIP MD equilibration schemas."""

from __future__ import annotations

from crystal.schemas.md.properties import (
    MdEquilibrationPlots,
    MdEquilibrationProperties,
    MdStageResult,
)
from crystal.schemas.md.settings import (
    MdDynamicsSettings,
    MdEnsemble,
    MdPotentialSettings,
    MdRunSettings,
    MdTimeSettings,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from tests.helpers.test_data_paths import TESTS_DATA_DIR

TEST_TRAJECTORY_PATH = TESTS_DATA_DIR / "md" / "test_trajectory.traj"

MOCK_MD_RUN_SETTINGS = MdRunSettings(
    time=MdTimeSettings(time_step_fs=2.0, simulation_time_ps=0.01),
    dynamics=MdDynamicsSettings(ensemble=MdEnsemble.nve),
    potential=MdPotentialSettings(
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    ),
)

MOCK_MD_PROPERTIES = MdEquilibrationProperties(
    ensemble=MdEnsemble.nve,
    md_stage_result=MdStageResult(
        ensemble=MdEnsemble.nve,
        trajectory_path=TEST_TRAJECTORY_PATH,
        default_log_path=TEST_TRAJECTORY_PATH.parent / "nve_default.log",
        user_defined_log_path=TEST_TRAJECTORY_PATH.parent / "nve_user_defined.log",
    ),
    check_results=[],
    time_averaging_window=0.2,
    plots=MdEquilibrationPlots(),
)

MOCK_DEFAULT_LOG_CONTENT = (
    "Time[ps] Etot[eV] Epot[eV] Ekin[eV] T[K]\n"
    "0.000 -10.000 -11.000 1.000 300.0\n"
    "0.100 -9.800 -10.800 1.000 305.0\n"
    "0.200 -9.700 -10.700 1.000 302.0\n"
)
MOCK_USER_DEFINED_LOG_CONTENT = (
    "time_ps, density, max_force_component_ev_per_angstrom\n"
    "0.000000, 2.500000, 0.100000\n"
    "0.100000, 2.450000, 0.090000\n"
    "0.200000, 2.470000, 0.080000\n"
)
