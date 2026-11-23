"""Tests for MLIP MD runner utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import ase
import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture
from matgl.ext.ase import Atoms2Graph
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from crystal.calculators.md.md_runner import MlipMdRunner
from crystal.helpers.md_plot_helpers import TrajectoryAnalyzer
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.md.data import MdSeries
from crystal.schemas.md.properties import MdStageResult
from crystal.schemas.md.settings import (
    MdBarostatSettings,
    MdDynamicsSettings,
    MdEnsemble,
    MdPotentialSettings,
    MdRunSettings,
    MdThermostatSettings,
    MdTimeSettings,
)
from crystal.schemas.md.utils import DEFAULT_USER_DEFINED_HEADERS
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from tests.helpers.constants.md import TEST_TRAJECTORY_PATH
from tests.helpers.constants.structure import TEST_BCC_FE_STRUCTURE, TEST_SI_STRUCTURE

TEST_THERMOSTAT_SETTINGS = MdThermostatSettings(
    temperature_k=300.0, time_constant_fs=2.0
)
TEST_BAROSTAT_SETTINGS = MdBarostatSettings(
    pressure_gpa=1e-3, temperature_damping_fs=2.0, pressure_damping_fs=4.0
)

DEFAULT_MD_POTENTIAL_SETTINGS = MdPotentialSettings(
    model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
)
DEFAULT_MOCK_MD_STAGE_RESULT = MdStageResult(
    ensemble=MdEnsemble.nve,
    trajectory_path=Path("nve.traj"),
    default_log_path=Path("nve_default.log"),
    user_defined_log_path=Path("nve_user_defined.log"),
    energy_series=MdSeries(time_ps=[0.0, 0.1], values=[-1.0, -0.9]),
    temperature_series=MdSeries(time_ps=[0.0, 0.1], values=[300.0, 301.0]),
    density_series=MdSeries(time_ps=[0.0, 0.1], values=[2.0, 2.1]),
)
DEFAULT_NVE_RUN_SETTINGS = MdRunSettings(
    time=MdTimeSettings(
        time_step_fs=2.0,
        simulation_time_ps=0.004,
        trajectory_interval=2,
        property_log_interval=2,
    ),
    dynamics=MdDynamicsSettings(ensemble=MdEnsemble.nve),
    potential=DEFAULT_MD_POTENTIAL_SETTINGS,
)
DEFAULT_NVT_RUN_SETTINGS = MdRunSettings(
    time=MdTimeSettings(time_step_fs=2.0, simulation_time_ps=0.004),
    dynamics=MdDynamicsSettings(
        ensemble=MdEnsemble.nvt, thermostat=TEST_THERMOSTAT_SETTINGS
    ),
    potential=DEFAULT_MD_POTENTIAL_SETTINGS,
)
DEFAULT_NPT_RUN_SETTINGS = MdRunSettings(
    time=MdTimeSettings(time_step_fs=2.0, simulation_time_ps=0.004),
    dynamics=MdDynamicsSettings(
        ensemble=MdEnsemble.npt,
        thermostat=TEST_THERMOSTAT_SETTINGS,
        barostat=TEST_BAROSTAT_SETTINGS,
    ),
    potential=DEFAULT_MD_POTENTIAL_SETTINGS,
)


@run_in_temp_dir
def test_mlip_md_runner_run_nve_uses_velocity_verlet() -> None:
    runner = MlipMdRunner(
        structure=deepcopy(TEST_BCC_FE_STRUCTURE), run_settings=DEFAULT_NVE_RUN_SETTINGS
    )

    with (
        patch(
            "crystal.calculators.md.md_runner.VelocityVerlet", return_value=Mock()
        ) as velocity_verlet,
        patch.object(
            runner,
            "_get_stage_result",
            return_value=DEFAULT_MOCK_MD_STAGE_RESULT.model_copy(
                update={"ensemble": MdEnsemble.nve}
            ),
        ),
    ):
        result = runner.run()
    velocity_verlet.assert_called()
    assert result.ensemble == MdEnsemble.nve


@run_in_temp_dir
def test_mlip_md_runner_run_nvt_uses_langevin() -> None:
    runner = MlipMdRunner(
        structure=deepcopy(TEST_BCC_FE_STRUCTURE), run_settings=DEFAULT_NVT_RUN_SETTINGS
    )
    with (
        patch(
            "crystal.calculators.md.md_runner.Langevin", return_value=Mock()
        ) as nvt_langevin,
        patch.object(
            runner,
            "_get_stage_result",
            return_value=DEFAULT_MOCK_MD_STAGE_RESULT.model_copy(
                update={"ensemble": MdEnsemble.nvt}
            ),
        ),
    ):
        result = runner.run()

    nvt_langevin.assert_called()
    assert result.ensemble == MdEnsemble.nvt


@run_in_temp_dir
def test_mlip_md_runner_run_npt_uses_masked_mtknpt() -> None:
    runner = MlipMdRunner(
        structure=deepcopy(TEST_BCC_FE_STRUCTURE), run_settings=DEFAULT_NPT_RUN_SETTINGS
    )
    with (
        patch(
            "crystal.calculators.md.md_runner.MaskedMTKNPT", return_value=Mock()
        ) as masked_mtknpt,
        patch.object(
            runner,
            "_get_stage_result",
            return_value=DEFAULT_MOCK_MD_STAGE_RESULT.model_copy(
                update={"ensemble": MdEnsemble.npt}
            ),
        ),
    ):
        result = runner.run()

    masked_mtknpt.assert_called()
    assert result.ensemble == MdEnsemble.npt


@run_in_temp_dir
def test_mlip_md_runner_applies_pressure_mask() -> None:
    run_settings = MdRunSettings(
        time=MdTimeSettings(time_step_fs=1.0, simulation_time_ps=0.002),
        dynamics=MdDynamicsSettings(
            ensemble=MdEnsemble.npt,
            thermostat=MdThermostatSettings(temperature_k=300.0, time_constant_fs=10.0),
            barostat=MdBarostatSettings(
                pressure_gpa=1e-3, temperature_damping_fs=10.0, pressure_damping_fs=20.0
            ),
        ),
        potential=DEFAULT_MD_POTENTIAL_SETTINGS,
        pressure_mask=(True, True, False),
        output_directory=Path.cwd(),
    )
    runner = MlipMdRunner(
        structure=deepcopy(TEST_SI_STRUCTURE), run_settings=run_settings
    )
    initial_cell = TEST_SI_STRUCTURE.lattice.matrix.copy()
    runner.run_npt()
    final_cell = (
        TrajectoryAnalyzer(trajectory_path=Path.cwd() / "npt.traj")
        .frames[-1]
        .cell.array
    )

    assert not np.allclose(final_cell[0], initial_cell[0])
    assert not np.allclose(final_cell[1], initial_cell[1])
    assert final_cell[2] == pytest.approx(initial_cell[2])


@run_in_temp_dir
def test_mlip_md_runner_uses_restart_trajectory_when_provided() -> None:
    run_settings = MdRunSettings(
        time=MdTimeSettings(time_step_fs=1.0, simulation_time_ps=0.002),
        dynamics=MdDynamicsSettings(ensemble=MdEnsemble.nve),
        potential=DEFAULT_MD_POTENTIAL_SETTINGS,
        output_directory=Path.cwd(),
    )
    runner = MlipMdRunner(
        run_settings=run_settings, trajectory_path_for_restart=TEST_TRAJECTORY_PATH
    )
    assert isinstance(runner._ase_atoms, ase.Atoms)
    assert runner._ase_atoms.get_chemical_formula(empirical=True) == "LiNbO3"  # type:ignore[no-untyped-call]


@run_in_temp_dir
def test_mlip_md_runner_skips_zero_rotation_for_degenerate_inertia(
    caplog: LogCaptureFixture,
) -> None:
    runner = MlipMdRunner(
        structure=deepcopy(TEST_BCC_FE_STRUCTURE), run_settings=DEFAULT_NVT_RUN_SETTINGS
    )

    with (
        caplog.at_level("WARNING"),
        patch.object(runner, "_attach_calculator"),
        patch("crystal.calculators.md.md_runner.ZeroRotation") as zero_rotation,
    ):
        runner._prepare_atoms(initialize_temperature=False)

    zero_rotation.assert_not_called()
    assert (
        "Skipping ZeroRotation because the structure has degenerate "
        "principal moments of inertia."
    ) in caplog.text


@run_in_temp_dir
def test_mlip_md_runner_normalizes_partial_pbc_before_attaching_calculator() -> None:
    run_settings = DEFAULT_NVE_RUN_SETTINGS.model_copy(
        update={"pbc": (True, True, False)}
    )
    runner = MlipMdRunner(
        structure=Structure(
            lattice=Lattice(matrix=np.diag([4.0, 10.0, 12.0])),
            species=["Si", "Si"],
            coords=[[0.0, 0.0, 5.0], [3.5, 0.0, 5.0]],
            coords_are_cartesian=True,
        ),
        run_settings=run_settings,
    )
    calculator = Mock()

    with patch.object(
        MlipMdRunner, "_calculator", new_callable=PropertyMock, return_value=calculator
    ):
        atoms = runner._prepare_atoms(initialize_temperature=False)

    matgl_graph_converter = Atoms2Graph(element_types=("Si",), cutoff=1.0)
    graph_from_matgl_ready_atoms, _, _ = matgl_graph_converter.get_graph(atoms)
    partial_pbc_atoms = atoms.copy()  # type: ignore[no-untyped-call]
    partial_pbc_atoms.pbc = (True, True, False)
    graph_from_partial_pbc_atoms, _, _ = matgl_graph_converter.get_graph(
        partial_pbc_atoms
    )

    assert graph_from_partial_pbc_atoms.edge_index.shape[1] == 0
    assert graph_from_matgl_ready_atoms.edge_index.shape[1] == 2


@run_in_temp_dir
def test_mlip_md_nvt_runner() -> None:
    runner = MlipMdRunner(
        structure=deepcopy(TEST_BCC_FE_STRUCTURE), run_settings=DEFAULT_NVT_RUN_SETTINGS
    )
    results = runner.run()
    assert results.ensemble == MdEnsemble.nvt
    assert results.trajectory_path == Path.cwd() / "nvt.traj"
    assert results.default_log_path == Path.cwd() / "nvt_default.log"
    assert results.user_defined_log_path == Path.cwd() / "nvt_user_defined.log"
    assert len(results.energy_series.values) >= 2
    assert len(results.temperature_series.values) >= 2
    assert len(results.density_series.values) >= 2
    with Path(results.user_defined_log_path).open("r") as f:
        lines = f.read().strip().split("\n")
    header_list = lines[0].split(", ")
    assert header_list == DEFAULT_USER_DEFINED_HEADERS
    analyzer = TrajectoryAnalyzer(trajectory_path=results.trajectory_path)
    assert len(analyzer.frames) == 3


@run_in_temp_dir
def test_mlip_md_runner_appends_to_trajectory() -> None:
    runner = MlipMdRunner(
        structure=deepcopy(TEST_BCC_FE_STRUCTURE), run_settings=DEFAULT_NVT_RUN_SETTINGS
    )
    results = runner.run()
    assert results.ensemble == MdEnsemble.nvt
    assert results.trajectory_path == Path.cwd() / "nvt.traj"
    assert results.default_log_path == Path.cwd() / "nvt_default.log"
    assert results.user_defined_log_path == Path.cwd() / "nvt_user_defined.log"
    analyzer = TrajectoryAnalyzer(trajectory_path=results.trajectory_path)
    assert len(analyzer.frames) == 3

    restart_runner = MlipMdRunner(
        run_settings=DEFAULT_NVT_RUN_SETTINGS,
        trajectory_path_for_restart=results.trajectory_path,
    )
    restart_results = restart_runner.run()
    restart_analyzer = TrajectoryAnalyzer(
        trajectory_path=restart_results.trajectory_path
    )
    assert len(restart_analyzer.frames) == 6
