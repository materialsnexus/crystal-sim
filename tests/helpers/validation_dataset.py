"""Shared fixtures for validation-dataset behavior tests."""

from __future__ import annotations

from pathlib import Path

from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.schemas.md.mlip_md_equilibration import (
    MlipMdEquilibrationSimulationSettings,
)
from crystal.schemas.md.settings import (
    MdDynamicsSettings,
    MdEnsemble,
    MdPotentialSettings,
    MdRunSettings,
    MdThermostatSettings,
    MdTimeSettings,
)
from crystal.schemas.mlip_model_ref import MlipModelRef
from tests.helpers.constants.structure import TEST_SI_STRUCTURE


def build_validation_dataset_md_simulation_settings_for_test() -> (
    MlipMdEquilibrationSimulationSettings
):
    """Build the short NVT settings shared by MLIP-MD tests."""
    return MlipMdEquilibrationSimulationSettings(
        run_settings=MdRunSettings(
            time=MdTimeSettings(time_step_fs=1.0, simulation_time_ps=0.01),
            dynamics=MdDynamicsSettings(
                ensemble=MdEnsemble.nvt,
                thermostat=MdThermostatSettings(
                    temperature_k=300.0, time_constant_fs=10.0
                ),
            ),
            potential=MdPotentialSettings(model=MlipModelRef(path=Path("tests/data"))),
        ),
        time_averaging_window=0.001,
    )


def write_validation_dataset_md_trajectory_for_test(
    path: Path, number_of_frames: int = 2
) -> None:
    """Write a deterministic trajectory for MLIP-MD artifact tests."""
    atoms = AseAtomsAdaptor().get_atoms(TEST_SI_STRUCTURE)
    with Trajectory(path, mode="w") as trajectory:  # type: ignore[no-untyped-call]
        for _ in range(number_of_frames):
            trajectory.write(atoms)


def build_validation_dataset_md_trajectory_bytes_for_test(
    number_of_frames: int,
) -> bytes:
    """Return deterministic trajectory bytes for frame-pool tests."""
    path = Path("source.traj")
    write_validation_dataset_md_trajectory_for_test(
        path=path, number_of_frames=number_of_frames
    )
    return path.read_bytes()
