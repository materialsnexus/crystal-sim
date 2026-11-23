"""Run MLIP molecular dynamics on slabs and interfaces for validation."""

from __future__ import annotations

from ase.io.trajectory import Trajectory
from jobflow.core.job import job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.md.md_runner import MlipMdRunner
from crystal.constants.validation_dataset import (
    VALIDATION_DATASET_TEMPERATURE_SAMPLING_REQUIRES_THERMOSTAT_ERROR,
    VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME,
)
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.helpers.mlip_helpers import configure_default_torch_device
from crystal.schemas.md.mlip_md_equilibration import (
    MlipMdEquilibrationSimulationSettings,
)
from crystal.schemas.md.settings import MdEnsemble
from crystal.schemas.validation_dataset import (
    ValidationDatasetMdProperties,
    ValidationDatasetMdRunSpec,
    ValidationDatasetMdTaskDocument,
)


def build_validation_dataset_md_settings_at_temperature(
    simulation_settings: MlipMdEquilibrationSimulationSettings, temperature_k: float
) -> MlipMdEquilibrationSimulationSettings:
    """Set the thermostat temperature for one validation-dataset run."""
    dynamics = simulation_settings.run_settings.dynamics
    if dynamics.ensemble is MdEnsemble.nve or dynamics.thermostat is None:
        raise ValueError(
            VALIDATION_DATASET_TEMPERATURE_SAMPLING_REQUIRES_THERMOSTAT_ERROR
        )
    thermostat = dynamics.thermostat.model_copy(update={"temperature_k": temperature_k})
    updated_dynamics = dynamics.model_copy(update={"thermostat": thermostat})
    updated_run_settings = simulation_settings.run_settings.model_copy(
        update={"dynamics": updated_dynamics}
    )
    return simulation_settings.model_copy(update={"run_settings": updated_run_settings})


@job(
    **{
        VALIDATION_DATASET_TRAJECTORY_BLOB_STORE_NAME: [
            ["output", "properties", "trajectory_bytes"]
        ]
    }
)  # type: ignore[call-overload]
def run_validation_dataset_md(
    *,
    run_spec: ValidationDatasetMdRunSpec,
    structure: PMG_Structure,
    simulation_settings: MlipMdEquilibrationSimulationSettings,
    requested_gpu_count: int = 0,
) -> ValidationDatasetMdTaskDocument:
    """Run one MLIP-MD simulation.

    This is a workflow step because its trajectory must be saved without passing the
    trajectory data through later job inputs.
    """
    try:
        configure_default_torch_device(requested_gpu_count=requested_gpu_count)
        temperature_settings = build_validation_dataset_md_settings_at_temperature(
            simulation_settings=simulation_settings,
            temperature_k=run_spec.temperature_k,
        )
        stage_result = MlipMdRunner(
            run_settings=temperature_settings.run_settings, structure=structure
        ).run()
        trajectory_path = stage_result.trajectory_path
        trajectory_bytes = trajectory_path.read_bytes()
        with Trajectory(  # type: ignore[no-untyped-call]
            trajectory_path, mode="r"
        ) as trajectory:
            frame_count = len(trajectory)
        properties = ValidationDatasetMdProperties(
            run_id=run_spec.run_id,
            source_kind=run_spec.source_kind,
            source_id=run_spec.source_id,
            temperature_k=run_spec.temperature_k,
            frame_count=frame_count,
            number_of_atoms=len(structure),
            trajectory_bytes=trajectory_bytes,
        )
        errors = ""
    except Exception as exc:  # noqa: BLE001 - preserve one-run diagnostics
        properties = None
        errors = get_diagnostic_exception_message(exc)

    return ValidationDatasetMdTaskDocument(
        run_id=run_spec.run_id, properties=properties, errors=errors
    )
