"""Workflow maker for MD adhesion energy calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jobflow.core.flow import Flow
from jobflow.core.job import Job, Response, job
from jobflow.core.reference import OutputReference
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.md.equilibration_maker import MlipMdEquilibrationMaker
from crystal.helpers.cohesive_energy import scale_total_energy_with_num_atoms
from crystal.schemas.base import BaseMaker
from crystal.schemas.maker_names import MakerName
from crystal.schemas.md.adhesion import (
    MDAdhesionEnergyResult,
    MDAdhesionEnergySimulationSettings,
    MDAdhesionStructures,
)
from crystal.schemas.md.mlip_md_equilibration import (
    MlipMdEquilibrationSimulationSettings,
    MlipMdEquilibrationTaskDocument,
)
from crystal.schemas.settings import ExecutorSettings
from crystal.structure.interfaces.builder import Interface


def _interface_area_in_angstrom_square(interface_structure: PMG_Structure) -> float:
    lattice_vectors = np.array(interface_structure.lattice.matrix, dtype=float)
    return float(np.linalg.norm(np.cross(lattice_vectors[0], lattice_vectors[1])))


@job
def get_md_equilibration_flow(
    stage_settings: list[MlipMdEquilibrationSimulationSettings],
    md_executor_settings: ExecutorSettings,
    structure: PMG_Structure,
) -> Response[MlipMdEquilibrationTaskDocument]:
    """Creates a flow for equilibrating structure with a series of MD Equilibration settings.

    Typical use case will follow, for e.g.,
    1. NVT simulations with the initial structure and check if the structure equilibrated successfully in this ensemble
    2. NPT simulations with the final trajectory (including atom velocity) and check structure equilibration
    """
    if len(stage_settings) == 0:
        err_msg = "At least one MD equilibration stage setting is required."
        raise ValueError(err_msg)

    first_md_equilibration_maker = MlipMdEquilibrationMaker(
        simulation_settings=stage_settings[0], executor_settings=md_executor_settings
    )

    first_md_equilibration_job = first_md_equilibration_maker.make(structure=structure)

    followup_md_jobs = continue_md_equilibration(
        previous_md_task_document=first_md_equilibration_job.output,
        remaining_stage_settings=stage_settings[1:],
        md_executor_settings=md_executor_settings,
    )
    flow = Flow(
        jobs=[first_md_equilibration_job, followup_md_jobs],
        output=followup_md_jobs.output,
    )
    return Response(replace=flow)


@job
def continue_md_equilibration(  # noqa: D103
    previous_md_task_document: MlipMdEquilibrationTaskDocument,
    remaining_stage_settings: list[MlipMdEquilibrationSimulationSettings],
    md_executor_settings: ExecutorSettings,
) -> Response[MlipMdEquilibrationTaskDocument]:
    if len(remaining_stage_settings) == 0 or (
        not previous_md_task_document.is_successful_md_equilibration
    ):
        return Response(output=previous_md_task_document)

    next_md_equilibration_maker = MlipMdEquilibrationMaker(
        simulation_settings=remaining_stage_settings[0],
        executor_settings=md_executor_settings,
    )
    assert previous_md_task_document.properties is not None
    next_md_equilibration_job = next_md_equilibration_maker.make(
        trajectory_path=previous_md_task_document.properties.md_stage_result.trajectory_path
    )

    continue_job = continue_md_equilibration(
        previous_md_task_document=next_md_equilibration_job.output,
        remaining_stage_settings=remaining_stage_settings[1:],
        md_executor_settings=md_executor_settings,
    )

    return Response(
        replace=Flow(
            jobs=[next_md_equilibration_job, continue_job], output=continue_job.output
        )
    )


@job
def prepare_production_structures(
    interface_md_output: MlipMdEquilibrationTaskDocument,
) -> MDAdhesionStructures:
    """Derive production structures from the final interface MD frame."""
    if not interface_md_output.is_successful_md_equilibration:
        error_message = (
            "Interface MD failed before production MD: "
            f"errors={interface_md_output.errors!r}"
        )
        raise RuntimeError(error_message)
    assert interface_md_output.properties is not None
    return MDAdhesionStructures.from_interface_structure(
        interface_md_output.properties.last_frame_structure
    )


@job
def collect_and_build_adhesion_energy_record(  # noqa: D103
    *,
    top_slab_md_output: MlipMdEquilibrationTaskDocument,
    bottom_slab_md_output: MlipMdEquilibrationTaskDocument,
    interface_md_output: MlipMdEquilibrationTaskDocument,
) -> MDAdhesionEnergyResult:
    failed_md_outputs = []
    for task_doc_name, task_doc in (
        ("Top Slab", top_slab_md_output),
        ("Bottom Slab", bottom_slab_md_output),
        ("Interface", interface_md_output),
    ):
        if not task_doc.is_successful_md_equilibration:
            failed_md_outputs.append(f"{task_doc_name}: errors={task_doc.errors!r}")
    if failed_md_outputs:
        raise RuntimeError(
            "One or more MD outputs are unconverged:\n" + "\n".join(failed_md_outputs)
        )
    assert top_slab_md_output.properties is not None
    assert bottom_slab_md_output.properties is not None
    assert interface_md_output.properties is not None
    top_slab_energy = top_slab_md_output.properties.average_energy
    bottom_slab_energy = bottom_slab_md_output.properties.average_energy
    interface_energy = interface_md_output.properties.average_energy
    top_slab_structure = top_slab_md_output.properties.last_frame_structure
    bottom_slab_structure = bottom_slab_md_output.properties.last_frame_structure
    final_interface_structure = Interface.from_structure(
        interface_md_output.properties.last_frame_structure
    )

    scaled_top_slab_energy = scale_total_energy_with_num_atoms(
        total_energy=top_slab_energy,
        initial_num_atoms=len(top_slab_structure),
        final_num_atoms=len(final_interface_structure.film),
    )
    scaled_bottom_slab_energy = scale_total_energy_with_num_atoms(
        total_energy=bottom_slab_energy,
        initial_num_atoms=len(bottom_slab_structure),
        final_num_atoms=len(final_interface_structure.substrate),
    )
    interface_area = _interface_area_in_angstrom_square(
        interface_structure=final_interface_structure
    )
    adhesion_energy = (
        scaled_top_slab_energy + scaled_bottom_slab_energy - interface_energy
    )
    work_of_adhesion_per_area = adhesion_energy / interface_area
    return MDAdhesionEnergyResult(
        top_slab_energy=top_slab_energy,
        bottom_slab_energy=bottom_slab_energy,
        interface_energy=interface_energy,
        interface_area=interface_area,
        work_of_adhesion_per_area=work_of_adhesion_per_area,
    )


@dataclass(kw_only=True)
class MDAdhesionEnergyMaker(BaseMaker):
    """Build MD adhesion energy workflow slices."""

    name: MakerName = MakerName.md_adhesion_energy
    simulation_settings: MDAdhesionEnergySimulationSettings

    @job
    def make(
        self, interface_structure: PMG_Structure
    ) -> Response[MDAdhesionEnergyResult]:
        """Build interface preparation, production MD and aggregation after resolving the interface."""
        jobs: list[Job] = []
        if self.simulation_settings.interface_md_settings:
            interface_md_flow = get_md_equilibration_flow(
                stage_settings=self.simulation_settings.interface_md_settings,
                md_executor_settings=self.simulation_settings.md_executor_settings,
                structure=interface_structure,
            )
            structures_job = prepare_production_structures(
                interface_md_output=interface_md_flow.output
            )
            structures_job.update_metadata(
                self.slurm_settings.to_queueadapter_spec(), dynamic=False
            )
            jobs.extend([interface_md_flow, structures_job])
            structures: MDAdhesionStructures | OutputReference = structures_job.output
        else:
            structures = MDAdhesionStructures.from_interface_structure(
                interface_structure
            )
        production_settings = [self.simulation_settings.production_md_setting]
        top_slab_md_flow = get_md_equilibration_flow(
            stage_settings=production_settings,
            md_executor_settings=self.simulation_settings.md_executor_settings,
            structure=structures.top_slab_structure,
        )
        bottom_slab_md_flow = get_md_equilibration_flow(
            stage_settings=production_settings,
            md_executor_settings=self.simulation_settings.md_executor_settings,
            structure=structures.bottom_slab_structure,
        )
        production_interface_md_flow = get_md_equilibration_flow(
            stage_settings=production_settings,
            md_executor_settings=self.simulation_settings.md_executor_settings,
            structure=structures.interface_structure,
        )
        adhesion_energy_job = collect_and_build_adhesion_energy_record(
            top_slab_md_output=top_slab_md_flow.output,
            bottom_slab_md_output=bottom_slab_md_flow.output,
            interface_md_output=production_interface_md_flow.output,
        )
        adhesion_energy_job.update_metadata(
            self.slurm_settings.to_queueadapter_spec(), dynamic=False
        )

        jobs.extend(
            [
                top_slab_md_flow,
                bottom_slab_md_flow,
                production_interface_md_flow,
                adhesion_energy_job,
            ]
        )
        return Response(replace=Flow(jobs=jobs, output=adhesion_energy_job.output))
