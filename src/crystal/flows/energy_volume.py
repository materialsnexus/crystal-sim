# noqa: D100
import logging
from typing import cast

import numpy as np
from jobflow.core.flow import Flow
from jobflow.core.job import Job, job
from numpy.typing import NDArray
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.relax_maker_factory import RelaxMaker
from crystal.calculators.single_point_maker_factory import SinglePointMaker
from crystal.helpers.structure import get_strained_structures
from crystal.schemas.energy_volume import EnergyVolumeOutput
from crystal.schemas.single_point import SinglePointTaskDocument

logger = logging.getLogger(__name__)


@job
def _strain_structure(
    structure: PMG_Structure, strains: NDArray[np.float64]
) -> list[PMG_Structure]:
    return get_strained_structures(structure=structure, strains=strains)


@job
def _process_single_points_for_energy_volume(
    single_points: list[SinglePointTaskDocument],
) -> list[EnergyVolumeOutput]:
    outputs: list[EnergyVolumeOutput] = []
    for task_doc in single_points:
        if task_doc.errors:
            logger.warning("Skipping task with errors: %s", task_doc.errors)
            continue
        outputs.append(EnergyVolumeOutput.from_single_point_task_doc(task_doc))
    return sorted(outputs, key=lambda x: x.volume)


def compute_energy_volume_curves(
    structure: PMG_Structure,
    linear_strain: tuple[float, float],
    strain_steps: int,
    single_point_maker: SinglePointMaker,
    relax_maker: RelaxMaker | None = None,
) -> Flow:
    """Computes energy volume for a structure by performing single point calculations on linearly strained structures.

    Optionally allows relaxing the structure if relax maker is provided.
    """
    all_jobs: list[Job] = []
    if relax_maker is None:
        initial_structure = structure
    else:
        relax_job = relax_maker.make(structure=structure)
        initial_structure = cast(
            "PMG_Structure", relax_job.output.properties.relaxed_structure
        )
        relax_job.name = "Relax Structure"
        all_jobs.append(relax_job)

    strains = np.linspace(linear_strain[0], linear_strain[1], strain_steps)
    strain_structure_job = _strain_structure(
        structure=initial_structure, strains=strains
    )
    strain_structure_job.name = "Strain Structure"
    all_jobs.append(strain_structure_job)

    single_point_jobs: list[Job] = []
    for i in range(strain_steps):
        # NOTE: the cast operation is only for static checkers, using indexes is a valid jobflow pattern
        strained_structure = cast("PMG_Structure", strain_structure_job.output[i])
        single_point_job = single_point_maker.make(structure=strained_structure)
        single_point_job.name = (
            f"Single point for strain index = {i} and strain = {strains[i]}"
        )
        single_point_jobs.append(single_point_job)
    all_jobs.extend(single_point_jobs)

    processing_job = _process_single_points_for_energy_volume(
        single_points=[j.output for j in single_point_jobs]
    )
    processing_job.name = "Extract Energy Volume Output"
    all_jobs.append(processing_job)

    return Flow(
        jobs=all_jobs,
        output=processing_job.output,
        name=f"E-V data for {structure.formula}",
    )
