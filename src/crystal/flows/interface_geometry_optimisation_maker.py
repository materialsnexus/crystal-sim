"""Workflow maker for standalone interface geometry optimisation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from jobflow.core.flow import Flow
from jobflow.core.job import Response, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.relax_maker import MlipRelaxMaker
from crystal.helpers.structure import split_items_into_batches
from crystal.schemas.base import BaseMaker
from crystal.schemas.interface_geometry_optimisation import (
    GeometryOptimisationCandidateRecord,
    GeometryOptimisationResult,
    InterfaceGeometryOptimisationInputs,
    InterfaceGeometryOptimisationSimulationSettings,
)
from crystal.schemas.maker_names import MakerName
from crystal.schemas.mlip import MlipRelaxTaskDocument
from crystal.schemas.properties import RelaxProperties

logger = logging.getLogger(__name__)


def _candidate_record_from_relax_output(
    *, candidate_index: int, relax_task_document: MlipRelaxTaskDocument
) -> GeometryOptimisationCandidateRecord:
    relax_properties = relax_task_document.properties
    if relax_properties is None:
        return GeometryOptimisationCandidateRecord(
            candidate_index=candidate_index,
            final_relax_energy=None,
            error_message=relax_task_document.errors,
        )
    assert isinstance(relax_properties, RelaxProperties)
    return GeometryOptimisationCandidateRecord(
        candidate_index=candidate_index,
        final_relax_energy=relax_properties.final_energy,
        error_message=relax_task_document.errors,
    )


def _successful_candidate_energy(
    candidate_record: GeometryOptimisationCandidateRecord,
) -> float:
    final_relax_energy = candidate_record.final_relax_energy
    if final_relax_energy is None:
        msg = "Successful geometry optimisation candidate is missing final energy."
        raise ValueError(msg)
    return final_relax_energy


@job
def select_minimum_energy_geometry_optimisation_result(
    *, relax_task_documents: list[MlipRelaxTaskDocument]
) -> Response[GeometryOptimisationResult]:
    """Select the lowest-energy successful relaxation result."""
    candidate_records = [
        _candidate_record_from_relax_output(
            candidate_index=candidate_index, relax_task_document=relax_task_document
        )
        for candidate_index, relax_task_document in enumerate(relax_task_documents)
    ]
    successful_candidate_records = [
        candidate_record
        for candidate_record in candidate_records
        if candidate_record.is_success
    ]
    unsuccessful_relax_count = len(relax_task_documents) - len(
        successful_candidate_records
    )
    if unsuccessful_relax_count > 0:
        logger.warning(
            "%s relax calculations failed out of total %s",
            unsuccessful_relax_count,
            len(relax_task_documents),
        )
    if not successful_candidate_records:
        failure_details = "; ".join(
            f"candidate {record.candidate_index}: "
            f"{record.error_message or 'unknown error'}"
            for record in candidate_records
        )
        msg = (
            "All geometry optimisation candidates failed; no optimised structure "
            f"can be selected. Candidate failures: {failure_details}"
        )
        logger.error(msg)
        return Response(stop_children=True)

    selected_candidate_record = min(
        successful_candidate_records, key=_successful_candidate_energy
    )
    selected_relax_properties = relax_task_documents[
        selected_candidate_record.candidate_index
    ].properties
    if not isinstance(selected_relax_properties, RelaxProperties):
        msg = (
            "Selected geometry optimisation candidate is missing relaxation properties."
        )
        raise TypeError(msg)

    selected_structure = selected_relax_properties.relaxed_structure
    return Response(
        output=GeometryOptimisationResult(
            selected_candidate_index=selected_candidate_record.candidate_index,
            selected_structure=selected_structure,
            selected_final_energy=_successful_candidate_energy(
                selected_candidate_record
            ),
            candidate_records=candidate_records,
        )
    )


def perform_geometry_optimisation(
    relax_maker: MlipRelaxMaker,
    *,
    structures: Sequence[PMG_Structure],
    structures_per_job: int | None = None,
) -> Flow:
    """Build relaxation jobs and select the minimum-energy successful result."""
    if len(structures) == 0:
        msg = "At least one structure is required for geometry optimisation."
        raise ValueError(msg)

    structure_batches = split_items_into_batches(
        structures, items_per_batch=structures_per_job
    )
    relax_flows: list[Flow] = []
    relax_outputs: list[MlipRelaxTaskDocument] = []
    for structure_batch in structure_batches:
        relax_jobs = [
            relax_maker.make(structure=structure) for structure in structure_batch
        ]
        relax_outputs.extend(relax_job.output for relax_job in relax_jobs)
        relax_flows.append(
            Flow(jobs=relax_jobs, output=[relax_job.output for relax_job in relax_jobs])
        )

    result_job = select_minimum_energy_geometry_optimisation_result(
        relax_task_documents=relax_outputs
    )
    return Flow(jobs=[*relax_flows, result_job], output=result_job.output)


@dataclass(kw_only=True)
class InterfaceGeometryOptimisationMaker(BaseMaker):
    """Build slab-driven interface geometry optimisation workflows."""

    name: MakerName = MakerName.interface_geometry_optimisation
    simulation_settings: InterfaceGeometryOptimisationSimulationSettings

    @property
    def relax_maker(self) -> MlipRelaxMaker:
        """Return the configured maker used for candidate relaxations."""
        relax_maker_settings = self.simulation_settings.relax_maker_settings
        return MlipRelaxMaker(
            simulation_settings=relax_maker_settings.simulation_settings,
            executor_settings=relax_maker_settings.executor_settings,
        )

    def make(self, inputs: InterfaceGeometryOptimisationInputs) -> Flow:
        """Build the interface candidate relaxation and selection workflow."""
        return perform_geometry_optimisation(
            self.relax_maker,
            structures=inputs.interface_candidates,
            structures_per_job=inputs.interface_jobs_per_batch,
        )
