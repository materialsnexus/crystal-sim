"""Deferred cohesive-energy filtering job."""

from __future__ import annotations

import logging
from pathlib import Path

from jobflow.core.job import job

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.helpers.cohesive_energy import filter_structures_by_cohesive_energy
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.schemas.filtering import (
    CohesiveEnergyFilteringSettings,
    CohesiveEnergyFilteringTaskDocument,
)
from crystal.schemas.filtering_statistics import CohesiveEnergyFilteringStatistics
from crystal.schemas.properties import CohesiveEnergyFilteringProperties, StructureList

logger = logging.getLogger(__name__)


@job(
    output_schema=CohesiveEnergyFilteringTaskDocument,
    **{STRUCTURE_BLOB_STORE_NAME: [["output", "properties", "structures"]]},
)  # type: ignore[call-overload]
def run_cohesive_energy_filtering(
    *,
    structures: StructureList,
    total_energies: list[float],
    settings: CohesiveEnergyFilteringSettings,
    failed_count: int = 0,
    warnings: tuple[str, ...] = (),
) -> CohesiveEnergyFilteringTaskDocument:
    """Filter structures by cohesive energy and return only survivors."""
    run_directory = Path.cwd()
    try:
        filtering_result = filter_structures_by_cohesive_energy(
            structures=structures, total_energies=total_energies, settings=settings
        )
        filtered_structures = (
            structures if filtering_result is None else filtering_result
        )
        statistics = CohesiveEnergyFilteringStatistics(
            considered=len(structures) + failed_count,
            passed=len(filtered_structures),
            filtered=len(structures) - len(filtered_structures),
            failed=failed_count,
        )
        properties = CohesiveEnergyFilteringProperties(
            structures=filtered_structures, statistics=statistics
        )
        errors = ""
    except Exception as exc:
        properties = None
        errors = get_diagnostic_exception_message(exc)
        logger.exception("Cohesive-energy filtering failed.")

    return CohesiveEnergyFilteringTaskDocument(
        dir_name=run_directory.as_posix(),
        properties=properties,
        errors=errors,
        warnings="\n".join(warnings),
    )
