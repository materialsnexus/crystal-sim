"""Jobflow jobs for single-point structure labelling."""

from __future__ import annotations

from jobflow.core.flow import Flow
from jobflow.core.job import Response, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.single_point_maker_factory import SinglePointMaker


@job
def fan_out_single_points(
    source_structures: list[PMG_Structure], single_point_maker: SinglePointMaker
) -> Response[Flow]:
    """Create one independent single-point job per source structure.

    The source structures are produced by an earlier Jobflow job, so ``make`` only
    sees a reference to them while the workflow is being built. This job runs after
    that reference is resolved and dynamically creates the child jobs at execution
    time.
    """
    if not source_structures:
        return Response(replace=Flow(jobs=[], output=[]))

    child_jobs = [
        single_point_maker.make(structure=structure) for structure in source_structures
    ]

    return Response(replace=Flow(jobs=child_jobs, output=None))
