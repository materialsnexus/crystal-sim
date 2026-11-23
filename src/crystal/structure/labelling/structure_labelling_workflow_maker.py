"""Maker for single-point structure-labelling workflows."""

from __future__ import annotations

from dataclasses import dataclass

from jobflow.core.flow import Flow
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.single_point_maker_factory import (
    SinglePointMaker,
    build_single_point_maker,
)
from crystal.schemas.base import BaseMaker
from crystal.schemas.maker_names import MakerName
from crystal.schemas.structure_labelling import StructureLabellingWorkflowSettings
from crystal.structure.labelling.structure_labelling_workflow import (
    fan_out_single_points,
)


@dataclass(kw_only=True)
class StructureLabellingWorkflowMaker(BaseMaker):
    """Orchestrate single-point labels for source structures."""

    name: MakerName = MakerName.structure_labelling_workflow
    simulation_settings: StructureLabellingWorkflowSettings

    @property
    def single_point_maker(self) -> SinglePointMaker:
        """Return the configured maker for each structure-labelling job."""
        return build_single_point_maker(
            single_point_maker_settings=self.simulation_settings.single_point_maker_settings
        )

    def make(self, source_structures: list[PMG_Structure]) -> Flow:
        """Build a single-point labelling workflow from source structures."""
        fanout_job = fan_out_single_points(
            source_structures=source_structures,
            single_point_maker=self.single_point_maker,
        )

        workflow_metadata = self.slurm_settings.to_queueadapter_spec()
        fanout_job.update_metadata(workflow_metadata, dynamic=False)

        return Flow(jobs=[fanout_job], output=None)
