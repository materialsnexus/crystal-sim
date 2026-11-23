"""Structure filtering maker."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from jobflow.core.job import Job, job

from crystal.helpers.cohesive_energy import CohesiveEnergiesCalculator
from crystal.helpers.properties import get_maximum_force_norm, get_stress_norm
from crystal.schemas.base import BaseMaker
from crystal.schemas.filtering import (
    SinglePointStructureFilteringSettings,
    SinglePointStructureFilteringTaskDocument,
)
from crystal.schemas.maker_names import MakerName
from crystal.schemas.properties import (
    SinglePointProperties,
    SinglePointStructureFilteringProperties,
    StructureWithSinglePointFilteringMetrics,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SinglePointStructureFilteringMaker(BaseMaker):
    """Filter structures by metrics from existing single-point calculations."""

    name: MakerName = MakerName.single_point_filter
    simulation_settings: SinglePointStructureFilteringSettings

    def _get_structures_with_metrics(
        self, filtering_inputs: list[SinglePointProperties]
    ) -> list[StructureWithSinglePointFilteringMetrics]:
        cohesive_energy_calculator = CohesiveEnergiesCalculator(
            isolated_atom_energies=self.simulation_settings.isolated_atom_energies.energies
        )
        cohesive_energies_per_atom = cohesive_energy_calculator.from_pmg_structures(
            structures=[single_point.structure for single_point in filtering_inputs],
            total_energies=[single_point.energy for single_point in filtering_inputs],
        )
        return [
            StructureWithSinglePointFilteringMetrics(
                structure=single_point.structure,
                cohesive_energy_per_atom=cohesive_energy_per_atom,
                maximum_force_norm=get_maximum_force_norm(single_point.forces),
                stress_norm=get_stress_norm(single_point.stress),
            )
            for single_point, cohesive_energy_per_atom in zip(
                filtering_inputs, cohesive_energies_per_atom, strict=True
            )
        ]

    def _get_filtering_mask(
        self, structures_with_metrics: list[StructureWithSinglePointFilteringMetrics]
    ) -> list[bool]:
        return [
            datum.cohesive_energy_per_atom
            <= self.simulation_settings.maximum_cohesive_energy_per_atom
            and datum.maximum_force_norm <= self.simulation_settings.maximum_force_norm
            and datum.stress_norm <= self.simulation_settings.maximum_stress_norm
            for datum in structures_with_metrics
        ]

    @job(output_schema=SinglePointStructureFilteringTaskDocument)
    def _run(
        self, single_points: list[SinglePointProperties]
    ) -> SinglePointStructureFilteringTaskDocument:
        try:
            structures_with_metrics = self._get_structures_with_metrics(
                filtering_inputs=single_points
            )
            filtering_mask = self._get_filtering_mask(
                structures_with_metrics=structures_with_metrics
            )
            properties = SinglePointStructureFilteringProperties(
                all_structures_with_metrics=structures_with_metrics,
                filtering_mask=filtering_mask,
            )
            errors = ""
            logger.info(
                "Single-point structure filtering complete: %d/%d structures passed.",
                len(properties.structures),
                len(single_points),
            )
        except Exception as exc:
            properties = None
            errors = str(exc)
            logger.exception("Single-point structure filtering failed")

        return SinglePointStructureFilteringTaskDocument(
            properties=properties, errors=errors
        )

    def make(self, single_points: list[SinglePointProperties]) -> Job:
        """Create a filtering job for a list of single-point calculations."""
        job = self._run(single_points)
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job
