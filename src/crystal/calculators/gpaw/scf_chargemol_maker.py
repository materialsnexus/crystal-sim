"""Maker for GPAW SCF followed by Chargemol in the same run directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jobflow.core.job import Job, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.gpaw.chargemol_maker import GpawChargemolMaker
from crystal.calculators.gpaw.gpaw_runner import run_gpaw_scf
from crystal.constants.gpaw import GPAW_TOTAL_DENSITY_CUBE_FILENAME
from crystal.schemas.base import BaseMaker
from crystal.schemas.gpaw import (
    GpawScfChargemolSimulationSettings,
    GpawScfChargemolTaskDocument,
)
from crystal.schemas.maker_names import MakerName


@dataclass(kw_only=True)
class GpawScfChargemolMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.gpaw_scf_chargemol_integration
    simulation_settings: GpawScfChargemolSimulationSettings

    @job(output_schema=GpawScfChargemolTaskDocument)
    def _run(self, structure: PMG_Structure) -> GpawScfChargemolTaskDocument:
        run_directory = Path.cwd().resolve()
        run_directory.mkdir(parents=True, exist_ok=True)
        try:
            scf_result = run_gpaw_scf(
                structure=structure,
                simulation_settings=self.simulation_settings.gpaw_scf_simulation_settings,
                number_of_processes=self.executor_settings.number_of_processes,
                run_directory=run_directory,
                local_scratch_policy=self.executor_settings.local_scratch_policy,
            )
        except RuntimeError as runtime_error:
            return GpawScfChargemolTaskDocument(
                dir_name=run_directory.as_posix(),
                errors=str(runtime_error),
                properties=None,
                structure=None,
                chargemol_run_info=None,
            )

        gpaw_chargemol_maker = GpawChargemolMaker(
            simulation_settings=self.simulation_settings.gpaw_chargemol_simulation_settings,
            executor_settings=self.executor_settings,
        )
        chargemol_task_document = gpaw_chargemol_maker.run_chargemol_workflow(
            job_control_directory=run_directory,
            structure=scf_result.properties.structure,
            properties=scf_result.properties,
        )
        if not chargemol_task_document.converged:
            return GpawScfChargemolTaskDocument.model_validate(
                chargemol_task_document.model_dump()
            )
        self._remove_total_density_cube_after_successful_labelling(
            run_directory=run_directory
        )
        return GpawScfChargemolTaskDocument.model_validate(
            chargemol_task_document.model_dump()
            | {
                "gpaw_scf_simulation_settings": (
                    self.simulation_settings.gpaw_scf_simulation_settings
                ),
                "runtime_provenance": scf_result.runtime_provenance,
            }
        )

    @staticmethod
    def _remove_total_density_cube_after_successful_labelling(
        *, run_directory: Path
    ) -> None:
        (run_directory / GPAW_TOTAL_DENSITY_CUBE_FILENAME).unlink(missing_ok=True)

    def make(self, structure: PMG_Structure) -> Job:  # noqa: D102
        gpaw_scf_chargemol_job = self._run(structure=structure)
        gpaw_scf_chargemol_job.update_metadata(
            self.slurm_settings.to_queueadapter_spec(), dynamic=False
        )
        return gpaw_scf_chargemol_job
