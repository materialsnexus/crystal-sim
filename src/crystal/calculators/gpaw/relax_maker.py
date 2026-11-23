"""Maker for GPAW relax calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from jobflow.core.job import Job, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.gpaw.relax_runner import run_gpaw_relax
from crystal.schemas.base import BaseMaker
from crystal.schemas.gpaw import GpawRelaxSimulationSettings, GpawRelaxTaskDocument
from crystal.schemas.maker_names import MakerName

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GpawRelaxMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.gpaw_relax
    simulation_settings: GpawRelaxSimulationSettings

    @job(output_schema=GpawRelaxTaskDocument)
    def _run(self, structure: PMG_Structure) -> GpawRelaxTaskDocument:
        logger.info(
            "Running GPAW relax calculation for structure: %s", structure.formula
        )
        run_directory = Path.cwd().resolve()
        run_directory.mkdir(parents=True, exist_ok=True)
        try:
            properties = run_gpaw_relax(
                structure=structure,
                simulation_settings=self.simulation_settings,
                run_directory=run_directory,
                number_of_processes=self.executor_settings.number_of_processes,
                local_scratch_policy=self.executor_settings.local_scratch_policy,
            )
            error_log = ""
        except RuntimeError as runtime_error:
            logger.info(str(runtime_error))
            properties = None
            error_log = str(runtime_error)
        task_document = GpawRelaxTaskDocument.from_directory_and_properties(
            dir_path=run_directory, properties=properties, error_log=error_log
        )
        logger.info("Finished GPAW relax calculation")
        return task_document

    def make(self, structure: PMG_Structure) -> Job:  # noqa: D102
        job = self._run(structure=structure)
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job
