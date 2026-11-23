# noqa: D100
import logging
from dataclasses import dataclass
from pathlib import Path

from jobflow.core.job import Job, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.qe.qe_runner import run_qe_relax
from crystal.schemas.base import BaseMaker
from crystal.schemas.maker_names import MakerName
from crystal.schemas.qe import QeRelaxSimulationSettings, QeRelaxTaskDocument

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class QeRelaxMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.qe_relax
    simulation_settings: QeRelaxSimulationSettings

    @job(output_schema=QeRelaxTaskDocument)
    def _run(self, structure: PMG_Structure) -> QeRelaxTaskDocument:
        logger.info("Running QE Relax calculation for structure: %s", structure.formula)
        try:
            properties = run_qe_relax(
                structure=structure,
                simulation_settings=self.simulation_settings,
                number_of_processes=self.executor_settings.number_of_processes,
            )
        except RuntimeError as rte:
            logger.info(str(rte))
            properties = None
        task_document = QeRelaxTaskDocument.from_directory_and_properties(
            dir_path=Path.cwd(), properties=properties
        )
        logger.info("Finished QE Relax calculation")
        return task_document

    def make(self, structure: PMG_Structure) -> Job:  # noqa: D102
        job = self._run(structure=structure)
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job
