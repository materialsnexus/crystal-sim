# noqa: D100


import logging
from dataclasses import dataclass

from jobflow.core.job import Job, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.mlip_runner import (
    run_mlip_single_point,
    run_mlip_single_point_batch,
)
from crystal.constants.mlip import MLIP_SINGLE_POINT_TASK_DOCUMENT_BLOB_STORE_NAME
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.helpers.mlip_helpers import configure_default_torch_device
from crystal.schemas.base import BaseMaker
from crystal.schemas.maker_names import MakerName
from crystal.schemas.mlip import (
    MlipSinglePointSimulationSettings,
    MlipSinglePointTaskDocument,
)
from crystal.schemas.properties import StructureList

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MlipSinglePointMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.mlip_single_point
    simulation_settings: MlipSinglePointSimulationSettings

    @job
    def _run(self, structure: PMG_Structure) -> MlipSinglePointTaskDocument:
        """Maker to run MLIP single point maker."""
        logger.info(
            "Running MLIP single point prediction for structure: %s", structure.formula
        )
        try:
            configure_default_torch_device(self.executor_settings.number_of_gpus)
            properties = run_mlip_single_point(
                structure=structure, simulation_settings=self.simulation_settings
            )
            errors = ""
        except Exception as exc:
            logger.exception("MLIP single point calculation failed.")
            properties = None
            errors = str(exc)

        task_document = MlipSinglePointTaskDocument(
            properties=properties, errors=errors
        )
        logger.info("Finished MLIP single point prediction")
        return task_document

    def make(self, structure: PMG_Structure) -> Job:  # noqa: D102
        job = self._run(structure=structure)
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job


@dataclass(kw_only=True)
class MlipSinglePointBatchMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.mlip_single_point_batch
    simulation_settings: MlipSinglePointSimulationSettings

    @job(
        **{MLIP_SINGLE_POINT_TASK_DOCUMENT_BLOB_STORE_NAME: MlipSinglePointTaskDocument}
    )
    def _run(self, structures: StructureList) -> list[MlipSinglePointTaskDocument]:
        """Run one MLIP single-point batch and preserve per-structure outcomes."""
        logger.info(
            "Running MLIP single-point calculation batch for %s structures",
            len(structures),
        )
        try:
            configure_default_torch_device(self.executor_settings.number_of_gpus)
            task_documents = run_mlip_single_point_batch(
                structures=structures, simulation_settings=self.simulation_settings
            )
        except Exception as exc:
            error = get_diagnostic_exception_message(exc)
            logger.exception("MLIP single-point calculation batch failed.")
            task_documents = [
                MlipSinglePointTaskDocument(properties=None, errors=error)
                for _ in structures
            ]
        logger.info("Finished MLIP single-point calculation batch")
        return task_documents

    def make(self, structures: StructureList) -> Job:  # noqa: D102
        job: Job = self._run(structures=structures)  # type: ignore[operator]
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job
