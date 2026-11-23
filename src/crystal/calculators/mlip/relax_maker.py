# noqa: D100
import logging
from dataclasses import dataclass
from pathlib import Path

from jobflow.core.job import Job, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.mlip_runner import run_mlip_relax, run_mlip_relax_batch
from crystal.constants.mlip import MLIP_RELAX_TASK_DOCUMENT_BLOB_STORE_NAME
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.helpers.mlip_helpers import configure_default_torch_device
from crystal.schemas.base import BaseMaker
from crystal.schemas.maker_names import MakerName
from crystal.schemas.mlip import MlipRelaxSimulationSettings, MlipRelaxTaskDocument

logger = logging.getLogger(__name__)


def _make_batch_run_directories(structure_count: int) -> list[Path]:
    return [Path.cwd() / f"relax-{index}" for index in range(structure_count)]


def _validate_batch_result_count(
    documents: list[MlipRelaxTaskDocument], structures: list[PMG_Structure]
) -> None:
    if len(documents) != len(structures):
        msg = (
            "MLIP relaxation batch returned "
            f"{len(documents)} documents for {len(structures)} structures."
        )
        raise ValueError(msg)


@dataclass(kw_only=True)
class MlipRelaxMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.mlip_relax
    simulation_settings: MlipRelaxSimulationSettings

    @job
    def _run(self, structure: PMG_Structure) -> MlipRelaxTaskDocument:
        logger.info(
            "Running MLIP Relax calculation for structure: %s", structure.formula
        )
        try:
            configure_default_torch_device(self.executor_settings.number_of_gpus)
            properties = run_mlip_relax(
                structure=structure, simulation_settings=self.simulation_settings
            )
            errors = ""
        except Exception as exc:
            properties = None
            errors = str(exc)
            logger.exception("MLIP relax calculation failed.")

        task_document = MlipRelaxTaskDocument(
            dir_name=Path.cwd().as_posix(), properties=properties, errors=errors
        )
        logger.info("Finished MLIP Relax calculation")
        return task_document

    def make(self, structure: PMG_Structure) -> Job:  # noqa: D102
        job = self._run(structure=structure)
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job


@dataclass(kw_only=True)
class MlipRelaxBatchMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.mlip_relax_batch
    simulation_settings: MlipRelaxSimulationSettings

    @job(**{MLIP_RELAX_TASK_DOCUMENT_BLOB_STORE_NAME: MlipRelaxTaskDocument})
    def _run(self, structures: list[PMG_Structure]) -> list[MlipRelaxTaskDocument]:
        logger.info(
            "Running MLIP Relax calculation batch for %s structures", len(structures)
        )
        run_directories = _make_batch_run_directories(len(structures))
        try:
            for run_directory in run_directories:
                run_directory.mkdir(parents=True, exist_ok=True)
            configure_default_torch_device(self.executor_settings.number_of_gpus)
            task_documents = run_mlip_relax_batch(
                structures=structures,
                simulation_settings=self.simulation_settings,
                run_directories=run_directories,
            )
            _validate_batch_result_count(
                documents=task_documents, structures=structures
            )
        except Exception as exc:
            error = get_diagnostic_exception_message(exc)
            logger.exception("MLIP relax calculation batch failed.")
            task_documents = [
                MlipRelaxTaskDocument(
                    dir_name=run_directory.as_posix(), properties=None, errors=error
                )
                for run_directory in run_directories
            ]
        logger.info("Finished MLIP Relax calculation batch")
        return task_documents

    def make(self, structures: list[PMG_Structure]) -> Job:  # noqa: D102
        job: Job = self._run(structures=structures)  # type: ignore[operator]
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job
