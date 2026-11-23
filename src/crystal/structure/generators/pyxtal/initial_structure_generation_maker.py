"""Jobflow maker for deterministic PyXtal structure generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from jobflow.core.job import Job, job
from pydantic import PositiveInt, validate_call
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.schemas.base import BaseMaker
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationProperties,
    InitialStructureGenerationSettings,
    StructureIndexRange,
)
from crystal.schemas.maker_names import MakerName
from crystal.schemas.structure_generators import InitialStructureGenerationTaskDocument
from crystal.structure.generators.pyxtal.initial_structure_generator import (
    InitialStructureGenerator,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class InitialStructureGenerationMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.initial_structure_generation
    simulation_settings: InitialStructureGenerationSettings

    @cached_property
    def _generator(self) -> InitialStructureGenerator:
        return InitialStructureGenerator(settings=self.simulation_settings)

    def _generate_structures(
        self, generation_range: StructureIndexRange
    ) -> list[PMG_Structure]:
        """Generate structures in order until one fails."""
        structures: list[PMG_Structure] = []
        for structure_index in generation_range.structure_indices:
            try:
                structures.append(
                    self._generator.generate(structure_index=structure_index)
                )
            except Exception as exc:  # noqa: BLE001 - record one bounded batch failure
                logger.warning("PyXtal generation stopped: %s", exc, exc_info=True)
                break
        return structures

    def _build_task_document(
        self,
        generation_range: StructureIndexRange,
        structures: list[PMG_Structure],
        run_directory: Path,
    ) -> InitialStructureGenerationTaskDocument:
        """Build the task document for a complete or partial generation."""
        if not structures:
            errors = (
                "Failed to generate any of the requested "
                f"{generation_range.number_of_structures} structures."
            )
            logger.error(errors)
            return InitialStructureGenerationTaskDocument(
                dir_name=run_directory.as_posix(), properties=None, errors=errors
            )

        terminated_early = len(structures) < generation_range.number_of_structures
        if terminated_early:
            logger.warning(
                "PyXtal generation stopped after generating "
                "%s of %s requested structures.",
                len(structures),
                generation_range.number_of_structures,
            )

        properties = InitialStructureGenerationProperties(
            generation_range=generation_range,
            structures=structures,
            terminated_early=terminated_early,
        )
        return InitialStructureGenerationTaskDocument(
            dir_name=run_directory.as_posix(), properties=properties
        )

    # For additional stores, Jobflow interprets each nested list as a path
    # in the job document; this selects output["properties"]["structures"].
    @job(
        output_schema=InitialStructureGenerationTaskDocument,
        **{STRUCTURE_BLOB_STORE_NAME: [["output", "properties", "structures"]]},
    )  # type: ignore[call-overload]
    def _run(
        self, generation_range: StructureIndexRange
    ) -> InitialStructureGenerationTaskDocument:
        run_directory = Path.cwd()
        logger.info(
            "Running PyXtal generation for structure indices %s to %s",
            generation_range.first_structure_index,
            generation_range.last_structure_index,
        )
        task_document = self._build_task_document(
            generation_range=generation_range,
            structures=self._generate_structures(generation_range),
            run_directory=run_directory,
        )
        logger.info("Finished PyXtal generation")
        return task_document

    @validate_call
    def make(self, number_of_structures: PositiveInt) -> Job:  # noqa: D102
        generation_range = StructureIndexRange(
            first_structure_index=0, last_structure_index=number_of_structures - 1
        )
        generation_job: Job = self._run(generation_range=generation_range)
        generation_job.update_metadata(
            self.slurm_settings.to_queueadapter_spec(), dynamic=False
        )
        return generation_job
