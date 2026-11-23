"""Maker for running Chargemol from a prepared GPAW run directory."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from jobflow.core.job import Job, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal import CRYSTAL_CONFIG
from crystal.constants.chargemol import (
    CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE,
    CHARGEMOL_EXECUTABLE_RELATIVE_PATH,
    CHARGEMOL_JOB_CONTROL_FILENAME,
    CHARGEMOL_MISSING_OUTPUT_FILES_MESSAGE_TEMPLATE,
    CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME,
    CHARGEMOL_PROCESS_NAME,
)
from crystal.constants.gpaw import GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES
from crystal.constants.process import EMPTY_ERROR_LOG, PROCESS_FAILURE_MESSAGE_TEMPLATE
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.helpers.chargemol.charge_parser import (
    ChargemolNetAtomicChargesParser,
    attach_chargemol_net_atomic_charges_to_single_point_properties,
)
from crystal.helpers.process_run_info import build_process_run_info_for_failure
from crystal.schemas.base import BaseMaker
from crystal.schemas.chargemol import (
    ChargemolInstallPathsConfigLoader,
    ChargemolJobControlInputBuilder,
    ChargemolJobControlTag,
)
from crystal.schemas.gpaw import (
    GpawChargemolSimulationSettings,
    GpawChargemolTaskDocument,
)
from crystal.schemas.maker_names import MakerName
from crystal.schemas.process import ProcessRunInfo
from crystal.schemas.properties import SinglePointProperties


@dataclass(kw_only=True)
class GpawChargemolMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.gpaw_chargemol_integration
    simulation_settings: GpawChargemolSimulationSettings

    @cached_property
    def executable(self) -> Path:  # noqa: D102
        chargemol_directory = CRYSTAL_CONFIG.programs.chargemol_dir
        if chargemol_directory is None:
            raise FileNotFoundError(CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE)
        return (
            Path(chargemol_directory) / CHARGEMOL_EXECUTABLE_RELATIVE_PATH
        ).resolve()

    def _run_chargemol(self, *, job_control_directory: Path) -> ProcessRunInfo:
        command_arguments: list[str] | None = None
        try:
            command_arguments = [
                self.executable.as_posix(),
                CHARGEMOL_JOB_CONTROL_FILENAME,
            ]
            self._prepare_job_control_for_atomic_densities_directory(
                job_control_directory=job_control_directory
            )
            completed_process = subprocess.run(
                command_arguments, cwd=job_control_directory, check=True, text=True
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            RuntimeError,
            OSError,
        ) as exc:
            return build_process_run_info_for_failure(
                error=exc,
                command_arguments=command_arguments or [],
                standard_output_file=None,
                standard_error_file=None,
            )
        return ProcessRunInfo(
            command_arguments=command_arguments,
            exit_code=completed_process.returncode,
            standard_output_file=None,
            standard_error_file=None,
        )

    @staticmethod
    def _replace_atomic_densities_directory_value(
        *, job_control_text: str, atomic_densities_directory: Path
    ) -> str:
        opening_tag = ChargemolJobControlInputBuilder.format_chargemol_opening_tag(
            ChargemolJobControlTag.atomic_densities_directory
        )
        closing_tag = ChargemolJobControlInputBuilder.format_chargemol_closing_tag(
            ChargemolJobControlTag.atomic_densities_directory
        )
        job_control_lines = job_control_text.splitlines()
        if opening_tag not in job_control_lines:
            return job_control_text
        opening_tag_index = job_control_lines.index(opening_tag)
        try:
            closing_tag_index = job_control_lines.index(
                closing_tag, opening_tag_index + 1
            )
        except ValueError:
            return job_control_text
        atomic_densities_directory_line = (
            ChargemolJobControlInputBuilder.format_chargemol_atomic_densities_directory(
                atomic_densities_directory
            )
        )
        updated_job_control_lines = [
            *job_control_lines[: opening_tag_index + 1],
            atomic_densities_directory_line,
            *job_control_lines[closing_tag_index:],
        ]
        if updated_job_control_lines == job_control_lines:
            return job_control_text
        updated_job_control_text = "\n".join(updated_job_control_lines)
        if job_control_text.endswith("\n"):
            return f"{updated_job_control_text}\n"
        return updated_job_control_text

    def _prepare_job_control_for_atomic_densities_directory(
        self, *, job_control_directory: Path
    ) -> None:
        job_control_path = job_control_directory / CHARGEMOL_JOB_CONTROL_FILENAME
        if not job_control_path.is_file():
            return
        job_control_text = job_control_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
        opening_tag = ChargemolJobControlInputBuilder.format_chargemol_opening_tag(
            ChargemolJobControlTag.atomic_densities_directory
        )
        closing_tag = ChargemolJobControlInputBuilder.format_chargemol_closing_tag(
            ChargemolJobControlTag.atomic_densities_directory
        )
        if opening_tag not in job_control_text or closing_tag not in job_control_text:
            return
        install_paths = ChargemolInstallPathsConfigLoader.discover()
        updated_job_control_text = self._replace_atomic_densities_directory_value(
            job_control_text=job_control_text,
            atomic_densities_directory=install_paths.atomic_densities_directory,
        )
        if updated_job_control_text == job_control_text:
            return
        job_control_path.write_text(
            updated_job_control_text, encoding=DEFAULT_TEXT_ENCODING
        )

    @staticmethod
    def _missing_expected_output_filenames(run_directory: Path) -> tuple[str, ...]:
        return tuple(
            filename
            for filename in GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES
            if not (run_directory / filename).is_file()
        )

    def run_chargemol_workflow(  # noqa: D102
        self,
        *,
        job_control_directory: Path | str,
        structure: PMG_Structure,
        properties: SinglePointProperties,
    ) -> GpawChargemolTaskDocument:
        if structure != properties.structure:
            msg = "GPAW Chargemol requires matching structure and properties.structure."
            raise ValueError(msg)
        job_control_directory = Path(job_control_directory)
        run_info = self._run_chargemol(job_control_directory=job_control_directory)
        if not run_info.exit_code_ok:
            error_log = (
                run_info.error_message
                or PROCESS_FAILURE_MESSAGE_TEMPLATE.format(
                    process_name=CHARGEMOL_PROCESS_NAME, exit_code=run_info.exit_code
                )
            )
            return GpawChargemolTaskDocument(
                dir_name=job_control_directory.as_posix(),
                errors=error_log,
                properties=None,
                structure=None,
                chargemol_run_info=None,
            )

        missing_output_filenames = self._missing_expected_output_filenames(
            run_directory=job_control_directory
        )
        if missing_output_filenames:
            missing_filenames = ", ".join(missing_output_filenames)
            error_log = CHARGEMOL_MISSING_OUTPUT_FILES_MESSAGE_TEMPLATE.format(
                missing_filenames=missing_filenames
            )
            return GpawChargemolTaskDocument(
                dir_name=job_control_directory.as_posix(),
                errors=error_log,
                properties=None,
                structure=None,
                chargemol_run_info=None,
            )
        try:
            net_atomic_charges = ChargemolNetAtomicChargesParser(
                chargemol_net_atomic_charges_path=(
                    job_control_directory / CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME
                )
            ).get_chargemol_atomic_charges()
            properties_with_charges = (
                attach_chargemol_net_atomic_charges_to_single_point_properties(
                    properties=properties, net_atomic_charges=net_atomic_charges
                )
            )
        except ValueError as exc:
            return GpawChargemolTaskDocument(
                dir_name=job_control_directory.as_posix(),
                errors=str(exc),
                properties=None,
                structure=None,
                chargemol_run_info=None,
            )
        return GpawChargemolTaskDocument(
            dir_name=job_control_directory.as_posix(),
            errors=EMPTY_ERROR_LOG,
            properties=properties_with_charges,
            structure=properties_with_charges.structure,
            chargemol_run_info=run_info,
        )

    @job(output_schema=GpawChargemolTaskDocument)
    def _run(
        self,
        job_control_directory: Path | str,
        structure: PMG_Structure,
        properties: SinglePointProperties,
    ) -> GpawChargemolTaskDocument:
        return self.run_chargemol_workflow(
            job_control_directory=job_control_directory,
            structure=structure,
            properties=properties,
        )

    def make(  # noqa: D102
        self,
        *,
        job_control_directory: Path,
        structure: PMG_Structure,
        properties: SinglePointProperties,
    ) -> Job:
        job = self._run(
            job_control_directory=job_control_directory,
            structure=structure,
            properties=properties,
        )
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job
