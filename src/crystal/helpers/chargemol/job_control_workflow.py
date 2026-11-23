"""Workflow helpers for writing Chargemol job_control.txt inputs."""

from __future__ import annotations

from pathlib import Path

from crystal.constants.chargemol import CHARGEMOL_JOB_CONTROL_FILENAME
from crystal.schemas.chargemol import (
    ChargemolInstallPathsConfigLoader,
    ChargemolJobControlInputBuilder,
    ChargemolJobControlSettings,
    QePpToChargemolWorkflowInputs,
)


class ChargemolJobControlWriter:
    """Write Chargemol job_control.txt inputs for QE pp.x workflows."""

    def __init__(self, workflow_inputs: QePpToChargemolWorkflowInputs) -> None:  # noqa: D107
        self.workflow_inputs = workflow_inputs

    @property
    def job_control_text(self) -> str:
        """Return job_control.txt contents using discovered Chargemol paths."""
        install_paths = ChargemolInstallPathsConfigLoader.discover()
        settings = ChargemolJobControlSettings.from_qe_pp_inputs(
            cube_output_directory=self.workflow_inputs.cube_output_directory,
            input_settings=self.workflow_inputs.input_settings,
            plot_settings=self.workflow_inputs.plot_settings,
            atomic_densities_directory=install_paths.atomic_densities_directory,
            pseudopotential_config=self.workflow_inputs.pseudopotential_config,
        )
        return ChargemolJobControlInputBuilder(settings=settings).job_control_text

    def write(self, job_control_directory: Path) -> Path:
        """Write job_control.txt to the provided directory."""
        job_control_path = job_control_directory / CHARGEMOL_JOB_CONTROL_FILENAME
        job_control_path.write_text(self.job_control_text)
        return job_control_path
