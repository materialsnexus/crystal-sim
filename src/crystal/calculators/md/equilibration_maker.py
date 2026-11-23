"""MLIP MD equilibration maker."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from jobflow.core.job import Job, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.md.md_runner import MlipMdRunner
from crystal.helpers.file import count_args_not_none
from crystal.helpers.md_plot_helpers import plot_time_series_with_distribution
from crystal.helpers.md_run_helpers import run_equilibration_check
from crystal.helpers.mlip_helpers import configure_default_torch_device
from crystal.schemas.base import BaseMaker
from crystal.schemas.maker_names import MakerName
from crystal.schemas.md.data import MdSeries
from crystal.schemas.md.mlip_md_equilibration import (
    MlipMdEquilibrationSimulationSettings,
    MlipMdEquilibrationTaskDocument,
)
from crystal.schemas.md.properties import (
    MdEquilibrationPlots,
    MdEquilibrationProperties,
)

logger = logging.getLogger(__name__)

ENERGY_PLOT_FILENAME = "energy_vs_time.png"
TEMPERATURE_PLOT_FILENAME = "temperature_vs_time.png"
DENSITY_PLOT_FILENAME = "density_vs_time.png"


def _time_averaging_fraction(
    series: MdSeries, time_averaging_window_ps: float
) -> float:
    time_span_ps = series.time_ps[-1] - series.time_ps[0]
    if time_span_ps <= 0.0:
        return 1.0
    return min(1.0, time_averaging_window_ps / time_span_ps)


@dataclass(kw_only=True)
class MlipMdEquilibrationMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.mlip_md_equilibration
    simulation_settings: MlipMdEquilibrationSimulationSettings

    def _generate_energy_plot(
        self, energy_series: MdSeries, output_directory: Path
    ) -> Path:
        energy_plot_path = output_directory / ENERGY_PLOT_FILENAME
        plot_time_series_with_distribution(
            series=energy_series,
            output_path=energy_plot_path,
            y_label="Energy [eV]",
            averaging_fraction=_time_averaging_fraction(
                energy_series, self.simulation_settings.time_averaging_window
            ),
        )
        return energy_plot_path

    def _generate_temperature_plot(
        self, temperature_series: MdSeries, output_directory: Path
    ) -> Path:
        temperature_plot_path = output_directory / TEMPERATURE_PLOT_FILENAME
        plot_time_series_with_distribution(
            series=temperature_series,
            output_path=temperature_plot_path,
            y_label="Temperature [K]",
            averaging_fraction=_time_averaging_fraction(
                temperature_series, self.simulation_settings.time_averaging_window
            ),
        )
        return temperature_plot_path

    def _generate_density_plot(
        self, density_series: MdSeries, output_directory: Path
    ) -> Path:
        density_plot_path = output_directory / DENSITY_PLOT_FILENAME
        plot_time_series_with_distribution(
            series=density_series,
            output_path=density_plot_path,
            y_label=r"Density [g $cm^{-3}$]",
            averaging_fraction=_time_averaging_fraction(
                density_series, self.simulation_settings.time_averaging_window
            ),
        )
        return density_plot_path

    @job(output_schema=MlipMdEquilibrationTaskDocument)
    def _run(
        self,
        structure: PMG_Structure | None = None,
        trajectory_path: Path | None = None,
    ) -> MlipMdEquilibrationTaskDocument:
        logger.info(
            "Running MLIP MD equilibration calculation for structure/trajectory with ensemble: %s",
            self.simulation_settings.run_settings.dynamics.ensemble.value,
        )
        try:
            configure_default_torch_device(self.executor_settings.number_of_gpus)
            mlip_md_runner = MlipMdRunner(
                run_settings=self.simulation_settings.run_settings,
                structure=structure,
                trajectory_path_for_restart=trajectory_path,
            )

            stage_result = mlip_md_runner.run()

            check_results = [
                run_equilibration_check(
                    md_stage_result=stage_result,
                    time_averaging_window=self.simulation_settings.time_averaging_window,
                    check=check,
                )
                for check in self.simulation_settings.checks
            ]

            output_directory = mlip_md_runner.output_directory
            plots = MdEquilibrationPlots(
                energy_plot_path=self._generate_energy_plot(
                    stage_result.energy_series, output_directory=output_directory
                ).as_posix(),
                temperature_plot_path=self._generate_temperature_plot(
                    stage_result.temperature_series, output_directory=output_directory
                ).as_posix(),
                density_plot_path=self._generate_density_plot(
                    stage_result.density_series, output_directory=output_directory
                ).as_posix(),
            )

            properties = MdEquilibrationProperties(
                ensemble=stage_result.ensemble,
                md_stage_result=stage_result,
                check_results=check_results,
                time_averaging_window=self.simulation_settings.time_averaging_window,
                plots=plots,
            )

            errors = ""
        except Exception as exc:
            properties = None
            errors = str(exc)
            logger.exception("MLIP MD equilibration calculation failed.")

        task_document = MlipMdEquilibrationTaskDocument(
            dir_name=Path.cwd().as_posix(), properties=properties, errors=errors
        )
        logger.info("Finished MLIP MD equilibration calculation")
        return task_document

    def make(  # noqa: D102
        self,
        structure: PMG_Structure | None = None,
        trajectory_path: Path | None = None,
    ) -> Job:
        if count_args_not_none(structure, trajectory_path) != 1:
            err_msg = "Exactly one of structure or trajectory path should be provided"
            raise ValueError(err_msg)
        job = self._run(structure=structure, trajectory_path=trajectory_path)
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job
