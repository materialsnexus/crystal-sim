"""QE SCF → pp.x → Chargemol maker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jobflow.core.job import Job, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.qe.qe_pp_runner import QePpChargeDensityRunner
from crystal.calculators.qe.qe_runner import run_qe_scf
from crystal.calculators.qe.scf_pp_chargemol_helpers import (
    build_error_log_from_process_run_info,
    build_failed_qe_scf_pp_chargemol_task_document,
    build_qe_pp_input_and_plot_settings_for_run_directory,
)
from crystal.constants.chargemol import (
    CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME,
    CHARGEMOL_PROCESS_NAME,
)
from crystal.constants.process import EMPTY_ERROR_LOG
from crystal.constants.qe_pp import QE_PP_PROCESS_NAME
from crystal.helpers.chargemol.charge_parser import (
    ChargemolNetAtomicChargesParser,
    attach_chargemol_net_atomic_charges_to_single_point_properties,
)
from crystal.helpers.chargemol.chargemol_runner import ChargemolRunner
from crystal.schemas.base import BaseMaker
from crystal.schemas.chargemol import QePpToChargemolWorkflowInputs
from crystal.schemas.maker_names import MakerName
from crystal.schemas.qe import QeScfPpChargemolTaskDocument
from crystal.schemas.qe_pp import QeScfPpChargemolSimulationSettings


@dataclass(kw_only=True)
class QeScfPpChargemolMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.qe_chargemol_integration
    simulation_settings: QeScfPpChargemolSimulationSettings

    @job(output_schema=QeScfPpChargemolTaskDocument)
    def _run(self, structure: PMG_Structure) -> QeScfPpChargemolTaskDocument:
        run_directory = Path.cwd()
        try:
            properties = run_qe_scf(
                structure=structure,
                simulation_settings=self.simulation_settings.qe_scf_simulation_settings,
                number_of_processes=self.executor_settings.number_of_processes,
            )
            (input_settings, plot_settings) = (
                build_qe_pp_input_and_plot_settings_for_run_directory(
                    run_directory=run_directory,
                    simulation_settings=self.simulation_settings,
                )
            )
            pp_run_info = QePpChargeDensityRunner(
                run_directory=run_directory,
                input_settings=input_settings,
                plot_settings=plot_settings,
            ).run()
            if pp_run_info.exit_code != 0:
                error_log = build_error_log_from_process_run_info(
                    run_info=pp_run_info, process_name=QE_PP_PROCESS_NAME
                )
                return build_failed_qe_scf_pp_chargemol_task_document(
                    run_directory=run_directory, error_log=error_log
                )
            chargemol_inputs = QePpToChargemolWorkflowInputs(
                cube_output_directory=run_directory,
                input_settings=input_settings,
                plot_settings=plot_settings,
                pseudopotential_config=(
                    self.simulation_settings.qe_scf_simulation_settings.pseudopotential_config
                ),
            )
            chargemol_run_info = ChargemolRunner(
                run_directory=run_directory, workflow_inputs=chargemol_inputs
            ).run()
            if chargemol_run_info.exit_code != 0:
                error_log = build_error_log_from_process_run_info(
                    run_info=chargemol_run_info, process_name=CHARGEMOL_PROCESS_NAME
                )
                return build_failed_qe_scf_pp_chargemol_task_document(
                    run_directory=run_directory, error_log=error_log
                )
            net_atomic_charges = ChargemolNetAtomicChargesParser(
                chargemol_net_atomic_charges_path=(
                    run_directory / CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME
                )
            ).get_chargemol_atomic_charges()
            properties_with_charges = (
                attach_chargemol_net_atomic_charges_to_single_point_properties(
                    properties=properties, net_atomic_charges=net_atomic_charges
                )
            )
            return QeScfPpChargemolTaskDocument(
                dir_name=run_directory.as_posix(),
                errors=EMPTY_ERROR_LOG,
                properties=properties_with_charges,
                structure=properties_with_charges.structure,
                pp_run_info=pp_run_info,
                chargemol_run_info=chargemol_run_info,
            )
        except (RuntimeError, OSError, ValueError, FileNotFoundError) as exc:
            return build_failed_qe_scf_pp_chargemol_task_document(
                run_directory=run_directory, error_log=str(exc)
            )

    def make(self, structure: PMG_Structure) -> Job:  # noqa: D102
        job = self._run(structure=structure)
        job.update_metadata(self.slurm_settings.to_queueadapter_spec(), dynamic=False)
        return job
