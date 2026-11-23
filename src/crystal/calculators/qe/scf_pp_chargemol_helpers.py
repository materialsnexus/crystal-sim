"""Helpers for QE SCF -> pp.x -> Chargemol workflows."""

from __future__ import annotations

from pathlib import Path

from crystal.constants.process import PROCESS_FAILURE_MESSAGE_TEMPLATE
from crystal.schemas.process import ProcessRunInfo
from crystal.schemas.qe import QeScfPpChargemolTaskDocument
from crystal.schemas.qe_pp import (
    DEFAULT_QE_PP_PREFIX,
    QePpInputSettings,
    QePpPlotSettings,
    QeScfPpChargemolSimulationSettings,
)


def build_qe_pp_input_and_plot_settings_for_run_directory(  # noqa: D103
    *, run_directory: Path, simulation_settings: QeScfPpChargemolSimulationSettings
) -> tuple[QePpInputSettings, QePpPlotSettings]:
    if simulation_settings.qe_pp_input_settings is None:
        input_settings = QePpInputSettings(
            prefix=DEFAULT_QE_PP_PREFIX, outdir=run_directory.as_posix()
        )
    else:
        input_settings = QePpInputSettings(
            prefix=simulation_settings.qe_pp_input_settings.prefix,
            outdir=run_directory.as_posix(),
        )
    plot_settings = simulation_settings.qe_pp_plot_settings or QePpPlotSettings()
    return input_settings, plot_settings


def build_error_log_from_process_run_info(  # noqa: D103
    *, run_info: ProcessRunInfo, process_name: str
) -> str:
    return run_info.error_message or PROCESS_FAILURE_MESSAGE_TEMPLATE.format(
        process_name=process_name, exit_code=run_info.exit_code
    )


def build_failed_qe_scf_pp_chargemol_task_document(  # noqa: D103
    *, run_directory: Path, error_log: str
) -> QeScfPpChargemolTaskDocument:
    return QeScfPpChargemolTaskDocument(
        dir_name=run_directory.as_posix(),
        errors=error_log,
        properties=None,
        structure=None,
        pp_run_info=None,
        chargemol_run_info=None,
    )
