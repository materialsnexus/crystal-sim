from __future__ import annotations

from crystal.constants.chargemol import CHARGEMOL_EXECUTABLE_NAME
from crystal.constants.process import PROCESS_EXIT_CODE_SUCCESS
from crystal.schemas.process import ProcessRunInfo


def build_successful_process_run_info_for_test(
    *, command_arguments: list[str] | None = None
) -> ProcessRunInfo:
    return ProcessRunInfo(
        command_arguments=command_arguments or [CHARGEMOL_EXECUTABLE_NAME],
        exit_code=PROCESS_EXIT_CODE_SUCCESS,
        standard_output_file=None,
        standard_error_file=None,
        error_message=None,
    )
