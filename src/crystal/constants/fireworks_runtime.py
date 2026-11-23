"""Runtime constants used by FireWorks runtime configuration helpers."""

from enum import StrEnum


class FireWorksRuntimeCommandFlag(StrEnum):  # noqa: D101
    RESERVE = "-r"
    FWORKER = "-w"
    LAUNCHPAD = "-l"
    QADAPTER = "-q"
    NLAUNCHES = "--nlaunches"


RUNTIME_FW_CONFIG_FILENAME = "FW_config.yaml"
FW_CONFIG_ENVIRONMENT_VARIABLE_NAME = "FW_CONFIG_FILE"
RUNTIME_LOCAL_LAUNCH_BINARY = "rlaunch"
RUNTIME_QUEUE_LAUNCH_BINARY = "qlaunch"
RUNTIME_LAUNCH_LOG_FILENAME = "runtime_launch.log"
RUNTIME_LAUNCH_PID_FILENAME = "runtime_launch.pid"
RUNTIME_MPI_ENVIRONMENT_QADAPTER_FILENAME = "runtime_mpi_environment_qadapter.yaml"
RUNTIME_QADAPTER_PRE_ROCKET_KEY = "pre_rocket"
RUNTIME_SOURCE_SHELL_COMMAND = "source"
RUNTIME_BASH_BINARY = "bash"
RUNTIME_BASH_LOGIN_COMMAND_FLAG = "-lc"
