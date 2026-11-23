"""Fireworks/jobstore path constants."""

FIREWORKS_DIRNAME = ".fireworks"
JOBSTORE_DIRNAME = ".jobstore"
LOCAL_DIRNAME = "local"
SLURM_DIRNAME = "slurm"

LOCAL_LAUNCHPAD_YAML_NAME = "local_launchpad.yaml"
LOCAL_FWORKER_YAML_NAME = "my_fworker.yaml"
LOCAL_JOBSTORE_YAML_NAME = "jobstore.yaml"

SLURM_LAUNCHPAD_YAML_NAME = "slurm_launchpad.yaml"
SLURM_FWORKER_YAML_NAME = "my_fworker.yaml"
SLURM_QADAPTER_YAML_NAME = "my_qadapter.yaml"
SLURM_JOBSTORE_YAML_NAME = "jobstore.yaml"

QUEUEADAPTER_SPEC_KEY = "_queueadapter"
QUEUEADAPTER_NTASKS_KEY = "ntasks"
QUEUEADAPTER_GRES_KEY = "gres"
QUEUEADAPTER_MEM_KEY = "mem"
QUEUEADAPTER_WALLTIME_KEY = "walltime"
QUEUEADAPTER_SIGNAL_KEY = "signal"

SLURM_MEMORY_DECIMAL_PLACES = 2
SLURM_MEMORY_GB_SUFFIX = "G"
MINUTES_PER_HOUR = 60
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = MINUTES_PER_HOUR * SECONDS_PER_MINUTE
SLURM_WALLTIME_FORMAT = "{hours:02d}:{minutes:02d}:{seconds:02d}"
SLURM_TERM_SIGNAL_FORMAT = "TERM@{seconds}"

QUEUEADAPTER_NEGATIVE_WALLTIME_ERROR = (
    "maximum_walltime_in_minutes must be non-negative"
)
QUEUEADAPTER_NEGATIVE_MEMORY_ERROR = "total_memory_in_gb must be non-negative"
