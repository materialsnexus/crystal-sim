# noqa: D100
from enum import StrEnum, auto

from pydantic import BaseModel

from crystal.constants.fireworks import QUEUEADAPTER_SPEC_KEY


class PartitionType(StrEnum):  # noqa: D101
    serial = auto()
    all = auto()
    gpu = auto()


class QosType(StrEnum):  # noqa: D101
    regular = auto()


class SlurmSettings(BaseModel):  # noqa: D101
    queue: PartitionType | None = None
    nodes: int | None = None
    ntasks: int | None = None
    ntasks_per_node: int | None = None
    ntasks_per_core: int | None = None
    cpus_per_task: int | None = None
    walltime: str | None = None
    mem: str | None = None
    qos: QosType | None = None
    account: str | None = None
    job_name: str | None = None
    pre_rocket: str | None = None
    post_rocket: str | None = None
    rocket_launch: str | None = None
    gres: str | None = None
    signal: str | None = None

    def to_queueadapter_spec(self) -> dict[str, dict[str, int | str]]:  # noqa: D102
        queueadapter_spec = self.model_dump(exclude_none=True)
        return {QUEUEADAPTER_SPEC_KEY: queueadapter_spec}
