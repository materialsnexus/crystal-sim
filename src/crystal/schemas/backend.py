# noqa: D100
from enum import StrEnum, auto


class RelaxBackend(StrEnum):  # noqa: D101
    qe = auto()
    mlip = auto()
    gpaw = auto()


class SinglePointBackend(StrEnum):  # noqa: D101
    qe = auto()
    mlip = auto()
    gpaw = auto()
