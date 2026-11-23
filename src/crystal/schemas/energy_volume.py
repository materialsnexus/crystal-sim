# noqa: D100
from typing import Self

from pymatgen.core.structure import Structure as PMG_Structure

from crystal.schemas.properties import SinglePointProperties
from crystal.schemas.serialization import MSONableBaseModel
from crystal.schemas.single_point import SinglePointTaskDocument


class EnergyVolumeOutput(MSONableBaseModel):  # noqa: D101
    strained_structure: PMG_Structure
    energy: float
    forces: list[list[float]]
    stresses: list[float]
    charges: list[float] | None = None

    @property
    def volume(self) -> float:  # noqa: D102
        return self.strained_structure.volume

    @classmethod
    def from_single_point_task_doc(cls, task_document: SinglePointTaskDocument) -> Self:  # noqa: D102
        properties = task_document.properties
        assert isinstance(properties, SinglePointProperties)
        return cls(
            strained_structure=properties.structure,
            energy=properties.energy,
            forces=properties.forces,
            stresses=properties.stress,
            charges=properties.charges,
        )
