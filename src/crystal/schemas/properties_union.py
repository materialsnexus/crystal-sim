# noqa: D100
from typing import Annotated, TypeAlias

from pydantic import Discriminator, TypeAdapter

from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationProperties,
)
from crystal.schemas.md.properties import MdEquilibrationProperties
from crystal.schemas.properties import (
    CohesiveEnergyFilteringProperties,
    RelaxProperties,
    SinglePointProperties,
    SinglePointStructureFilteringProperties,
    StructureGenerationProperties,
)
from crystal.schemas.structure_augmentation import (
    StructureAugmentationAndFilteringProperties,
)

PropertiesUnion: TypeAlias = (
    InitialStructureGenerationProperties
    | SinglePointProperties
    | RelaxProperties
    | StructureGenerationProperties
    | SinglePointStructureFilteringProperties
    | CohesiveEnergyFilteringProperties
    | StructureAugmentationAndFilteringProperties
    | MdEquilibrationProperties
)
PropertiesDiscriminatedUnion = Annotated[PropertiesUnion, Discriminator("type")]
PropertiesUnionAdapter: TypeAdapter[PropertiesUnion] = TypeAdapter(PropertiesUnion)
