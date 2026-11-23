"""Create slabs from relaxed bulk structures for validation."""

from __future__ import annotations

from jobflow.core.job import job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.schemas.validation_dataset import (
    ValidationDatasetSlabGenerationSettings,
    ValidationDatasetSlabPreparationProperties,
)
from crystal.structure.slabs.builder import Slab, SlabFactory


def generate_validation_dataset_slab_structures(
    *,
    structure: PMG_Structure,
    slab_generation_settings: ValidationDatasetSlabGenerationSettings,
) -> list[Slab]:
    """Generate eligible slabs for every configured signed Miller index."""
    generated_slabs: list[Slab] = []
    for miller_index in slab_generation_settings.miller_indices:
        slab_factory = SlabFactory(
            structure=structure,
            supercell=slab_generation_settings.supercell,
            vacuum_thickness_in_layers=slab_generation_settings.vacuum_thickness_in_layers,
            surface_miller_indices=miller_index,
            orthogonalize_slabs=slab_generation_settings.orthogonalize_slabs,
            reduce_slabs_to_primitive=slab_generation_settings.reduce_slabs_to_primitive,
        )
        for generation_attempt_index, slab in enumerate(slab_factory.generate_slabs()):
            if (
                generation_attempt_index
                >= slab_generation_settings.maximum_generation_attempts
            ):
                break
            if len(slab) <= slab_generation_settings.maximum_atoms:
                generated_slabs.append(slab)
    return generated_slabs


@job(**{STRUCTURE_BLOB_STORE_NAME: [["output", "slabs"]]})  # type: ignore[call-overload]
def generate_validation_dataset_slab_preparation_properties(
    *,
    source_structure_id: str,
    structure: PMG_Structure,
    slab_generation_settings: ValidationDatasetSlabGenerationSettings,
) -> ValidationDatasetSlabPreparationProperties:
    """Build slabs for one relaxed bulk structure.

    This is a workflow step because the generated structures are passed to later
    stages through the configured data store.
    """
    return ValidationDatasetSlabPreparationProperties(
        source_structure_id=source_structure_id,
        slabs=generate_validation_dataset_slab_structures(
            structure=structure, slab_generation_settings=slab_generation_settings
        ),
    )
