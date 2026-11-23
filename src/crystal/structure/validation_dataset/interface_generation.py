"""Create interface structures from generated slabs for validation."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from itertools import islice

from jobflow.core.job import job

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.exceptions import NoInterfaceMatchException, OverlappingAtomsException
from crystal.schemas.validation_dataset import (
    ValidationDatasetInterfaceGenerationProperties,
    ValidationDatasetInterfaceGenerationSettings,
)
from crystal.structure.interfaces.builder import Interface, InterfaceFactory
from crystal.structure.interfaces.utils import compute_interface_gaps
from crystal.structure.slabs.builder import Slab

logger = logging.getLogger(__name__)


def _generate_validation_dataset_interface_variants(
    *,
    interface_factory: InterfaceFactory,
    top_slab: Slab,
    bottom_slab: Slab,
    gap: float,
    vacuum: float,
    offset: tuple[float, float],
    maximum_generation_attempts: int,
) -> Iterator[Interface]:
    """Yield usable candidates while tolerating invalid interface geometry.

    The interface factory can yield valid candidates before a later candidate
    raises an overlap error. Handling that error here preserves the candidates
    already yielded and lets the validation-dataset generation continue.
    """
    try:
        yield from islice(
            interface_factory.generate_interfaces(
                gap=gap, vacuum=vacuum, in_plane_offset=offset
            ),
            maximum_generation_attempts,
        )
    except OverlappingAtomsException as exc:
        logger.info(
            "Skipping invalid interface candidate for slab pair %s/%s at gap "
            "%s and offset %s: %s",
            top_slab.miller_index,
            bottom_slab.miller_index,
            gap,
            offset,
            exc,
        )


def generate_validation_dataset_interface_structures(
    *,
    top_slabs: Sequence[Slab],
    bottom_slabs: Sequence[Slab],
    interface_generation_settings: ValidationDatasetInterfaceGenerationSettings,
) -> list[Interface]:
    """Generate bounded gap/offset variants for every slab pair."""
    interface_match_settings = interface_generation_settings.match_settings
    interface_match_policy = interface_match_settings.interface_match_policy
    interface_assembly_settings = interface_generation_settings.assembly_settings

    generated_interfaces: list[Interface] = []
    for top_slab in top_slabs:
        for bottom_slab in bottom_slabs:
            try:
                interface_factory = InterfaceFactory(
                    top_slab=top_slab,
                    bottom_slab=bottom_slab,
                    interface_match_policy=interface_match_policy,
                )
                gap_values = compute_interface_gaps(
                    gaps=interface_assembly_settings.gaps,
                    multipliers=interface_assembly_settings.gap_multipliers,
                    estimated_gap=interface_factory.estimated_vacuum_gap_in_angstrom,
                )
                selected_match_indices: list[int] = []
                generated_interface_variants: dict[
                    tuple[float, tuple[float, float]], list[Interface]
                ] = {}
                for gap in gap_values:
                    for offset in interface_assembly_settings.offsets:
                        interface_variants = list(
                            _generate_validation_dataset_interface_variants(
                                interface_factory=interface_factory,
                                top_slab=top_slab,
                                bottom_slab=bottom_slab,
                                gap=gap,
                                vacuum=interface_assembly_settings.vacuum,
                                offset=offset,
                                maximum_generation_attempts=(
                                    interface_match_settings.maximum_generation_attempts
                                ),
                            )
                        )
                        generated_interface_variants[(gap, offset)] = interface_variants
                        if not selected_match_indices:
                            selected_match_indices = [
                                index
                                for index, interface in enumerate(interface_variants)
                                if len(interface)
                                <= interface_match_settings.maximum_atoms
                            ][: interface_match_settings.maximum_matches_per_slab_pair]

                for interface_variants in generated_interface_variants.values():
                    generated_interfaces.extend(
                        interface
                        for index, interface in enumerate(interface_variants)
                        if index in selected_match_indices
                        and len(interface) <= interface_match_settings.maximum_atoms
                    )
            except NoInterfaceMatchException as exc:
                logger.warning("%s", exc)
    return generated_interfaces


@job(**{STRUCTURE_BLOB_STORE_NAME: [["output", "interfaces"]]})  # type: ignore[call-overload]
def generate_validation_dataset_interface_properties(
    *,
    top_slabs: list[Slab],
    bottom_slabs: list[Slab],
    interface_generation_settings: ValidationDatasetInterfaceGenerationSettings,
) -> ValidationDatasetInterfaceGenerationProperties:
    """Build interface variants for the prepared slabs.

    This is a workflow step because the generated structures are passed to later
    stages through the configured data store.
    """
    return ValidationDatasetInterfaceGenerationProperties(
        interfaces=generate_validation_dataset_interface_structures(
            top_slabs=top_slabs,
            bottom_slabs=bottom_slabs,
            interface_generation_settings=interface_generation_settings,
        )
    )
