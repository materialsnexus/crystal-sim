"""Structure augmentation and geometric filtering functions."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.structure_generators import STAGE_LABEL_PROPERTY_KEY
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.helpers.structure import (
    get_maximum_box_aspect_ratio,
    get_minimum_interatomic_distance,
    passes_geometric_structure_filter,
)
from crystal.schemas.filtering import GeometricStructureFilteringSettings
from crystal.schemas.structure_augmentation import (
    StructureAugmentationAndFilteringProperties,
    StructureAugmentationAndFilteringResult,
    StructureAugmentationAndFilteringStatistics,
    StructureAugmentationRootFailureSummary,
    StructureGenerationSource,
    StructureGenerationSourceStatistics,
)
from crystal.schemas.structure_generators import (
    StretchCellSettings,
    StructureAugmentationAndFilteringSettings,
)
from crystal.structure.generators.perturbation.rattle import rattle_atom_positions
from crystal.structure.generators.perturbation.stretch import stretch_cell
from crystal.structure.generators.randomness import spawn_random_seeds

logger = logging.getLogger(__name__)


@dataclass
class SourceStatisticsAccumulator:
    """Accumulate candidate counts for one generation stage."""

    considered: int = 0
    passed: int = 0
    filtered: int = 0

    def add(self, *, passed: bool) -> None:
        """Record one candidate result."""
        self.considered += 1
        if passed:
            self.passed += 1
        else:
            self.filtered += 1

    def to_model(self) -> StructureGenerationSourceStatistics:
        """Return the validated statistics model."""
        return StructureGenerationSourceStatistics(
            considered=self.considered, passed=self.passed, filtered=self.filtered
        )


@dataclass
class RootFailureAccumulator:
    """Aggregate root failures by source and exception type."""

    _summaries: dict[
        tuple[StructureGenerationSource, str], StructureAugmentationRootFailureSummary
    ] = field(default_factory=dict)

    def add(self, *, source: StructureGenerationSource, error: BaseException) -> None:
        """Record a root failure while retaining its first diagnostic."""
        exception_type = type(error).__name__
        key = (source, exception_type)
        summary = self._summaries.get(key)
        if summary is None:
            self._summaries[key] = StructureAugmentationRootFailureSummary(
                source=source,
                exception_type=exception_type,
                count=1,
                first_error_message=get_diagnostic_exception_message(error),
            )
            return
        summary.count += 1

    @property
    def summaries(self) -> tuple[StructureAugmentationRootFailureSummary, ...]:
        """Return summaries in first-occurrence order."""
        return tuple(self._summaries.values())

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return compact user-facing warnings for every failure group."""
        return tuple(
            (
                f"{summary.source.value} / {summary.exception_type}: "
                f"{summary.count} root{'s' if summary.count != 1 else ''}; "
                f"{summary.first_error_message}"
            )
            for summary in self.summaries
        )


@dataclass
class StructureCandidateAccumulator:
    """Filter candidates and retain survivors grouped by generation stage."""

    geometric_filtering_settings: GeometricStructureFilteringSettings
    _statistics: dict[StructureGenerationSource, SourceStatisticsAccumulator] = field(
        init=False
    )
    _survivors: dict[StructureGenerationSource, list[PMG_Structure]] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize accumulators for every generation stage."""
        self._statistics = {
            source: SourceStatisticsAccumulator()
            for source in StructureGenerationSource
        }
        self._survivors = {source: [] for source in StructureGenerationSource}

    def add(
        self, *, source: StructureGenerationSource, structure: PMG_Structure
    ) -> None:
        """Filter one candidate and retain it when it passes."""
        filtering_settings = self.geometric_filtering_settings
        passes_filter = passes_geometric_structure_filter(
            minimum_interatomic_distance=get_minimum_interatomic_distance(structure),
            maximum_box_aspect_ratio=get_maximum_box_aspect_ratio(structure),
            minimum_allowed_interatomic_distance=filtering_settings.minimum_interatomic_distance,
            maximum_allowed_box_aspect_ratio=filtering_settings.maximum_box_aspect_ratio,
        )
        self._statistics[source].add(passed=passes_filter)
        if passes_filter:
            labelled_structure = structure.copy()
            labelled_structure.properties = dict(structure.properties)
            labelled_structure.properties[STAGE_LABEL_PROPERTY_KEY] = source.value
            self._survivors[source].append(labelled_structure)

    def build_properties(
        self,
        *,
        root_failure_summaries: tuple[StructureAugmentationRootFailureSummary, ...],
    ) -> StructureAugmentationAndFilteringProperties:
        """Build the filtered structure properties in stage order."""
        source_statistics = {
            source: accumulator.to_model()
            for source, accumulator in self._statistics.items()
        }
        return StructureAugmentationAndFilteringProperties(
            structures=[
                structure
                for source in StructureGenerationSource
                for structure in self._survivors[source]
            ],
            statistics=StructureAugmentationAndFilteringStatistics.from_source_statistics(
                source_statistics, root_failure_summaries=root_failure_summaries
            ),
        )


def _iter_rattled_variants(
    structure: PMG_Structure, settings: StructureAugmentationAndFilteringSettings
) -> Iterator[PMG_Structure]:
    rattle_settings = settings.rattle_settings
    for variant_random_seed in spawn_random_seeds(
        rattle_settings.random_seed, settings.number_of_rattled_variants
    ):
        yield rattle_atom_positions(
            structure=structure,
            random_seed=variant_random_seed,
            noise_scaling_factor=rattle_settings.noise_scaling_factor,
            displacement_mode=rattle_settings.displacement_mode,
        )


def _iter_stretched_variants(
    structure: PMG_Structure,
    *,
    stretch_settings: StretchCellSettings,
    number_of_variants: int,
) -> Iterator[PMG_Structure]:
    """Yield valid stretched variants and fail if none can be generated."""
    invalid_variant_count = 0
    for variant_random_seed in spawn_random_seeds(
        stretch_settings.random_seed, number_of_variants
    ):
        variant = stretch_cell(
            structure=structure,
            random_seed=variant_random_seed,
            max_normal_strain_fraction=stretch_settings.max_normal_strain_fraction,
            max_shear_strain_fraction=stretch_settings.max_shear_strain_fraction,
        )
        if variant is not None:
            yield variant
        else:
            invalid_variant_count += 1

    if invalid_variant_count:
        logger.warning(
            "Skipped %d invalid stretched variants out of %d requested variants.",
            invalid_variant_count,
            number_of_variants,
        )
    if invalid_variant_count == number_of_variants:
        msg = "All random deformation matrices were invalid; 0 variants generated successfully."
        raise RuntimeError(msg)


def iter_root_structure_candidates(
    *,
    relaxed_structure: PMG_Structure,
    settings: StructureAugmentationAndFilteringSettings,
) -> Iterator[tuple[StructureGenerationSource, PMG_Structure]]:
    """Yield one relaxed root and its transient augmented candidates."""
    yield StructureGenerationSource.mlip_relax, relaxed_structure

    for rattled_structure in _iter_rattled_variants(
        structure=relaxed_structure, settings=settings
    ):
        yield StructureGenerationSource.rattle_after_mlip_relax, rattled_structure
        for stretched_structure in _iter_stretched_variants(
            structure=rattled_structure,
            stretch_settings=settings.stretch_after_rattle_settings,
            number_of_variants=settings.number_of_stretched_variants_after_rattle,
        ):
            yield StructureGenerationSource.stretch_after_rattle, stretched_structure

    for stretched_structure in _iter_stretched_variants(
        structure=relaxed_structure,
        stretch_settings=settings.stretch_after_mlip_relax_settings,
        number_of_variants=settings.number_of_stretched_variants,
    ):
        yield StructureGenerationSource.stretch_after_mlip_relax, stretched_structure


def augment_and_filter_structures(
    *,
    initial_structures: Sequence[PMG_Structure],
    relaxed_structures: Sequence[PMG_Structure],
    settings: StructureAugmentationAndFilteringSettings,
) -> StructureAugmentationAndFilteringResult:
    """Augment and filter structures while isolating root failures."""
    accumulator = StructureCandidateAccumulator(
        geometric_filtering_settings=settings.geometric_filtering_settings
    )
    root_failure_accumulator = RootFailureAccumulator()

    _filter_initial_structures(
        structures=initial_structures,
        accumulator=accumulator,
        root_failure_accumulator=root_failure_accumulator,
    )
    _augment_and_filter_relaxed_structures(
        structures=relaxed_structures,
        accumulator=accumulator,
        settings=settings,
        root_failure_accumulator=root_failure_accumulator,
    )

    return StructureAugmentationAndFilteringResult(
        properties=accumulator.build_properties(
            root_failure_summaries=root_failure_accumulator.summaries
        ),
        warnings=root_failure_accumulator.warnings,
    )


def _filter_initial_structures(
    *,
    structures: Sequence[PMG_Structure],
    accumulator: StructureCandidateAccumulator,
    root_failure_accumulator: RootFailureAccumulator,
) -> None:
    """Filter initial structures while preserving per-structure failures."""
    for structure_index, structure in enumerate(structures):
        try:
            accumulator.add(
                source=StructureGenerationSource.initial, structure=structure
            )
        except Exception as exc:
            root_failure_accumulator.add(
                source=StructureGenerationSource.initial, error=exc
            )
            logger.exception(
                "Initial structure %d augmentation failed.", structure_index
            )


def _augment_and_filter_relaxed_structures(
    *,
    structures: Sequence[PMG_Structure],
    accumulator: StructureCandidateAccumulator,
    settings: StructureAugmentationAndFilteringSettings,
    root_failure_accumulator: RootFailureAccumulator,
) -> None:
    """Augment and filter relaxed structures while preserving root failures."""
    for structure_index, structure in enumerate(structures):
        try:
            for source, candidate in iter_root_structure_candidates(
                relaxed_structure=structure, settings=settings
            ):
                accumulator.add(source=source, structure=candidate)
        except Exception as exc:
            root_failure_accumulator.add(
                source=StructureGenerationSource.mlip_relax, error=exc
            )
            logger.exception(
                "MLIP-relaxed structure %d augmentation failed.", structure_index
            )
