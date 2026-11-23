"""Contract tests for structure augmentation and geometric filtering."""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pytest
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.structure_generators import STAGE_LABEL_PROPERTY_KEY
from crystal.schemas.properties_union import PropertiesUnionAdapter
from crystal.schemas.structure_augmentation import (
    StructureAugmentationAndFilteringProperties,
    StructureAugmentationAndFilteringStatistics,
    StructureAugmentationRootFailureSummary,
    StructureGenerationSource,
    StructureGenerationSourceStatistics,
)
from crystal.schemas.structure_generators import (
    StructureAugmentationAndFilteringSettings,
    StructureAugmentationAndFilteringTaskDocument,
)
from crystal.structure.generators import augmentation_filtering
from crystal.structure.generators.augmentation_filtering import (
    iter_root_structure_candidates,
)
from crystal.structure.generators.perturbation.rattle import (
    RattleAtomDisplacementMode,
    rattle_atom_positions,
)
from tests.helpers.constants.structure import TEST_BCC_FE_STRUCTURE, TEST_SI_STRUCTURE
from tests.helpers.constants.structure_generators import (
    TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS,
    TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
)


def _source_statistics() -> StructureGenerationSourceStatistics:
    return StructureGenerationSourceStatistics(considered=3, passed=2, filtered=1)


def _augmentation_and_filtering_statistics() -> (
    StructureAugmentationAndFilteringStatistics
):
    return StructureAugmentationAndFilteringStatistics(
        initial=_source_statistics(),
        mlip_relax=_source_statistics(),
        rattle_after_mlip_relax=_source_statistics(),
        stretch_after_rattle=_source_statistics(),
        stretch_after_mlip_relax=_source_statistics(),
    )


def _labelled_structure(
    structure: PMG_Structure, source: StructureGenerationSource
) -> PMG_Structure:
    labelled_structure = structure.copy()
    labelled_structure.properties = dict(structure.properties)
    labelled_structure.properties[STAGE_LABEL_PROPERTY_KEY] = source.value
    return labelled_structure


def test_source_statistics_require_considered_structures_to_be_accounted_for() -> None:
    with pytest.raises(ValueError, match=r"passed.*filtered.*considered"):
        StructureGenerationSourceStatistics(considered=3, passed=1, filtered=1)


def test_augmentation_statistics_reject_extra_fields() -> None:
    statistics_data = _augmentation_and_filtering_statistics().model_dump()
    statistics_data["unexpected"] = 1

    with pytest.raises(ValueError, match="extra"):
        StructureAugmentationAndFilteringStatistics.model_validate(statistics_data)


def test_properties_reject_structures_without_generation_stage_labels() -> None:
    with pytest.raises(ValueError, match="generation-stage label"):
        StructureAugmentationAndFilteringProperties(
            structures=[TEST_SI_STRUCTURE],
            statistics=_augmentation_and_filtering_statistics(),
        )


def test_properties_reject_structures_with_invalid_generation_stage_labels() -> None:
    structure = TEST_SI_STRUCTURE.copy()
    structure.properties[STAGE_LABEL_PROPERTY_KEY] = "unknown stage"

    with pytest.raises(ValueError, match="generation-stage label"):
        StructureAugmentationAndFilteringProperties(
            structures=[structure], statistics=_augmentation_and_filtering_statistics()
        )


def test_properties_preserve_direct_sampling_structure_order() -> None:
    structures = [
        _labelled_structure(TEST_BCC_FE_STRUCTURE, StructureGenerationSource.initial),
        _labelled_structure(TEST_SI_STRUCTURE, StructureGenerationSource.mlip_relax),
    ]

    properties = StructureAugmentationAndFilteringProperties(
        structures=structures, statistics=_augmentation_and_filtering_statistics()
    )

    assert properties.structures == structures


def test_properties_union_discriminates_augmentation_and_filtering_properties() -> None:
    properties = StructureAugmentationAndFilteringProperties(
        structures=[
            _labelled_structure(TEST_SI_STRUCTURE, StructureGenerationSource.mlip_relax)
        ],
        statistics=_augmentation_and_filtering_statistics(),
    )

    parsed_properties = PropertiesUnionAdapter.validate_python(properties.model_dump())

    assert isinstance(parsed_properties, StructureAugmentationAndFilteringProperties)
    assert parsed_properties.statistics == properties.statistics


def test_properties_do_not_require_rejected_structures_or_filtering_metrics() -> None:
    properties = StructureAugmentationAndFilteringProperties(
        structures=[
            _labelled_structure(TEST_SI_STRUCTURE, StructureGenerationSource.mlip_relax)
        ],
        statistics=_augmentation_and_filtering_statistics(),
    )

    assert not hasattr(properties, "all_structures_with_metrics")
    assert not hasattr(properties, "filtering_mask")


def test_augmentation_and_filtering_does_not_mutate_input_properties() -> None:
    initial_structure = TEST_BCC_FE_STRUCTURE.copy()
    initial_structure.properties["input_marker"] = "initial"
    relaxed_structure = TEST_SI_STRUCTURE.copy()
    relaxed_structure.properties["input_marker"] = "relaxed"

    result = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[initial_structure],
        relaxed_structures=[relaxed_structure],
        settings=StructureAugmentationAndFilteringSettings(
            number_of_rattled_variants=1,
            number_of_stretched_variants=1,
            rattle_settings=TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        ),
    )

    assert result.warnings == ()
    assert initial_structure.properties == {"input_marker": "initial"}
    assert relaxed_structure.properties == {"input_marker": "relaxed"}


def test_task_document_exposes_filtered_structures_for_direct_sampling() -> None:
    properties = StructureAugmentationAndFilteringProperties(
        structures=[
            _labelled_structure(TEST_SI_STRUCTURE, StructureGenerationSource.mlip_relax)
        ],
        statistics=_augmentation_and_filtering_statistics(),
    )

    task_document = StructureAugmentationAndFilteringTaskDocument(properties=properties)

    assert task_document.errors == ""
    assert task_document.properties is properties
    assert task_document.properties.structures == properties.structures


def test_augmentation_and_filtering_preserves_current_source_order() -> None:
    result = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[TEST_BCC_FE_STRUCTURE],
        relaxed_structures=[TEST_SI_STRUCTURE],
        settings=StructureAugmentationAndFilteringSettings(
            number_of_rattled_variants=1,
            number_of_stretched_variants=1,
            rattle_settings=TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        ),
    )
    properties = result.properties

    assert [
        structure.properties[STAGE_LABEL_PROPERTY_KEY]
        for structure in properties.structures
    ] == [
        StructureGenerationSource.initial.value,
        StructureGenerationSource.mlip_relax.value,
        StructureGenerationSource.rattle_after_mlip_relax.value,
        StructureGenerationSource.stretch_after_rattle.value,
        StructureGenerationSource.stretch_after_mlip_relax.value,
    ]
    assert properties.statistics.initial.considered == 1
    assert properties.statistics.mlip_relax.considered == 1
    assert properties.statistics.rattle_after_mlip_relax.considered == 1
    assert properties.statistics.stretch_after_rattle.considered == 1
    assert properties.statistics.stretch_after_mlip_relax.considered == 1


def test_augmentation_and_filtering_is_deterministic() -> None:
    settings = StructureAugmentationAndFilteringSettings(
        number_of_rattled_variants=1,
        number_of_stretched_variants=1,
        rattle_settings=TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS,
        stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
    )

    first_result = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[TEST_BCC_FE_STRUCTURE],
        relaxed_structures=[TEST_SI_STRUCTURE],
        settings=settings,
    )
    second_result = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[TEST_BCC_FE_STRUCTURE],
        relaxed_structures=[TEST_SI_STRUCTURE],
        settings=settings,
    )

    assert first_result.warnings == second_result.warnings == ()
    assert first_result.properties.statistics == second_result.properties.statistics
    assert [
        structure.properties[STAGE_LABEL_PROPERTY_KEY]
        for structure in first_result.properties.structures
    ] == [
        structure.properties[STAGE_LABEL_PROPERTY_KEY]
        for structure in second_result.properties.structures
    ]
    for first_structure, second_structure in zip(
        first_result.properties.structures,
        second_result.properties.structures,
        strict=True,
    ):
        assert np.allclose(
            first_structure.lattice.matrix, second_structure.lattice.matrix
        )
        assert np.allclose(first_structure.frac_coords, second_structure.frac_coords)


def test_root_candidate_stream_generates_each_rattle_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rattle_calls = 0

    def _mock_rattle_atom_positions(
        structure: PMG_Structure,
        random_seed: int,
        noise_scaling_factor: float,
        displacement_mode: RattleAtomDisplacementMode,
    ) -> PMG_Structure:
        nonlocal rattle_calls
        _ = random_seed, noise_scaling_factor, displacement_mode
        rattle_calls += 1
        rattled_structure = structure.copy()
        rattled_structure.properties["candidate_kind"] = "rattle"
        return rattled_structure

    def _mock_stretch_cell(
        structure: PMG_Structure,
        random_seed: int,
        max_normal_strain_fraction: float,
        max_shear_strain_fraction: float,
    ) -> PMG_Structure:
        _ = random_seed, max_normal_strain_fraction, max_shear_strain_fraction
        stretched_structure = structure.copy()
        stretched_structure.properties["candidate_kind"] = "stretch"
        return stretched_structure

    monkeypatch.setattr(
        "crystal.structure.generators.augmentation_filtering.rattle_atom_positions",
        _mock_rattle_atom_positions,
    )
    monkeypatch.setattr(augmentation_filtering, "stretch_cell", _mock_stretch_cell)

    candidates = list(
        iter_root_structure_candidates(
            relaxed_structure=TEST_SI_STRUCTURE,
            settings=StructureAugmentationAndFilteringSettings(
                number_of_rattled_variants=2,
                number_of_stretched_variants=1,
                stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
                stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
            ),
        )
    )

    assert rattle_calls == 2
    assert [source for source, _ in candidates] == [
        StructureGenerationSource.mlip_relax,
        StructureGenerationSource.rattle_after_mlip_relax,
        StructureGenerationSource.stretch_after_rattle,
        StructureGenerationSource.rattle_after_mlip_relax,
        StructureGenerationSource.stretch_after_rattle,
        StructureGenerationSource.stretch_after_mlip_relax,
    ]


def test_augmentation_and_filtering_groups_sources_for_multiple_relaxed_structures() -> (
    None
):
    first_relaxed = TEST_SI_STRUCTURE.copy()
    first_relaxed.properties["root"] = "first"
    second_relaxed = TEST_SI_STRUCTURE.copy()
    second_relaxed.properties["root"] = "second"

    result = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[],
        relaxed_structures=[first_relaxed, second_relaxed],
        settings=StructureAugmentationAndFilteringSettings(
            number_of_rattled_variants=1,
            number_of_stretched_variants=1,
            stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        ),
    )
    properties = result.properties

    assert [
        (
            structure.properties[STAGE_LABEL_PROPERTY_KEY],
            structure.properties.get("root"),
        )
        for structure in properties.structures
    ] == [
        ("MLIP Relax", "first"),
        ("MLIP Relax", "second"),
        ("Rattle after MLIP Relax", "first"),
        ("Rattle after MLIP Relax", "second"),
        ("Stretch after Rattle", "first"),
        ("Stretch after Rattle", "second"),
        ("Stretch after MLIP Relax", "first"),
        ("Stretch after MLIP Relax", "second"),
    ]


def test_stretch_after_rattle_is_generated_before_final_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _mock_rattle_atom_positions(
        structure: PMG_Structure,
        random_seed: int,
        noise_scaling_factor: float,
        displacement_mode: RattleAtomDisplacementMode,
    ) -> PMG_Structure:
        _ = random_seed, noise_scaling_factor, displacement_mode
        rattled_structure = structure.copy()
        rattled_structure.properties["candidate_kind"] = "rattle"
        return rattled_structure

    def _mock_stretch_cell(
        structure: PMG_Structure,
        random_seed: int,
        max_normal_strain_fraction: float,
        max_shear_strain_fraction: float,
    ) -> PMG_Structure:
        _ = random_seed, max_normal_strain_fraction, max_shear_strain_fraction
        stretched_structure = structure.copy()
        stretched_structure.properties["candidate_kind"] = (
            "stretch_after_rattle"
            if structure.properties.get("candidate_kind") == "rattle"
            else "stretch_after_mlip_relax"
        )
        return stretched_structure

    monkeypatch.setattr(
        "crystal.structure.generators.augmentation_filtering.rattle_atom_positions",
        _mock_rattle_atom_positions,
    )
    monkeypatch.setattr(augmentation_filtering, "stretch_cell", _mock_stretch_cell)

    def _mock_minimum_interatomic_distance(structure: PMG_Structure) -> float:
        return 0.0 if structure.properties.get("candidate_kind") == "rattle" else 1.0

    monkeypatch.setattr(
        augmentation_filtering,
        "get_minimum_interatomic_distance",
        _mock_minimum_interatomic_distance,
    )

    result = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[],
        relaxed_structures=[TEST_SI_STRUCTURE],
        settings=StructureAugmentationAndFilteringSettings(
            number_of_rattled_variants=1,
            number_of_stretched_variants=1,
            rattle_settings=TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        ),
    )
    properties = result.properties

    assert [
        structure.properties[STAGE_LABEL_PROPERTY_KEY]
        for structure in properties.structures
    ] == [
        StructureGenerationSource.mlip_relax.value,
        StructureGenerationSource.stretch_after_rattle.value,
        StructureGenerationSource.stretch_after_mlip_relax.value,
    ]
    assert (
        properties.statistics.rattle_after_mlip_relax
        == StructureGenerationSourceStatistics(considered=1, passed=0, filtered=1)
    )
    assert (
        properties.statistics.stretch_after_rattle
        == StructureGenerationSourceStatistics(considered=1, passed=1, filtered=0)
    )


def test_root_isolation_retains_successful_roots_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rattle_atom_positions = rattle_atom_positions

    def _mock_rattle_atom_positions(
        structure: PMG_Structure,
        random_seed: int,
        noise_scaling_factor: float,
        displacement_mode: RattleAtomDisplacementMode,
    ) -> PMG_Structure:
        if structure.properties.get("root") == "failed":
            msg = "rattle failed"
            raise RuntimeError(msg)
        return original_rattle_atom_positions(
            structure=structure,
            random_seed=random_seed,
            noise_scaling_factor=noise_scaling_factor,
            displacement_mode=displacement_mode,
        )

    monkeypatch.setattr(
        "crystal.structure.generators.augmentation_filtering.rattle_atom_positions",
        _mock_rattle_atom_positions,
    )
    successful_structure = TEST_SI_STRUCTURE.copy()
    successful_structure.properties["root"] = "successful"
    failed_structure = TEST_SI_STRUCTURE.copy()
    failed_structure.properties["root"] = "failed"

    run = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[],
        relaxed_structures=[successful_structure, failed_structure],
        settings=StructureAugmentationAndFilteringSettings(
            number_of_rattled_variants=1,
            number_of_stretched_variants=1,
            rattle_settings=TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        ),
    )

    assert [
        (structure.properties[STAGE_LABEL_PROPERTY_KEY], structure.properties["root"])
        for structure in run.properties.structures
    ] == [
        (StructureGenerationSource.mlip_relax.value, "successful"),
        (StructureGenerationSource.mlip_relax.value, "failed"),
        (StructureGenerationSource.rattle_after_mlip_relax.value, "successful"),
        (StructureGenerationSource.stretch_after_rattle.value, "successful"),
        (StructureGenerationSource.stretch_after_mlip_relax.value, "successful"),
    ]
    assert run.warnings == ("MLIP Relax / RuntimeError: 1 root; rattle failed",)


def test_root_failures_include_exception_type_when_message_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_empty_error(**_: object) -> NoReturn:
        raise RuntimeError

    monkeypatch.setattr(
        augmentation_filtering, "passes_geometric_structure_filter", _raise_empty_error
    )

    result = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[TEST_BCC_FE_STRUCTURE],
        relaxed_structures=[TEST_SI_STRUCTURE],
        settings=StructureAugmentationAndFilteringSettings(),
    )

    assert result.warnings == (
        "Structure Generation / RuntimeError: 1 root; RuntimeError",
        "MLIP Relax / RuntimeError: 1 root; RuntimeError",
    )


def test_root_failures_are_aggregated_by_source_and_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_rattle_error(**_: object) -> NoReturn:
        msg = "rattle failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        augmentation_filtering, "rattle_atom_positions", _raise_rattle_error
    )

    result = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[],
        relaxed_structures=[TEST_SI_STRUCTURE, TEST_SI_STRUCTURE.copy()],
        settings=StructureAugmentationAndFilteringSettings(
            number_of_rattled_variants=1,
            number_of_stretched_variants=1,
            rattle_settings=TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        ),
    )

    assert result.properties.statistics.root_failure_summaries == (
        StructureAugmentationRootFailureSummary(
            source=StructureGenerationSource.mlip_relax,
            exception_type="RuntimeError",
            count=2,
            first_error_message="rattle failed",
        ),
    )
    assert result.warnings == ("MLIP Relax / RuntimeError: 2 roots; rattle failed",)


def test_all_invalid_stretch_variants_fail_the_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(augmentation_filtering, "stretch_cell", lambda **_: None)

    result = augmentation_filtering.augment_and_filter_structures(
        initial_structures=[],
        relaxed_structures=[TEST_SI_STRUCTURE],
        settings=StructureAugmentationAndFilteringSettings(
            number_of_rattled_variants=1,
            number_of_stretched_variants=1,
            stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
            stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        ),
    )

    assert len(result.warnings) == 1
    assert "All random deformation matrices were invalid" in result.warnings[0]
