"""Behavior tests for post-augmentation MLIP evaluation and filtering."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from jobflow.core.flow import Flow
from jobflow.core.job import job
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import MemoryStore
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.single_point_maker import MlipSinglePointBatchMaker
from crystal.constants.mlip import MLIP_SINGLE_POINT_TASK_DOCUMENT_BLOB_STORE_NAME
from crystal.constants.structure_generators import (
    POST_AUGMENTATION_FAILURE_WARNING_MAX_LENGTH,
    POST_AUGMENTATION_MAX_FAILURE_WARNINGS,
    STAGE_LABEL_PROPERTY_KEY,
    STRUCTURE_BLOB_STORE_NAME,
)
from crystal.schemas.filtering import (
    CohesiveEnergyFilteringSettings,
    CohesiveEnergyFilteringTaskDocument,
)
from crystal.schemas.mlip import (
    MlipSinglePointSimulationSettings,
    MlipSinglePointTaskDocument,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.properties import (
    CohesiveEnergyFilteringProperties,
    SinglePointProperties,
)
from crystal.schemas.structure_augmentation import (
    StructureAugmentationAndFilteringProperties,
    StructureAugmentationAndFilteringStatistics,
    StructureGenerationSource,
)
from crystal.schemas.structure_generators import (
    StructureAugmentationAndFilteringTaskDocument,
)
from crystal.structure.generators.post_augmentation_filtering import (
    build_post_augmentation_cohesive_energy_filtering_flow,
    prepare_post_augmentation_cohesive_energy_filtering_inputs,
)
from crystal.structure.handlers.cohesive_energy_filtering import (
    run_cohesive_energy_filtering,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE


def _filtering_settings() -> CohesiveEnergyFilteringSettings:
    return CohesiveEnergyFilteringSettings(
        isolated_atom_energies={"Si": -1.0}, maximum_cohesive_energy_per_atom=0.0
    )


def _augmented_task_document(
    structures: list[PMG_Structure], errors: str = ""
) -> StructureAugmentationAndFilteringTaskDocument:
    empty_statistics = StructureAugmentationAndFilteringStatistics(
        initial={"considered": 0, "passed": 0, "filtered": 0},
        mlip_relax={"considered": 0, "passed": 0, "filtered": 0},
        rattle_after_mlip_relax={"considered": 0, "passed": 0, "filtered": 0},
        stretch_after_rattle={"considered": 0, "passed": 0, "filtered": 0},
        stretch_after_mlip_relax={"considered": 0, "passed": 0, "filtered": 0},
    )
    return StructureAugmentationAndFilteringTaskDocument(
        errors=errors,
        properties=(
            None
            if errors
            else StructureAugmentationAndFilteringProperties(
                structures=structures, statistics=empty_statistics
            )
        ),
    )


@job(  # type: ignore[call-overload]
    **{STRUCTURE_BLOB_STORE_NAME: [["output", "properties", "structures"]]}
)
def _store_augmented_task_document(
    task_document: StructureAugmentationAndFilteringTaskDocument,
) -> StructureAugmentationAndFilteringTaskDocument:
    return task_document


def _single_point_batch_maker() -> MlipSinglePointBatchMaker:
    model = MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    return MlipSinglePointBatchMaker(
        simulation_settings=MlipSinglePointSimulationSettings(model=model)
    )


def _single_point_task_document(
    structure: PMG_Structure, energy: float
) -> MlipSinglePointTaskDocument:
    return MlipSinglePointTaskDocument(
        properties=SinglePointProperties(
            step=0,
            structure=structure,
            forces=[[0.0, 0.0, 0.0] for _ in structure],
            stress=[0.0] * 6,
            energy=energy,
        )
    )


def _jobstore() -> JobStore:
    jobstore = JobStore(
        docs_store=MemoryStore(),
        additional_stores={
            "structures": MemoryStore(),
            MLIP_SINGLE_POINT_TASK_DOCUMENT_BLOB_STORE_NAME: MemoryStore(),
        },
    )
    jobstore.connect()
    return jobstore


def test_post_augmentation_filter_keeps_order_and_stage_labels(tmp_path: Path) -> None:
    accepted = deepcopy(TEST_SI_STRUCTURE)
    accepted.properties[STAGE_LABEL_PROPERTY_KEY] = (
        StructureGenerationSource.initial.value
    )
    rejected = deepcopy(TEST_SI_STRUCTURE)
    rejected.properties[STAGE_LABEL_PROPERTY_KEY] = (
        StructureGenerationSource.mlip_relax.value
    )
    failed = deepcopy(TEST_SI_STRUCTURE)
    failed.properties[STAGE_LABEL_PROPERTY_KEY] = (
        StructureGenerationSource.rattle_after_mlip_relax.value
    )
    evaluation_documents = [
        _single_point_task_document(accepted, energy=-2.0),
        _single_point_task_document(rejected, energy=0.0),
        MlipSinglePointTaskDocument(properties=None, errors="evaluation failed"),
    ]

    preparation_job = prepare_post_augmentation_cohesive_energy_filtering_inputs(
        source_structures=[accepted, rejected, failed],
        evaluation_task_documents=evaluation_documents,
    )
    filtering_job = run_cohesive_energy_filtering(
        structures=preparation_job.output.structures,
        total_energies=preparation_job.output.total_energies,
        settings=_filtering_settings(),
        failed_count=preparation_job.output.failed_count,
        warnings=preparation_job.output.warnings,
    )
    run_result = run_locally(
        Flow([preparation_job, filtering_job]),
        root_dir=tmp_path,
        create_folders=True,
        store=_jobstore(),
    )
    task_document = run_result[filtering_job.uuid][1].output

    assert isinstance(task_document, CohesiveEnergyFilteringTaskDocument)
    assert isinstance(task_document.properties, CohesiveEnergyFilteringProperties)
    assert task_document.properties.structures == [accepted]
    assert task_document.properties.structures[0].properties[
        STAGE_LABEL_PROPERTY_KEY
    ] == (StructureGenerationSource.initial.value)
    statistics = task_document.properties.statistics
    assert statistics.considered == 3
    assert statistics.passed == 1
    assert statistics.filtered == 1
    assert statistics.failed == 1
    assert "evaluation failed" in task_document.warnings


def test_post_augmentation_filter_bounds_failure_warnings(tmp_path: Path) -> None:
    source_structures = [
        deepcopy(TEST_SI_STRUCTURE)
        for _ in range(POST_AUGMENTATION_MAX_FAILURE_WARNINGS + 1)
    ]
    long_error = "evaluation failed " + ("details " * 200)
    evaluation_documents = [
        MlipSinglePointTaskDocument(properties=None, errors=long_error)
        for _ in source_structures
    ]
    preparation_job = prepare_post_augmentation_cohesive_energy_filtering_inputs(
        source_structures=source_structures,
        evaluation_task_documents=evaluation_documents,
    )
    filtering_job = run_cohesive_energy_filtering(
        structures=preparation_job.output.structures,
        total_energies=preparation_job.output.total_energies,
        settings=_filtering_settings(),
        failed_count=preparation_job.output.failed_count,
        warnings=preparation_job.output.warnings,
    )

    run_result = run_locally(
        Flow([preparation_job, filtering_job]),
        root_dir=tmp_path,
        create_folders=True,
        store=_jobstore(),
    )
    task_document = run_result[filtering_job.uuid][1].output

    assert isinstance(task_document, CohesiveEnergyFilteringTaskDocument)
    warning_lines = task_document.warnings.splitlines()
    diagnostic_warnings = warning_lines[:-1]
    assert len(diagnostic_warnings) == POST_AUGMENTATION_MAX_FAILURE_WARNINGS
    assert all(
        len(warning) <= POST_AUGMENTATION_FAILURE_WARNING_MAX_LENGTH
        for warning in diagnostic_warnings
    )
    assert len(task_document.warnings) <= (
        POST_AUGMENTATION_MAX_FAILURE_WARNINGS
        * (POST_AUGMENTATION_FAILURE_WARNING_MAX_LENGTH + 1)
        + len(warning_lines[-1])
    )
    assert "additional MLIP single-point failure diagnostics omitted" in (
        task_document.warnings
    )
    assert task_document.warnings.count("[truncated]") == (
        POST_AUGMENTATION_MAX_FAILURE_WARNINGS
    )
    assert isinstance(task_document.properties, CohesiveEnergyFilteringProperties)
    assert task_document.properties.statistics.failed == len(source_structures)


def test_post_augmentation_filter_uses_bounded_single_point_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_structures = [deepcopy(TEST_SI_STRUCTURE) for _ in range(5)]
    for structure in source_structures:
        structure.properties[STAGE_LABEL_PROPERTY_KEY] = (
            StructureGenerationSource.initial.value
        )

    def _evaluate_batch(
        *,
        structures: list[PMG_Structure],
        simulation_settings: MlipSinglePointSimulationSettings,
    ) -> list[MlipSinglePointTaskDocument]:
        _ = simulation_settings
        return [
            _single_point_task_document(structure, energy=-2.0)
            for structure in structures
        ]

    evaluate_batch = Mock(side_effect=_evaluate_batch)
    resolved_models: list[MlipModelRef | None] = []

    def _get_isolated_atom_energies(model: MlipModelRef | None) -> dict[str, float]:
        resolved_models.append(model)
        return {"Si": -1.0}

    monkeypatch.setattr(
        "crystal.calculators.mlip.single_point_maker.run_mlip_single_point_batch",
        evaluate_batch,
    )
    monkeypatch.setattr(
        "crystal.helpers.cohesive_energy.get_mlip_isolated_atom_energies",
        _get_isolated_atom_energies,
    )
    single_point_batch_maker = _single_point_batch_maker()
    augmentation_job = _store_augmented_task_document(
        _augmented_task_document(source_structures)
    )
    dynamic_job = build_post_augmentation_cohesive_energy_filtering_flow(
        augmentation_and_filtering_task_document=augmentation_job.output,
        source_structures_job_uuid=augmentation_job.uuid,
        single_point_batch_maker=single_point_batch_maker,
        structures_per_job=2,
        cohesive_energy_filtering_settings=CohesiveEnergyFilteringSettings(
            evaluator_model=single_point_batch_maker.simulation_settings.model,
            maximum_cohesive_energy_per_atom=0.0,
        ),
    )

    run_result = run_locally(
        Flow([augmentation_job, dynamic_job]),
        root_dir=tmp_path,
        create_folders=True,
        store=_jobstore(),
    )

    assert [
        len(call.kwargs["structures"]) for call in evaluate_batch.call_args_list
    ] == [2, 2, 1]
    outputs = [
        response.output
        for responses in run_result.values()
        for response in responses.values()
    ]
    filtering_output = next(
        output
        for output in outputs
        if isinstance(output, CohesiveEnergyFilteringTaskDocument)
    )
    assert isinstance(filtering_output.properties, CohesiveEnergyFilteringProperties)
    assert len(filtering_output.properties.structures) == 5
    assert resolved_models == [single_point_batch_maker.simulation_settings.model]


def test_disabled_post_augmentation_filtering_skips_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_structures = [deepcopy(TEST_SI_STRUCTURE) for _ in range(2)]
    for structure in source_structures:
        structure.properties[STAGE_LABEL_PROPERTY_KEY] = (
            StructureGenerationSource.initial.value
        )
    evaluation_batch = Mock(side_effect=AssertionError("evaluation should be skipped"))
    monkeypatch.setattr(
        "crystal.calculators.mlip.single_point_maker.run_mlip_single_point_batch",
        evaluation_batch,
    )

    augmentation_job = _store_augmented_task_document(
        _augmented_task_document(source_structures)
    )
    dynamic_job = build_post_augmentation_cohesive_energy_filtering_flow(
        augmentation_and_filtering_task_document=augmentation_job.output,
        source_structures_job_uuid=augmentation_job.uuid,
        single_point_batch_maker=_single_point_batch_maker(),
        structures_per_job=2,
        cohesive_energy_filtering_settings=CohesiveEnergyFilteringSettings(),
    )

    run_result = run_locally(
        Flow([augmentation_job, dynamic_job]), root_dir=tmp_path, create_folders=True
    )
    output = run_result[dynamic_job.uuid][1].output

    assert evaluation_batch.call_count == 0
    assert isinstance(output, CohesiveEnergyFilteringTaskDocument)
    assert isinstance(output.properties, CohesiveEnergyFilteringProperties)
    assert output.properties.structures == source_structures
    statistics = output.properties.statistics
    assert statistics.considered == 2
    assert statistics.passed == 2
    assert statistics.filtered == 0
    assert statistics.failed == 0


def test_empty_augmented_candidates_create_no_single_point_jobs(tmp_path: Path) -> None:
    augmentation_job = _store_augmented_task_document(_augmented_task_document([]))
    dynamic_job = build_post_augmentation_cohesive_energy_filtering_flow(
        augmentation_and_filtering_task_document=augmentation_job.output,
        source_structures_job_uuid=augmentation_job.uuid,
        single_point_batch_maker=_single_point_batch_maker(),
        structures_per_job=2,
        cohesive_energy_filtering_settings=_filtering_settings(),
    )

    run_result = run_locally(
        Flow([augmentation_job, dynamic_job]), root_dir=tmp_path, create_folders=True
    )
    output = run_result[dynamic_job.uuid][1].output

    assert isinstance(output, CohesiveEnergyFilteringTaskDocument)
    assert isinstance(output.properties, CohesiveEnergyFilteringProperties)
    assert output.properties.structures == []
    assert output.properties.statistics.considered == 0


def test_augmentation_failure_skips_post_augmentation_evaluation(
    tmp_path: Path,
) -> None:
    augmentation_job = _store_augmented_task_document(
        _augmented_task_document([], errors="augmentation failed")
    )
    dynamic_job = build_post_augmentation_cohesive_energy_filtering_flow(
        augmentation_and_filtering_task_document=augmentation_job.output,
        source_structures_job_uuid=augmentation_job.uuid,
        single_point_batch_maker=_single_point_batch_maker(),
        structures_per_job=2,
        cohesive_energy_filtering_settings=_filtering_settings(),
    )

    run_result = run_locally(
        Flow([augmentation_job, dynamic_job]), root_dir=tmp_path, create_folders=True
    )
    output = run_result[dynamic_job.uuid][1].output

    assert isinstance(output, CohesiveEnergyFilteringTaskDocument)
    assert output.properties is None
    assert "augmentation failed" in output.errors
