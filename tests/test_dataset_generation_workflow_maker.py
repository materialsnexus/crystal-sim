"""Tests for the scalable dataset-generation workflow."""

from __future__ import annotations

from copy import deepcopy
from json import dumps
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jobflow.core.flow import Flow
from jobflow.core.job import job
from jobflow.managers.local import run_locally
from pydantic import ValidationError
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.relax_maker import MlipRelaxBatchMaker
from crystal.constants.structure_generators import STAGE_LABEL_PROPERTY_KEY
from crystal.schemas.filtering import CohesiveEnergyFilteringSettings
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationProperties,
    InitialStructureGenerationSettings,
    StructureIndexRange,
)
from crystal.schemas.mlip import (
    MlipRelaxMakerSettings,
    MlipRelaxSimulationSettings,
    MlipRelaxTaskDocument,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.properties import (
    CohesiveEnergyFilteringProperties,
    RelaxProperties,
    SinglePointProperties,
    StructureGenerationProperties,
)
from crystal.schemas.relax_settings import OptimizerType
from crystal.schemas.structure_augmentation import StructureGenerationSource
from crystal.schemas.structure_generators import (
    BirchClusteringSettings,
    ClusterSamplingSettings,
    DatasetGenerationWorkflowSettings,
    DirectSamplingMakerSettings,
    DirectSamplingSettings,
    DirectSamplingTaskDocument,
    InitialStructureGenerationMakerSettings,
    InitialStructureGenerationTaskDocument,
    PostAugmentationCohesiveEnergyFilteringSettings,
    RattleAtomPositionsSettings,
    StretchCellSettings,
    StructureAugmentationAndFilteringMakerSettings,
    StructureAugmentationAndFilteringSettings,
    StructureAugmentationAndFilteringTaskDocument,
)
from crystal.schemas.structure_labelling import StructureLabellingWorkflowSettings
from crystal.structure.generators.dataset_generation_workflow_maker import (
    DatasetGenerationWorkflowMaker,
    build_direct_sampling_flow_after_augmentation_and_filtering,
    build_structure_generation_workflow_task_document,
    build_structure_labelling_flow_after_direct_sampling,
)
from crystal.structure.generators.pyxtal.initial_structure_generation_maker import (
    InitialStructureGenerationMaker,
)
from crystal.structure.generators.sampling.direct_maker import DirectSamplingMaker
from crystal.structure.labelling.structure_labelling_workflow import (
    fan_out_single_points,
)
from crystal.structure.labelling.structure_labelling_workflow_maker import (
    StructureLabellingWorkflowMaker,
)
from tests.helpers.builders.single_point_maker_settings import (
    build_mlip_single_point_maker_settings_for_test,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.jobflow import make_blob_jobstore


def _dataset_generation_workflow_settings() -> DatasetGenerationWorkflowSettings:
    model = MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    return DatasetGenerationWorkflowSettings(
        number_of_initial_structures=1,
        initial_structure_generation_maker_settings=InitialStructureGenerationMakerSettings(
            simulation_settings=InitialStructureGenerationSettings(random_seed=42)
        ),
        mlip_relax_maker_settings=MlipRelaxMakerSettings(
            simulation_settings=MlipRelaxSimulationSettings(
                optimizer_type=OptimizerType.BFGS,
                max_steps=2,
                force_threshold_in_ev_per_angstrom=0.01,
                model=model,
            )
        ),
        structure_augmentation_and_filtering_maker_settings=StructureAugmentationAndFilteringMakerSettings(
            simulation_settings=StructureAugmentationAndFilteringSettings(
                number_of_rattled_variants=1,
                number_of_stretched_variants=1,
                rattle_settings=RattleAtomPositionsSettings(random_seed=42),
                stretch_after_rattle_settings=StretchCellSettings(random_seed=42),
                stretch_after_mlip_relax_settings=StretchCellSettings(random_seed=42),
                cohesive_energy_filtering_settings=CohesiveEnergyFilteringSettings(
                    isolated_atom_energies={"Si": -1.0},
                    maximum_cohesive_energy_per_atom=1.0,
                ),
            )
        ),
        direct_sampling_maker_settings=DirectSamplingMakerSettings(
            simulation_settings=DirectSamplingSettings(
                structure_encoder_model=model,
                birch_clustering_settings=BirchClusteringSettings(number_of_clusters=2),
                cluster_sampling_settings=ClusterSamplingSettings(
                    num_samples_per_cluster=1
                ),
            )
        ),
        structure_labelling_workflow_settings=StructureLabellingWorkflowSettings(
            single_point_maker_settings=build_mlip_single_point_maker_settings_for_test()
        ),
    )


def _build_relaxed_mlip_task_document(
    structure: PMG_Structure,
) -> MlipRelaxTaskDocument:
    return MlipRelaxTaskDocument(
        dir_name="mock",
        properties=RelaxProperties(
            relax_steps=[
                SinglePointProperties(
                    step=0,
                    structure=structure,
                    forces=[[0.0, 0.0, 0.0] for _ in structure],
                    stress=[0.0] * 6,
                    energy=-4.0,
                )
            ]
        ),
    )


@job(output_schema=InitialStructureGenerationTaskDocument)
def _build_initial_structure_generation_task_document(
    number_of_structures: int,
) -> InitialStructureGenerationTaskDocument:
    return InitialStructureGenerationTaskDocument(
        properties=InitialStructureGenerationProperties(
            generation_range=StructureIndexRange(
                last_structure_index=number_of_structures - 1
            ),
            structures=[
                deepcopy(TEST_SI_STRUCTURE) for _ in range(number_of_structures)
            ],
        )
    )


@job
def _build_relaxed_mlip_task_documents(
    structures: list[PMG_Structure],
) -> list[MlipRelaxTaskDocument]:
    return [_build_relaxed_mlip_task_document(structure) for structure in structures]


@job(output_schema=DirectSamplingTaskDocument)
def _build_direct_sampling_task_document(
    structures: list[PMG_Structure],
) -> DirectSamplingTaskDocument:
    return DirectSamplingTaskDocument(
        properties=StructureGenerationProperties(structures=structures)
    )


def test_post_augmentation_filtering_rejects_mismatched_evaluator_model() -> None:
    with pytest.raises(ValidationError, match="must match the MLIP single-point model"):
        PostAugmentationCohesiveEnergyFilteringSettings(
            cohesive_energy_filtering_settings=CohesiveEnergyFilteringSettings(
                evaluator_model=MlipModelRef(
                    registry_model=QetModelRegistry.finetuned_linbosi
                ),
                maximum_cohesive_energy_per_atom=1.0,
            ),
            mlip_single_point_maker_settings=build_mlip_single_point_maker_settings_for_test(),
        )


def test_post_augmentation_filtering_requires_a_threshold() -> None:
    with pytest.raises(
        ValidationError, match="requires maximum_cohesive_energy_per_atom"
    ):
        PostAugmentationCohesiveEnergyFilteringSettings(
            mlip_single_point_maker_settings=build_mlip_single_point_maker_settings_for_test()
        )


def test_workflow_adds_post_augmentation_filtering_when_configured() -> None:
    settings = _dataset_generation_workflow_settings().model_copy(
        update={
            "post_augmentation_cohesive_energy_filtering_settings": PostAugmentationCohesiveEnergyFilteringSettings(
                cohesive_energy_filtering_settings=CohesiveEnergyFilteringSettings(
                    isolated_atom_energies={"Si": -1.0},
                    maximum_cohesive_energy_per_atom=1.0,
                ),
                mlip_single_point_maker_settings=build_mlip_single_point_maker_settings_for_test(),
            )
        }
    )

    flow = DatasetGenerationWorkflowMaker(simulation_settings=settings).make()

    assert len(flow.jobs) == 7


def test_direct_sampling_flow_accepts_missing_source_properties(tmp_path: Path) -> None:
    model = MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    dynamic_job = build_direct_sampling_flow_after_augmentation_and_filtering(
        structure_source_properties=None,
        source_structures_job_uuid="source-job",
        direct_sampling_maker=DirectSamplingMaker(
            simulation_settings=DirectSamplingSettings(structure_encoder_model=model)
        ),
    )
    jobstore, _ = make_blob_jobstore()

    run_result = run_locally(
        dynamic_job, root_dir=tmp_path, create_folders=True, store=jobstore
    )
    direct_output = next(
        response.output
        for response_by_index in run_result.values()
        for response in response_by_index.values()
        if isinstance(response.output, DirectSamplingTaskDocument)
    )

    assert isinstance(direct_output.properties, StructureGenerationProperties)
    assert direct_output.properties.structures == []


def test_direct_sampling_replacement_keeps_source_structures_as_reference() -> None:
    model = MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    source_properties = CohesiveEnergyFilteringProperties(
        structures=[deepcopy(TEST_SI_STRUCTURE) for _ in range(100)],
        statistics={"considered": 100, "passed": 100},
    )
    original_builder = vars(
        build_direct_sampling_flow_after_augmentation_and_filtering
    )["original"]
    response = original_builder(
        structure_source_properties=source_properties,
        source_structures_job_uuid="source-job",
        direct_sampling_maker=DirectSamplingMaker(
            simulation_settings=DirectSamplingSettings(structure_encoder_model=model)
        ),
    )
    [direct_sampling_job] = response.replace.jobs

    serialized_job = dumps(direct_sampling_job.as_dict())
    assert len(serialized_job) < 100_000
    assert "source-job" in serialized_job


@patch.object(StructureLabellingWorkflowMaker, "make", autospec=True)
@patch.object(DirectSamplingMaker, "make")
@patch.object(MlipRelaxBatchMaker, "make")
@patch.object(InitialStructureGenerationMaker, "make")
def test_workflow_routes_augmentation_results_to_direct_without_post_filtering(
    mock_initial_make: MagicMock,
    mock_mlip_make: MagicMock,
    mock_direct_make: MagicMock,
    mock_labelling_make: MagicMock,
    tmp_path: Path,
) -> None:
    mock_initial_make.side_effect = _build_initial_structure_generation_task_document
    mock_mlip_make.side_effect = _build_relaxed_mlip_task_documents
    mock_direct_make.side_effect = _build_direct_sampling_task_document

    def build_labelling_flow(
        maker: StructureLabellingWorkflowMaker,
        *,
        source_structures: list[PMG_Structure],
    ) -> Flow:
        _ = source_structures
        return Flow(
            jobs=[
                fan_out_single_points(
                    source_structures=[], single_point_maker=maker.single_point_maker
                )
            ],
            output=None,
        )

    mock_labelling_make.side_effect = build_labelling_flow
    jobstore, _ = make_blob_jobstore()

    flow = DatasetGenerationWorkflowMaker(
        simulation_settings=_dataset_generation_workflow_settings()
    ).make()
    assert len(flow.jobs) == 6
    run_result = run_locally(
        flow, root_dir=tmp_path, create_folders=True, store=jobstore
    )
    outputs = [response[1].output for response in run_result.values()]
    output = next(
        output
        for output in outputs
        if hasattr(output, "augmentation_and_filtering_statistics")
    )

    assert isinstance(output.properties, StructureGenerationProperties)
    source_structures = mock_labelling_make.call_args.kwargs["source_structures"]

    assert source_structures == output.properties.structures
    assert mock_labelling_make.call_count == 1
    assert [
        structure.properties[STAGE_LABEL_PROPERTY_KEY]
        for structure in output.properties.structures
    ] == [
        StructureGenerationSource.initial.value,
        StructureGenerationSource.mlip_relax.value,
        StructureGenerationSource.rattle_after_mlip_relax.value,
        StructureGenerationSource.stretch_after_rattle.value,
        StructureGenerationSource.stretch_after_mlip_relax.value,
    ]
    assert output.augmentation_and_filtering_statistics is not None
    assert output.post_augmentation_cohesive_energy_filtering_statistics is None


def test_final_workflow_document_retains_direct_sampling_warning(
    tmp_path: Path,
) -> None:
    workflow_task_document_job = build_structure_generation_workflow_task_document(
        initial_structure_generation_task_document=InitialStructureGenerationTaskDocument(
            properties=InitialStructureGenerationProperties(
                generation_range=StructureIndexRange(last_structure_index=0),
                structures=[TEST_SI_STRUCTURE],
            )
        ),
        mlip_relax_task_documents=[],
        direct_sampling_task_document=DirectSamplingTaskDocument(
            properties=StructureGenerationProperties(structures=[]),
            warnings="DIRECT sampling warning",
        ),
        augmentation_and_filtering_task_document=(
            StructureAugmentationAndFilteringTaskDocument(
                properties=None, errors="augmentation failed"
            )
        ),
    )
    jobstore, _ = make_blob_jobstore()

    run_result = run_locally(
        workflow_task_document_job,
        root_dir=tmp_path,
        create_folders=True,
        store=jobstore,
    )

    final_workflow_result = run_result[workflow_task_document_job.uuid][1].output

    assert final_workflow_result.properties is None
    assert final_workflow_result.errors == "augmentation failed"
    assert "DIRECT sampling warning" in final_workflow_result.warnings


def test_failed_direct_sampling_skips_structure_labelling(tmp_path: Path) -> None:
    labelling_job = build_structure_labelling_flow_after_direct_sampling(
        direct_sampling_task_document=DirectSamplingTaskDocument(
            properties=None, errors="DIRECT sampling failed"
        ),
        structure_labelling_workflow_maker=StructureLabellingWorkflowMaker(
            simulation_settings=StructureLabellingWorkflowSettings(
                single_point_maker_settings=build_mlip_single_point_maker_settings_for_test()
            )
        ),
    )
    jobstore, _ = make_blob_jobstore()

    run_result = run_locally(
        labelling_job, root_dir=tmp_path, create_folders=True, store=jobstore
    )

    assert list(run_result) == [labelling_job.uuid]
    assert run_result[labelling_job.uuid][1].output is None


def test_missing_labelling_settings_skips_structure_labelling(tmp_path: Path) -> None:
    labelling_job = build_structure_labelling_flow_after_direct_sampling(
        direct_sampling_task_document=DirectSamplingTaskDocument(
            properties=StructureGenerationProperties(structures=[TEST_SI_STRUCTURE])
        ),
        structure_labelling_workflow_maker=None,
    )
    jobstore, _ = make_blob_jobstore()

    run_result = run_locally(
        labelling_job, root_dir=tmp_path, create_folders=True, store=jobstore
    )

    assert list(run_result) == [labelling_job.uuid]
    assert run_result[labelling_job.uuid][1].output is None
