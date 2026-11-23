"""Tests for the structure augmentation and filtering maker."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

import pytest
from jobflow.core.job import Job
from jobflow.managers.local import run_locally
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.structure_generators import STAGE_LABEL_PROPERTY_KEY
from crystal.schemas.filtering import CohesiveEnergyFilteringSettings
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationProperties,
    StructureIndexRange,
)
from crystal.schemas.mlip import MlipRelaxTaskDocument
from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from crystal.schemas.structure_augmentation import (
    StructureAugmentationAndFilteringProperties,
    StructureGenerationSource,
)
from crystal.schemas.structure_generators import (
    InitialStructureGenerationTaskDocument,
    StructureAugmentationAndFilteringSettings,
    StructureAugmentationAndFilteringTaskDocument,
)
from crystal.structure.generators.augmentation_filtering_maker import (
    StructureAugmentationAndFilteringMaker,
)
from tests.helpers.constants.structure import TEST_BCC_FE_STRUCTURE, TEST_SI_STRUCTURE
from tests.helpers.constants.structure_generators import (
    TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS,
    TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
)

ZERO_FORCE_VECTOR = [0.0, 0.0, 0.0]
ZERO_STRESS_VECTOR = [0.0] * 6


def _settings(
    *,
    isolated_atom_energies: dict[str, float] | None = None,
    maximum_cohesive_energy_per_atom: float | None = 1.0,
) -> StructureAugmentationAndFilteringSettings:
    return StructureAugmentationAndFilteringSettings(
        number_of_rattled_variants=1,
        number_of_stretched_variants=1,
        rattle_settings=TEST_RATTLE_STRUCTURE_GENERATION_SETTINGS,
        stretch_after_rattle_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        stretch_after_mlip_relax_settings=TEST_STRETCH_STRUCTURE_GENERATION_SETTINGS,
        cohesive_energy_filtering_settings=CohesiveEnergyFilteringSettings(
            isolated_atom_energies=(
                isolated_atom_energies
                if isolated_atom_energies is not None
                else {"Si": -1.0}
            ),
            maximum_cohesive_energy_per_atom=maximum_cohesive_energy_per_atom,
        ),
    )


def _initial_structure_generation_task_document(
    structures: list[PMG_Structure],
) -> InitialStructureGenerationTaskDocument:
    return InitialStructureGenerationTaskDocument(
        errors="",
        properties=InitialStructureGenerationProperties(
            generation_range=StructureIndexRange(
                last_structure_index=max(1, len(structures)) - 1
            ),
            structures=structures,
        ),
    )


def _mlip_relax_task_document(
    structure: PMG_Structure, *, energy: float = 0.0
) -> MlipRelaxTaskDocument:
    return MlipRelaxTaskDocument(
        dir_name="mocked-mlip-relax",
        errors="",
        properties=RelaxProperties(
            relax_steps=[
                SinglePointProperties(
                    step=0,
                    structure=structure,
                    forces=[ZERO_FORCE_VECTOR for _ in structure],
                    stress=ZERO_STRESS_VECTOR,
                    energy=energy,
                )
            ]
        ),
    )


def _run_maker(
    *,
    maker: StructureAugmentationAndFilteringMaker,
    initial_structures: list[PMG_Structure],
    relaxed_structures: list[PMG_Structure],
    root_dir: Path,
    mlip_relax_task_documents: list[MlipRelaxTaskDocument] | None = None,
) -> tuple[Job, StructureAugmentationAndFilteringTaskDocument]:
    job = maker.make(
        initial_structure_generation_task_document=_initial_structure_generation_task_document(
            initial_structures
        ),
        mlip_relax_task_documents=(
            mlip_relax_task_documents
            if mlip_relax_task_documents is not None
            else [
                _mlip_relax_task_document(structure) for structure in relaxed_structures
            ]
        ),
    )
    run_result = run_locally(job, root_dir=root_dir, create_folders=True)
    task_document = run_result[job.uuid][1].output

    assert isinstance(task_document, StructureAugmentationAndFilteringTaskDocument)
    return job, task_document


def test_maker_returns_direct_ready_filtered_structures(tmp_path: Path) -> None:
    maker = StructureAugmentationAndFilteringMaker(
        simulation_settings=_settings(maximum_cohesive_energy_per_atom=0.0)
    )

    _, task_document = _run_maker(
        maker=maker,
        initial_structures=[TEST_BCC_FE_STRUCTURE],
        relaxed_structures=[TEST_SI_STRUCTURE],
        root_dir=tmp_path,
        mlip_relax_task_documents=[
            _mlip_relax_task_document(TEST_SI_STRUCTURE, energy=-2.0)
        ],
    )

    assert task_document.errors == ""
    assert task_document.warnings == ""
    assert isinstance(
        task_document.properties, StructureAugmentationAndFilteringProperties
    )
    assert [
        structure.properties[STAGE_LABEL_PROPERTY_KEY]
        for structure in task_document.properties.structures
    ] == [
        StructureGenerationSource.initial.value,
        StructureGenerationSource.mlip_relax.value,
        StructureGenerationSource.rattle_after_mlip_relax.value,
        StructureGenerationSource.stretch_after_rattle.value,
        StructureGenerationSource.stretch_after_mlip_relax.value,
    ]
    cohesive_energy_statistics = (
        task_document.properties.statistics.pre_augmentation_cohesive_energy
    )
    assert cohesive_energy_statistics is not None
    assert cohesive_energy_statistics.model_dump() == {
        "considered": 1,
        "passed": 1,
        "filtered": 0,
        "failed": 0,
    }


def test_maker_rejects_high_energy_relaxed_root_before_augmentation(
    tmp_path: Path,
) -> None:
    maker = StructureAugmentationAndFilteringMaker(
        simulation_settings=_settings(maximum_cohesive_energy_per_atom=0.0)
    )

    _, task_document = _run_maker(
        maker=maker,
        initial_structures=[],
        relaxed_structures=[TEST_SI_STRUCTURE],
        root_dir=tmp_path,
        mlip_relax_task_documents=[
            _mlip_relax_task_document(TEST_SI_STRUCTURE, energy=-1.0)
        ],
    )

    assert task_document.errors == ""
    assert task_document.warnings == ""
    assert isinstance(
        task_document.properties, StructureAugmentationAndFilteringProperties
    )
    assert task_document.properties.structures == []
    cohesive_energy_statistics = (
        task_document.properties.statistics.pre_augmentation_cohesive_energy
    )
    assert cohesive_energy_statistics is not None
    assert cohesive_energy_statistics.model_dump() == {
        "considered": 1,
        "passed": 0,
        "filtered": 1,
        "failed": 0,
    }
    statistics = task_document.properties.statistics
    assert all(
        stage_statistics.considered == 0
        for stage_statistics in (
            statistics.mlip_relax,
            statistics.rattle_after_mlip_relax,
            statistics.stretch_after_rattle,
            statistics.stretch_after_mlip_relax,
        )
    )


def test_maker_disables_cohesive_energy_filtering_when_settings_are_omitted(
    tmp_path: Path,
) -> None:
    maker = StructureAugmentationAndFilteringMaker(
        simulation_settings=_settings().model_copy(
            update={"cohesive_energy_filtering_settings": None}
        )
    )

    _, task_document = _run_maker(
        maker=maker,
        initial_structures=[],
        relaxed_structures=[TEST_SI_STRUCTURE],
        root_dir=tmp_path,
        mlip_relax_task_documents=[
            _mlip_relax_task_document(TEST_SI_STRUCTURE, energy=100.0)
        ],
    )

    assert task_document.errors == ""
    assert isinstance(
        task_document.properties, StructureAugmentationAndFilteringProperties
    )
    assert len(task_document.properties.structures) == 4
    assert task_document.properties.statistics.pre_augmentation_cohesive_energy is None


def test_maker_excludes_failed_relaxation_documents_with_diagnostic_warning(
    tmp_path: Path,
) -> None:
    failed_document = MlipRelaxTaskDocument(
        dir_name="failed-mlip-relax", errors="MLIP relaxation failed", properties=None
    )
    maker = StructureAugmentationAndFilteringMaker(
        simulation_settings=_settings(maximum_cohesive_energy_per_atom=0.0)
    )

    _, task_document = _run_maker(
        maker=maker,
        initial_structures=[],
        relaxed_structures=[],
        root_dir=tmp_path,
        mlip_relax_task_documents=[failed_document],
    )

    assert task_document.errors == ""
    assert isinstance(
        task_document.properties, StructureAugmentationAndFilteringProperties
    )
    assert task_document.properties.structures == []
    assert "MLIP relaxation failed" in task_document.warnings
    cohesive_energy_statistics = (
        task_document.properties.statistics.pre_augmentation_cohesive_energy
    )
    assert cohesive_energy_statistics is not None
    assert cohesive_energy_statistics.model_dump() == {
        "considered": 1,
        "passed": 0,
        "filtered": 0,
        "failed": 1,
    }


def test_maker_excludes_relaxation_document_without_steps(tmp_path: Path) -> None:
    incomplete_document = MlipRelaxTaskDocument(
        dir_name="incomplete-mlip-relax", properties=RelaxProperties(relax_steps=[])
    )
    maker = StructureAugmentationAndFilteringMaker(
        simulation_settings=_settings(maximum_cohesive_energy_per_atom=0.0)
    )

    _, task_document = _run_maker(
        maker=maker,
        initial_structures=[],
        relaxed_structures=[],
        root_dir=tmp_path,
        mlip_relax_task_documents=[incomplete_document],
    )

    assert task_document.errors == ""
    assert task_document.warnings == (
        "MLIP relaxation failed: no relaxation steps were recorded"
    )
    assert isinstance(
        task_document.properties, StructureAugmentationAndFilteringProperties
    )
    cohesive_energy_statistics = (
        task_document.properties.statistics.pre_augmentation_cohesive_energy
    )
    assert cohesive_energy_statistics is not None
    assert cohesive_energy_statistics.model_dump() == {
        "considered": 1,
        "passed": 0,
        "filtered": 0,
        "failed": 1,
    }


def test_maker_retains_partial_results_and_records_root_failure(tmp_path: Path) -> None:
    def _raise_rattle_error(**_: object) -> PMG_Structure:
        msg = "rattle failed"
        raise RuntimeError(msg)

    with patch(
        "crystal.structure.generators.augmentation_filtering.rattle_atom_positions",
        _raise_rattle_error,
    ):
        maker = StructureAugmentationAndFilteringMaker(simulation_settings=_settings())

        _, task_document = _run_maker(
            maker=maker,
            initial_structures=[],
            relaxed_structures=[TEST_SI_STRUCTURE],
            root_dir=tmp_path,
        )

    assert task_document.errors == ""
    assert isinstance(
        task_document.properties, StructureAugmentationAndFilteringProperties
    )
    assert len(task_document.properties.structures) == 1
    assert task_document.warnings == "MLIP Relax / RuntimeError: 1 root; rattle failed"


def test_maker_normalizes_terminal_failure_without_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise_empty_error(**_: object) -> NoReturn:
        raise RuntimeError

    monkeypatch.setattr(
        "crystal.structure.generators.augmentation_filtering_maker.augment_and_filter_structures",
        _raise_empty_error,
    )
    maker = StructureAugmentationAndFilteringMaker(simulation_settings=_settings())

    _, task_document = _run_maker(
        maker=maker,
        initial_structures=[],
        relaxed_structures=[TEST_SI_STRUCTURE],
        root_dir=tmp_path,
    )

    assert task_document.properties is None
    assert task_document.warnings == ""
    assert task_document.errors == "RuntimeError"
