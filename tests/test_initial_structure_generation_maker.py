"""Tests for the initial-structure generation maker contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from jobflow.core.job import Job
from jobflow.managers.local import run_locally
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationProperties,
    StructureIndexRange,
)
from crystal.schemas.structure_generators import InitialStructureGenerationTaskDocument
from crystal.structure.generators.pyxtal.initial_structure_generation_maker import (
    InitialStructureGenerationMaker,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.constants.structure_generators import (
    make_initial_structure_generation_settings,
)


@run_in_temp_dir
def test_initial_structure_generation_maker_uses_zero_based_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_indexes: list[int] = []

    def _mock_generate(_self: object, structure_index: int) -> PMG_Structure:
        generated_indexes.append(structure_index)
        return deepcopy(TEST_SI_STRUCTURE)

    monkeypatch.setattr(
        "crystal.structure.generators.pyxtal.initial_structure_generation_maker.InitialStructureGenerator.generate",
        _mock_generate,
    )
    maker = InitialStructureGenerationMaker(
        simulation_settings=make_initial_structure_generation_settings(
            space_groups=[225]
        )
    )
    expected_range = StructureIndexRange(
        first_structure_index=0, last_structure_index=2
    )

    generation_job = maker.make(number_of_structures=3)

    assert isinstance(generation_job, Job)
    run_result = run_locally(generation_job, root_dir=Path.cwd(), create_folders=True)
    output_document = run_result[generation_job.uuid][1].output

    assert isinstance(output_document, InitialStructureGenerationTaskDocument)
    assert output_document.errors == ""
    assert isinstance(output_document.properties, InitialStructureGenerationProperties)
    assert output_document.properties.generation_range == expected_range
    assert len(output_document.properties.structures) == 3
    assert generated_indexes == [0, 1, 2]


@run_in_temp_dir
def test_initial_structure_generation_maker_retains_partial_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_indexes: list[int] = []

    def _mock_generate(_self: object, structure_index: int) -> PMG_Structure:
        generated_indexes.append(structure_index)
        if structure_index == 1:
            failure_message = "mock generation failure"
            raise RuntimeError(failure_message)
        return deepcopy(TEST_SI_STRUCTURE)

    monkeypatch.setattr(
        "crystal.structure.generators.pyxtal.initial_structure_generation_maker.InitialStructureGenerator.generate",
        _mock_generate,
    )
    maker = InitialStructureGenerationMaker(
        simulation_settings=make_initial_structure_generation_settings(
            space_groups=[225]
        )
    )
    expected_range = StructureIndexRange(
        first_structure_index=0, last_structure_index=2
    )

    generation_job = maker.make(number_of_structures=3)
    run_result = run_locally(generation_job, root_dir=Path.cwd(), create_folders=True)
    output_document = run_result[generation_job.uuid][1].output

    assert output_document.errors == ""
    assert isinstance(output_document.properties, InitialStructureGenerationProperties)
    assert output_document.properties.generation_range == expected_range
    assert output_document.properties.terminated_early is True
    assert len(output_document.properties.structures) == 1
    assert generated_indexes == [0, 1]


@run_in_temp_dir
def test_initial_structure_generation_maker_reports_all_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_indexes: list[int] = []

    def _mock_generate(_self: object, structure_index: int) -> PMG_Structure:
        generated_indexes.append(structure_index)
        failure_message = "mock generation failure"
        raise RuntimeError(failure_message)

    monkeypatch.setattr(
        "crystal.structure.generators.pyxtal.initial_structure_generation_maker.InitialStructureGenerator.generate",
        _mock_generate,
    )
    maker = InitialStructureGenerationMaker(
        simulation_settings=make_initial_structure_generation_settings(
            space_groups=[225]
        )
    )
    generation_job = maker.make(number_of_structures=3)
    run_result = run_locally(generation_job, root_dir=Path.cwd(), create_folders=True)
    output_document = run_result[generation_job.uuid][1].output

    assert output_document.properties is None
    assert output_document.errors == (
        "Failed to generate any of the requested 3 structures."
    )
    assert generated_indexes == [0]
