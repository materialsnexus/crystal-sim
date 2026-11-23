"""Tests for slab-driven interface geometry optimisation workflows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from jobflow.core.flow import Flow
from jobflow.core.job import Job, job
from jobflow.managers.local import run_locally
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.calculators.mlip.relax_maker import MlipRelaxMaker
from crystal.flows.interface_geometry_optimisation_maker import (
    InterfaceGeometryOptimisationMaker,
    perform_geometry_optimisation,
    select_minimum_energy_geometry_optimisation_result,
)
from crystal.schemas.interface_geometry_optimisation import (
    GeometryOptimisationCandidateRecord,
    GeometryOptimisationResult,
    GeometryOptimisationSlabInputs,
    InterfaceGeometryOptimisationInputs,
    InterfaceGeometryOptimisationSimulationSettings,
)
from crystal.schemas.mlip import MlipRelaxTaskDocument
from crystal.structure.interfaces.builder import (
    Interface,
    RequiredInterfaceFieldsInStructure,
)
from crystal.structure.interfaces.settings import (
    DEFAULT_INTERFACE_VACUUM_ANGSTROM,
    InterfaceAssemblySettings,
)
from tests.helpers.builders.properties import build_relax_properties_for_test
from tests.helpers.builders.relax_maker_settings import (
    build_mlip_relax_maker_settings_for_test,
)
from tests.helpers.constants.structure import (
    FLOATING_POINT_TOLERANCE,
    TEST_BOTTOM_SLAB,
    TEST_INTERFACE,
    TEST_INTERFACE_FACTORY,
    TEST_INTERFACE_JOBS_PER_BATCH,
    TEST_TOP_SLAB,
)

TEST_OFFSETS = ((0.0, 0.0), (0.5, 0.0))
TEST_GAPS = (1.5, 2.0)
TEST_VACUUM_IN_ANGSTROM = 12.0


@job
def _mock_relax_job(
    *, structure: PMG_Structure, energy: float, dir_name: str
) -> MlipRelaxTaskDocument:
    return MlipRelaxTaskDocument(
        dir_name=dir_name,
        errors="",
        properties=build_relax_properties_for_test(structure=structure, energy=energy),
    )


@job
def _mock_failed_relax_job(*, dir_name: str) -> MlipRelaxTaskDocument:
    return MlipRelaxTaskDocument(
        dir_name=dir_name, errors="failed_relax", properties=None
    )


DEFAULT_INTERFACE_GEOMETRY_OPTIMISATION_MAKER = InterfaceGeometryOptimisationMaker(
    simulation_settings=InterfaceGeometryOptimisationSimulationSettings(
        relax_maker_settings=build_mlip_relax_maker_settings_for_test()
    )
)


def _make_inputs(
    assembly_settings: InterfaceAssemblySettings,
) -> InterfaceGeometryOptimisationInputs:
    return InterfaceGeometryOptimisationInputs(
        slab_inputs=GeometryOptimisationSlabInputs(
            top_slab=deepcopy(TEST_INTERFACE_FACTORY.top_slab),
            bottom_slab=deepcopy(TEST_INTERFACE_FACTORY.bottom_slab),
        ),
        assembly_settings=assembly_settings,
    )


def test_slab_inputs_gap_uses_larger_neighbor_distance() -> None:
    slab_inputs = GeometryOptimisationSlabInputs(
        top_slab=deepcopy(TEST_INTERFACE_FACTORY.top_slab),
        bottom_slab=deepcopy(TEST_INTERFACE_FACTORY.bottom_slab),
    )
    expected_gap = max(
        slab_inputs.orthogonal_top_slab.minimum_neighbor_distance_in_angstrom,
        slab_inputs.orthogonal_bottom_slab.minimum_neighbor_distance_in_angstrom,
    )

    assert slab_inputs.gap == pytest.approx(expected_gap)


def test_slab_inputs_reference_interface_uses_gap_as_vacuum() -> None:
    slab_inputs = GeometryOptimisationSlabInputs(
        top_slab=deepcopy(TEST_INTERFACE_FACTORY.top_slab),
        bottom_slab=deepcopy(TEST_INTERFACE_FACTORY.bottom_slab),
    )

    reference_interface = slab_inputs.reference_interface

    assert reference_interface.vacuum_over_film == pytest.approx(slab_inputs.gap)


def test_interface_inputs_gap_uses_reference_interface_gap() -> None:
    inputs = InterfaceGeometryOptimisationInputs(interface=deepcopy(TEST_INTERFACE))

    assert inputs.gap == pytest.approx(TEST_INTERFACE.gap)


def test_default_settings_generate_one_candidate_with_zero_offset_and_positive_gap() -> (
    None
):
    inputs = InterfaceGeometryOptimisationInputs(
        slab_inputs=GeometryOptimisationSlabInputs(
            top_slab=deepcopy(TEST_INTERFACE_FACTORY.top_slab),
            bottom_slab=deepcopy(TEST_INTERFACE_FACTORY.bottom_slab),
        )
    )

    [candidate] = inputs.interface_candidates

    assert isinstance(candidate, Interface)
    assert inputs.assembly_settings.offsets == [(0.0, 0.0)]
    assert inputs.assembly_settings.gap_multipliers == [1.0]
    assert np.allclose(candidate.in_plane_offset, (0.0, 0.0))
    assert candidate.gap > 0.0
    assert candidate.vacuum_over_film == pytest.approx(
        DEFAULT_INTERFACE_VACUUM_ANGSTROM
    )


def test_explicit_offsets_and_gaps_generate_candidates_in_deterministic_order() -> None:
    inputs = _make_inputs(
        InterfaceAssemblySettings(
            offsets=TEST_OFFSETS, gaps=TEST_GAPS, vacuum=TEST_VACUUM_IN_ANGSTROM
        )
    )

    candidates = inputs.interface_candidates

    assert len(candidates) == len(TEST_OFFSETS) * len(TEST_GAPS)
    assert [candidate.gap for candidate in candidates] == [1.5, 1.5, 2.0, 2.0]
    assert [tuple(candidate.in_plane_offset) for candidate in candidates] == [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.0),
        (0.5, 0.0),
    ]


def test_explicit_offsets_and_gap_multipliers_generate_candidates() -> None:
    inputs = _make_inputs(
        InterfaceAssemblySettings(offsets=TEST_OFFSETS, gap_multipliers=[1.0, 1.1])
    )

    candidates = inputs.interface_candidates

    assert len(candidates) == 4
    assert all(candidate.gap > 0.0 for candidate in candidates)


def test_generated_candidates_preserve_composition_and_atom_count() -> None:
    inputs = _make_inputs(
        InterfaceAssemblySettings(offsets=TEST_OFFSETS, gaps=TEST_GAPS)
    )
    candidates = inputs.interface_candidates
    reference_candidate = candidates[0]

    assert all(
        candidate.composition == reference_candidate.composition
        and len(candidate) == len(reference_candidate)
        for candidate in candidates
    )


def test_interface_jobs_per_batch_delegates_to_assembly_settings() -> None:
    inputs = _make_inputs(
        InterfaceAssemblySettings(
            offsets=[(0.0, 0.0)], interface_jobs_per_batch=TEST_INTERFACE_JOBS_PER_BATCH
        )
    )

    assert inputs.interface_jobs_per_batch == TEST_INTERFACE_JOBS_PER_BATCH


def test_interface_mode_is_valid() -> None:
    inputs = InterfaceGeometryOptimisationInputs(interface=deepcopy(TEST_INTERFACE))

    assert inputs.interface is not None
    assert inputs.slab_inputs is None


def test_no_source_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="slab_inputs or interface"):
        InterfaceGeometryOptimisationInputs()


def test_mixed_slab_and_interface_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        InterfaceGeometryOptimisationInputs(
            interface=deepcopy(TEST_INTERFACE),
            slab_inputs=GeometryOptimisationSlabInputs(
                top_slab=deepcopy(TEST_TOP_SLAB), bottom_slab=deepcopy(TEST_BOTTOM_SLAB)
            ),
        )


def test_interface_mode_explicit_settings_preserve_candidate_metadata() -> None:
    inputs = InterfaceGeometryOptimisationInputs(
        interface=deepcopy(TEST_INTERFACE),
        assembly_settings=InterfaceAssemblySettings(
            offsets=TEST_OFFSETS, gaps=TEST_GAPS
        ),
    )

    candidates = inputs.interface_candidates

    assert [candidate.gap for candidate in candidates] == [1.5, 1.5, 2.0, 2.0]
    assert [tuple(candidate.in_plane_offset) for candidate in candidates] == [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.0),
        (0.5, 0.0),
    ]
    assert all(len(candidate) == len(TEST_INTERFACE) for candidate in candidates)
    assert all(
        candidate.composition == TEST_INTERFACE.composition for candidate in candidates
    )
    assert all(
        candidate.site_properties["interface_label"]
        == TEST_INTERFACE.site_properties["interface_label"]
        for candidate in candidates
    )
    assert all(
        all(
            field in candidate.properties
            for field in RequiredInterfaceFieldsInStructure.values()
        )
        for candidate in candidates
    )


@patch.object(MlipRelaxMaker, "make")
def test_maker_creates_one_relaxation_job_per_candidate_and_honours_batching(
    mock_relax_make: MagicMock,
) -> None:
    inputs = _make_inputs(
        InterfaceAssemblySettings(
            offsets=[(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0)],
            gaps=[1.5],
            interface_jobs_per_batch=2,
        )
    )
    candidates = inputs.interface_candidates
    mock_relax_make.side_effect = [
        _mock_relax_job(
            structure=candidate,
            energy=float(candidate_index),
            dir_name=f"relax-{candidate_index}",
        )
        for candidate_index, candidate in enumerate(candidates)
    ]

    flow = DEFAULT_INTERFACE_GEOMETRY_OPTIMISATION_MAKER.make(inputs)

    batch_relax_flows = flow.jobs[:-1]
    batch_job_counts: list[int] = []
    for batch_relax_flow in batch_relax_flows:
        assert isinstance(batch_relax_flow, Flow)
        batch_job_counts.append(len(batch_relax_flow.jobs))

    assert mock_relax_make.call_count == len(candidates)
    assert batch_job_counts == [2, 2, 1]
    assert isinstance(flow.jobs[-1], Job)


def test_selected_candidate_record_returns_selected_record() -> None:
    selected_record = GeometryOptimisationCandidateRecord(
        candidate_index=1, final_relax_energy=-1.0
    )
    result = GeometryOptimisationResult(
        selected_candidate_index=1,
        selected_structure=deepcopy(TEST_INTERFACE),
        selected_final_energy=-1.0,
        candidate_records=[
            GeometryOptimisationCandidateRecord(
                candidate_index=0, final_relax_energy=2.0
            ),
            selected_record,
        ],
    )

    assert result.selected_candidate_record == selected_record


def test_successful_candidates_select_lowest_final_energy(tmp_path: Path) -> None:
    relax_maker = Mock()
    relax_maker.make.side_effect = [
        _mock_relax_job(
            structure=deepcopy(TEST_INTERFACE), energy=2.0, dir_name="interface-high"
        ),
        _mock_relax_job(
            structure=deepcopy(TEST_INTERFACE), energy=-1.0, dir_name="interface-low"
        ),
    ]
    geometry_flow = perform_geometry_optimisation(
        relax_maker, structures=[deepcopy(TEST_INTERFACE), deepcopy(TEST_INTERFACE)]
    )

    run_flow = run_locally(geometry_flow, root_dir=tmp_path, create_folders=True)
    result = run_flow[geometry_flow.jobs[-1].uuid][1].output

    assert isinstance(result, GeometryOptimisationResult)
    assert result.selected_candidate_index == 1
    assert result.selected_final_energy == pytest.approx(-1.0)


def test_failed_candidate_is_retained_and_excluded_from_selection(
    tmp_path: Path,
) -> None:
    relax_maker = Mock()
    relax_maker.make.side_effect = [
        _mock_failed_relax_job(dir_name="interface-failed"),
        _mock_relax_job(
            structure=deepcopy(TEST_INTERFACE), energy=3.0, dir_name="interface-high"
        ),
        _mock_relax_job(
            structure=deepcopy(TEST_INTERFACE), energy=1.0, dir_name="interface-low"
        ),
    ]
    geometry_flow = perform_geometry_optimisation(
        relax_maker,
        structures=[
            deepcopy(TEST_INTERFACE),
            deepcopy(TEST_INTERFACE),
            deepcopy(TEST_INTERFACE),
        ],
    )

    run_flow = run_locally(geometry_flow, root_dir=tmp_path, create_folders=True)
    result = run_flow[geometry_flow.jobs[-1].uuid][1].output

    assert isinstance(result, GeometryOptimisationResult)
    assert result.selected_candidate_index == 2
    assert [record.is_success for record in result.candidate_records] == [
        False,
        True,
        True,
    ]


def test_all_failed_candidates_stop_children(tmp_path: Path) -> None:
    selection_job = select_minimum_energy_geometry_optimisation_result(
        relax_task_documents=[
            MlipRelaxTaskDocument(
                dir_name="failed-relax-1", errors="first failure", properties=None
            ),
            MlipRelaxTaskDocument(
                dir_name="failed-relax-2", errors="second failure", properties=None
            ),
        ]
    )

    child_job = select_minimum_energy_geometry_optimisation_result(
        relax_task_documents=selection_job.output
    )

    responses = run_locally(
        Flow([selection_job, child_job]),
        root_dir=tmp_path,
        create_folders=True,
        raise_immediately=True,
        ensure_success=False,
    )

    assert responses[selection_job.uuid][1].stop_children
    assert responses[selection_job.uuid][1].output is None
    assert child_job.uuid not in responses


def test_maker_returns_interface_result_with_preserved_metadata(tmp_path: Path) -> None:
    inputs = _make_inputs(
        InterfaceAssemblySettings(offsets=TEST_OFFSETS, gaps=TEST_GAPS)
    )
    candidates = inputs.interface_candidates
    ase_atoms_adaptor = AseAtomsAdaptor()
    relaxed_structures: list[PMG_Structure] = [
        ase_atoms_adaptor.get_structure(ase_atoms_adaptor.get_atoms(candidate))
        for candidate in candidates
    ]

    with patch.object(MlipRelaxMaker, "make") as mock_relax_make:
        mock_relax_make.side_effect = [
            _mock_relax_job(
                structure=relaxed_structure,
                energy=0.0 if candidate_index == 2 else 10.0,
                dir_name=f"relax-{candidate_index}",
            )
            for candidate_index, relaxed_structure in enumerate(relaxed_structures)
        ]
        flow = DEFAULT_INTERFACE_GEOMETRY_OPTIMISATION_MAKER.make(inputs)

    run_flow = run_locally(flow, root_dir=tmp_path, create_folders=True)
    result = run_flow[flow.jobs[-1].uuid][1].output

    assert isinstance(result, GeometryOptimisationResult)
    assert isinstance(result.selected_structure, PMG_Structure)
    assert result.selected_candidate_index == 2
    assert result.selected_final_energy == pytest.approx(0.0)
    selected_candidate = candidates[result.selected_candidate_index]
    for property_name in RequiredInterfaceFieldsInStructure.values():
        assert result.selected_structure.properties[property_name] == pytest.approx(
            selected_candidate.properties[property_name], abs=FLOATING_POINT_TOLERANCE
        )
    assert (
        result.selected_structure.site_properties["interface_label"]
        == selected_candidate.site_properties["interface_label"]
    )
    reconstructed_interface = Interface.from_structure(result.selected_structure)
    assert len(reconstructed_interface) == len(selected_candidate)


def test_interface_mode_selects_lowest_energy_candidate(tmp_path: Path) -> None:
    inputs = InterfaceGeometryOptimisationInputs(
        interface=deepcopy(TEST_INTERFACE),
        assembly_settings=InterfaceAssemblySettings(
            offsets=TEST_OFFSETS, gaps=TEST_GAPS
        ),
    )
    candidates = inputs.interface_candidates

    relax_maker_side_effects = [
        _mock_relax_job(
            structure=candidate,
            energy=float(len(candidates) - candidate_index),
            dir_name=f"interface-{candidate_index}",
        )
        for candidate_index, candidate in enumerate(candidates)
    ]

    with patch.object(MlipRelaxMaker, "make") as mock_relax_make:
        mock_relax_make.side_effect = relax_maker_side_effects
        geometry_flow = DEFAULT_INTERFACE_GEOMETRY_OPTIMISATION_MAKER.make(inputs)

    run_flow = run_locally(geometry_flow, root_dir=tmp_path, create_folders=True)
    result = run_flow[geometry_flow.jobs[-1].uuid][1].output

    assert isinstance(result, GeometryOptimisationResult)
    assert result.selected_candidate_index == len(candidates) - 1


def test_empty_structure_sequence_raises_value_error() -> None:
    relax_maker = Mock()

    with pytest.raises(ValueError, match="At least one structure"):
        _ = perform_geometry_optimisation(relax_maker, structures=[])
