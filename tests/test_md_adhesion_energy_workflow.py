"""Tests for the MD-only adhesion workflow."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import pytest
from jobflow.core.flow import Flow
from jobflow.core.job import Job, job
from jobflow.core.store import JobStore
from jobflow.managers.local import run_locally
from maggma.stores import MemoryStore

from crystal.calculators.md.equilibration_maker import MlipMdEquilibrationMaker
from crystal.calculators.mlip.relax_maker import MlipRelaxMaker
from crystal.flows.interface_geometry_optimisation_maker import (
    InterfaceGeometryOptimisationMaker,
)
from crystal.flows.md_adhesion_workflow_maker import (
    MDAdhesionEnergyMaker,
    _interface_area_in_angstrom_square,
    collect_and_build_adhesion_energy_record,
    get_md_equilibration_flow,
    prepare_production_structures,
)
from crystal.schemas.interface_geometry_optimisation import (
    GeometryOptimisationResult,
    InterfaceGeometryOptimisationInputs,
    InterfaceGeometryOptimisationSimulationSettings,
)
from crystal.schemas.md.adhesion import (
    MDAdhesionEnergyResult,
    MDAdhesionEnergySimulationSettings,
)
from crystal.schemas.md.mlip_md_equilibration import (
    MlipMdEquilibrationSimulationSettings,
    MlipMdEquilibrationTaskDocument,
)
from crystal.schemas.md.properties import (
    MdCheckResult,
    MdEquilibrationPlots,
    MdEquilibrationProperties,
)
from crystal.structure.interfaces.settings import InterfaceAssemblySettings
from tests.helpers.builders.relax_maker_settings import (
    build_mlip_relax_maker_settings_for_test,
)
from tests.helpers.constants.md import MOCK_MD_PROPERTIES, MOCK_MD_RUN_SETTINGS
from tests.helpers.constants.structure import (
    FLOATING_POINT_TOLERANCE,
    TEST_INTERFACE,
    TEST_TOP_SLAB,
)
from tests.test_interface_geometry_optimisation_workflow import _mock_relax_job

TEST_MD_STAGE_SETTINGS = MlipMdEquilibrationSimulationSettings(
    run_settings=MOCK_MD_RUN_SETTINGS, time_averaging_window=0.005
)
TEST_SETTINGS = MDAdhesionEnergySimulationSettings(
    interface_md_settings=[], production_md_setting=TEST_MD_STAGE_SETTINGS
)


def _mock_md_task_document() -> MlipMdEquilibrationTaskDocument:
    properties = MOCK_MD_PROPERTIES.model_copy(
        update={
            "check_results": [
                MdCheckResult(
                    check_type="AverageEnergyDriftCheck",
                    passed=True,
                    threshold=1.0,
                    observed_value=0.0,
                )
            ],
            "plots": MdEquilibrationPlots(energy_plot_path="energy-plot-path"),
        }
    )
    return MlipMdEquilibrationTaskDocument(
        dir_name="md-energy", errors="", properties=properties
    )


def test_interface_area_uses_in_plane_lattice() -> None:
    area = _interface_area_in_angstrom_square(TEST_INTERFACE)
    assert area == pytest.approx(8.179, abs=FLOATING_POINT_TOLERANCE)


@patch.object(MlipMdEquilibrationMaker, "make")
def test_failed_first_md_stage_prevents_second_and_preserves_failure(
    mock_make: Mock, tmp_path: Path
) -> None:
    failed_task_document = MlipMdEquilibrationTaskDocument(
        dir_name="failed", errors="first MD stage failed", properties=None
    )

    @job
    def failed_md() -> MlipMdEquilibrationTaskDocument:
        return failed_task_document

    mock_make.return_value = failed_md()
    flow_job = get_md_equilibration_flow(
        stage_settings=[TEST_MD_STAGE_SETTINGS, TEST_MD_STAGE_SETTINGS],
        md_executor_settings=TEST_SETTINGS.md_executor_settings,
        structure=TEST_TOP_SLAB,
    )
    run = run_locally(flow_job, root_dir=tmp_path, create_folders=True)

    replacement = run[flow_job.uuid][1].replace
    assert isinstance(replacement, Flow)
    continuation_job = replacement.jobs[1]
    assert isinstance(continuation_job, Job)
    assert run[continuation_job.uuid][1].output == failed_task_document
    assert mock_make.call_count == 1


def test_settings_round_trip() -> None:
    restored = MDAdhesionEnergySimulationSettings.from_dict(TEST_SETTINGS.as_dict())
    assert restored == TEST_SETTINGS
    assert restored.interface_md_settings == []


def test_failed_interface_md_stops_production_structure_preparation(
    tmp_path: Path,
) -> None:
    failed_output = MlipMdEquilibrationTaskDocument(
        dir_name="failed", errors="interface preparation failed", properties=None
    )
    preparation_job = prepare_production_structures(interface_md_output=failed_output)

    with pytest.raises(RuntimeError, match="interface preparation failed"):
        run_locally(
            preparation_job,
            root_dir=tmp_path,
            create_folders=True,
            raise_immediately=True,
        )


def test_production_structures_use_final_interface_frame(tmp_path: Path) -> None:
    task_document = _mock_md_task_document()
    preparation_job = prepare_production_structures(interface_md_output=task_document)

    with patch.object(
        MdEquilibrationProperties,
        "last_frame_structure",
        new_callable=PropertyMock,
        return_value=TEST_INTERFACE,
    ):
        run = run_locally(preparation_job, root_dir=tmp_path, create_folders=True)

    structures = run[preparation_job.uuid][1].output
    assert structures.interface_structure == TEST_INTERFACE
    assert structures.top_slab_structure == TEST_INTERFACE.get_centered_top_slab()
    assert structures.bottom_slab_structure == TEST_INTERFACE.get_centered_bottom_slab()
    final_interface = structures.interface_structure
    assert {
        "gap_angstrom",
        "in_plane_offset",
        "vacuum_over_film",
    } <= final_interface.properties.keys()
    assert set(final_interface.site_properties["interface_label"]) == {
        "film",
        "substrate",
    }


def test_production_structure_preparation_requires_preserved_interface_metadata(
    tmp_path: Path,
) -> None:
    task_document = _mock_md_task_document()
    preparation_job = prepare_production_structures(interface_md_output=task_document)
    structure_without_metadata = TEST_INTERFACE.copy()
    structure_without_metadata.properties.pop("gap_angstrom")

    with (
        patch.object(
            MdEquilibrationProperties,
            "last_frame_structure",
            new_callable=PropertyMock,
            return_value=structure_without_metadata,
        ),
        pytest.raises(ValueError, match="Missing required interface properties"),
    ):
        run_locally(
            preparation_job,
            root_dir=tmp_path,
            create_folders=True,
            raise_immediately=True,
        )


def test_aggregation_reports_failed_md_branch(tmp_path: Path) -> None:
    failed_output = MlipMdEquilibrationTaskDocument(
        dir_name="failed", errors="interface branch failed", properties=None
    )
    successful_output = _mock_md_task_document()
    aggregation_job = collect_and_build_adhesion_energy_record(
        top_slab_md_output=successful_output,
        bottom_slab_md_output=successful_output,
        interface_md_output=failed_output,
    )

    with pytest.raises(RuntimeError, match=r"interface branch failed"):
        run_locally(
            aggregation_job,
            root_dir=tmp_path,
            create_folders=True,
            raise_immediately=True,
        )


def test_successful_aggregation_scales_energies_and_uses_interface_area(
    tmp_path: Path,
) -> None:
    task_document = _mock_md_task_document()
    aggregation_job = collect_and_build_adhesion_energy_record(
        top_slab_md_output=task_document,
        bottom_slab_md_output=task_document,
        interface_md_output=task_document,
    )

    with (
        patch.object(
            MdEquilibrationProperties,
            "average_energy",
            new_callable=PropertyMock,
            side_effect=[10.0, 20.0, 25.0],
        ),
        patch.object(
            MdEquilibrationProperties,
            "last_frame_structure",
            new_callable=PropertyMock,
            side_effect=[TEST_TOP_SLAB[:1], TEST_TOP_SLAB[:1], TEST_INTERFACE],
        ),
    ):
        run = run_locally(aggregation_job, root_dir=tmp_path, create_folders=True)

    result = run[aggregation_job.uuid][1].output
    scaled_adhesion_energy = 10.0 * 2 + 20.0 * 2 - 25.0
    assert result.top_slab_energy == pytest.approx(10.0)
    assert result.bottom_slab_energy == pytest.approx(20.0)
    assert result.interface_energy == pytest.approx(25.0)
    assert result.interface_area == pytest.approx(
        _interface_area_in_angstrom_square(TEST_INTERFACE)
    )
    assert result.work_of_adhesion_per_area == pytest.approx(
        scaled_adhesion_energy / result.interface_area
    )


def test_structures_and_aggregation_jobs_have_queue_metadata() -> None:
    settings = TEST_SETTINGS.model_copy(
        update={"interface_md_settings": [TEST_MD_STAGE_SETTINGS]}
    )
    maker = MDAdhesionEnergyMaker(simulation_settings=settings)
    adhesion_job = maker.make(TEST_INTERFACE)
    store = JobStore(docs_store=MemoryStore())
    store.connect()

    response = adhesion_job.run(store=store)
    flow = response.replace

    assert isinstance(flow, Flow)
    structures_job, aggregation_job = flow.jobs[1], flow.jobs[-2]
    assert isinstance(structures_job, Job)
    assert isinstance(aggregation_job, Job)
    assert aggregation_job.name == "collect_and_build_adhesion_energy_record"
    expected_metadata = maker.slurm_settings.to_queueadapter_spec()
    assert structures_job.metadata == expected_metadata
    assert aggregation_job.metadata == expected_metadata


@pytest.mark.parametrize(
    "interface_md_settings",
    [[], [TEST_MD_STAGE_SETTINGS]],
    ids=["without-interface-md", "with-interface-md"],
)
def test_geometry_optimisation_feeds_md_adhesion_energy(
    interface_md_settings: list[MlipMdEquilibrationSimulationSettings], tmp_path: Path
) -> None:
    settings = TEST_SETTINGS.model_copy(
        update={"interface_md_settings": interface_md_settings}
    )
    inputs = InterfaceGeometryOptimisationInputs(
        interface=deepcopy(TEST_INTERFACE),
        assembly_settings=InterfaceAssemblySettings(
            offsets=[(0.0, 0.0), (0.5, 0.0)], gaps=[2.0]
        ),
    )
    candidates = inputs.interface_candidates
    selected_interface = candidates[1]
    expected_structures = (
        selected_interface.get_centered_top_slab(),
        selected_interface.get_centered_bottom_slab(),
        selected_interface,
    )
    expected_last_frames = [
        *([selected_interface] if interface_md_settings else []),
        *expected_structures,
    ]
    md_energies = (-5.0, -7.0, -15.0)
    store = JobStore(docs_store=MemoryStore())
    store.connect()

    with (
        patch.object(
            MlipRelaxMaker,
            "make",
            side_effect=[
                _mock_relax_job(
                    structure=candidate, energy=energy, dir_name=f"relax-{index}"
                )
                for index, (candidate, energy) in enumerate(
                    zip(candidates, (2.0, -1.0), strict=True)
                )
            ],
        ) as mock_relax,
        patch.object(
            MlipMdEquilibrationMaker,
            "make",
            side_effect=[job(_mock_md_task_document)() for _ in expected_last_frames],
        ) as mock_md_make,
        patch.object(
            MdEquilibrationProperties,
            "average_energy",
            new_callable=PropertyMock,
            side_effect=md_energies,
        ),
        patch.object(
            MdEquilibrationProperties,
            "last_frame_structure",
            new_callable=PropertyMock,
            side_effect=expected_last_frames,
        ),
    ):
        optimisation_flow = InterfaceGeometryOptimisationMaker(
            simulation_settings=InterfaceGeometryOptimisationSimulationSettings(
                relax_maker_settings=build_mlip_relax_maker_settings_for_test()
            )
        ).make(inputs)
        adhesion_job = MDAdhesionEnergyMaker(simulation_settings=settings).make(
            optimisation_flow.output.selected_structure
        )
        combined_flow = Flow(
            [optimisation_flow, adhesion_job], output=adhesion_job.output
        )
        responses = run_locally(
            combined_flow,
            store=store,
            root_dir=tmp_path,
            create_folders=True,
            raise_immediately=True,
        )
    selection_job = optimisation_flow.jobs[-1]
    optimisation_result = responses[selection_job.uuid][1].output
    assert isinstance(optimisation_result, GeometryOptimisationResult)
    assert optimisation_result.selected_candidate_index == 1
    assert mock_relax.call_count == 2
    assert mock_md_make.call_count == len(expected_last_frames)
    result = combined_flow.output.resolve(store)
    assert isinstance(result, MDAdhesionEnergyResult)
    assert result.top_slab_energy == pytest.approx(md_energies[0])
    assert result.bottom_slab_energy == pytest.approx(md_energies[1])
    assert result.interface_energy == pytest.approx(md_energies[2])
    area = _interface_area_in_angstrom_square(selected_interface)
    assert result.interface_area == pytest.approx(area)
    assert result.work_of_adhesion_per_area == pytest.approx(3.0 / area)


def test_failed_interface_md_prevents_production_md(tmp_path: Path) -> None:
    settings = MDAdhesionEnergySimulationSettings(
        interface_md_settings=[TEST_MD_STAGE_SETTINGS],
        production_md_setting=TEST_MD_STAGE_SETTINGS,
    )
    adhesion_job = MDAdhesionEnergyMaker(simulation_settings=settings).make(
        TEST_INTERFACE
    )

    with (
        patch.object(
            MlipMdEquilibrationMaker,
            "make",
            side_effect=[job(_mock_md_task_document)()],
        ) as mock_md_make,
        patch.object(
            MdEquilibrationProperties,
            "all_checks_passed",
            new_callable=PropertyMock,
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="Interface MD failed before production MD"),
    ):
        run_locally(
            adhesion_job, root_dir=tmp_path, create_folders=True, raise_immediately=True
        )

    assert mock_md_make.call_count == 1
