from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from jobflow.core.flow import Flow
from jobflow.core.job import Job, Response, job
from jobflow.managers.local import run_locally
from pydantic import ValidationError
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure as PMG_Structure

from crystal import CRYSTAL_CONFIG
from crystal.constants.adhesion import (
    INTERFACE_CHARGEMOL_NOT_CONVERGED_ERROR,
    INTERFACE_RECORD_COUNT_MISMATCH_ERROR,
    SLAB_CHARGEMOL_NOT_CONVERGED_ERROR,
    SLAB_RELAX_NOT_CONVERGED_ERROR,
)
from crystal.constants.chargemol import (
    CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE,
    CHARGEMOL_EXECUTABLE_RELATIVE_PATH,
    CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME,
)
from crystal.constants.process import PROCESS_EXIT_CODE_SUCCESS
from crystal.constants.qe import QE_PW_EXECUTABLE_NAME
from crystal.constants.qe_pp import QE_PP_CHARGE_DENSITY_BASENAME, QE_PP_EXECUTABLE_NAME
from crystal.flows.adhesion_workflow_helpers import (
    build_adhesion_energy_records_from_single_point_task_docs,
    extract_relax_final_structure_or_raise,
    relax_properties_converged,
)
from crystal.flows.dft_adhesion_workflow_maker import (
    AdhesionWorkflowMaker,
    build_slab_chargemol_jobs_before_interface_relax,
    extract_relaxed_slabs_after_slab_relax,
)
from crystal.helpers.structure import split_items_into_batches
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.adhesion import (
    AdhesionEnergyRecord,
    AdhesionSimulationSettings,
    AdhesionWorkflowInputs,
    AdhesionWorkflowResult,
)
from crystal.schemas.process import ProcessRunInfo
from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from crystal.schemas.qe import (
    QeRelaxMakerSettings,
    QeRelaxTaskDocument,
    QeScfPpChargemolTaskDocument,
)
from crystal.schemas.relaxation_constraints import (
    ExternalFacesFreezeSettings,
    RegionBasedFixedAtomSelectionType,
    RelaxationConstraints,
)
from crystal.schemas.settings import ExecutorSettings
from crystal.structure.interfaces.settings import InterfaceAssemblySettings
from crystal.structure.slabs.builder import Slab
from tests.helpers.builders.relax_maker_settings import (
    build_qe_relax_maker_settings_for_test,
)
from tests.helpers.builders.single_point_maker_settings import (
    build_mlip_single_point_maker_settings_for_test,
    build_qe_single_point_maker_settings_for_test,
)
from tests.helpers.constants.qe import (
    QE_MISSING_PSEUDOS_MESSAGE,
    QE_NOT_CONFIGURED_MESSAGE,
)
from tests.helpers.constants.structure import (
    DEFAULT_INTERFACE_GAPS,
    DEFAULT_INTERFACE_OFFSETS,
    DEFAULT_INTERFACE_VACUUM,
    MOCK_ERROR_LOG_MESSAGE,
    MOCK_TASK_DIR_NAME,
    SLAB_ASSEMBLY_TEST_BOTTOM_PATH,
    SLAB_ASSEMBLY_TEST_TOP_PATH,
    TEST_ADHESION_RECORD_COUNT_SINGLE,
    TEST_C_AXIS_INDEX,
    TEST_FIRST_INDEX,
    TEST_INTERFACE_ASSEMBLER_BOTTOM_LATTICE_KEY,
    TEST_INTERFACE_ASSEMBLER_TOP_LATTICE_KEY,
    TEST_INTERFACE_BATCH_SIZES,
    TEST_INTERFACE_COUNT,
    TEST_INTERFACE_ENERGY_A,
    TEST_INTERFACE_ENERGY_B,
    TEST_INTERFACE_ENERGY_SINGLE,
    TEST_INTERFACE_GAP_MULTIPLIERS_CUSTOM,
    TEST_INTERFACE_GAPS_CUSTOM,
    TEST_INTERFACE_JOBS_PER_BATCH,
    TEST_INTERFACE_OFFSETS_CUSTOM,
    TEST_INTERFACE_RELAX_EXTERNAL_FACES_FREEZE_THICKNESS_IN_ANGSTROM,
    TEST_INTERFACE_TASK_DOC_COUNT_MULTI,
    TEST_INTERFACE_TASK_DOC_COUNT_SINGLE,
    TEST_INTERFACE_VACUUM_CUSTOM,
    TEST_LINBO3_SLAB,
    TEST_MANUAL_FIXED_ATOM_INDEX,
    TEST_RELAX_ENERGY_HIGH,
    TEST_RELAX_ENERGY_LOW,
    TEST_RELAX_ENERGY_PLACEHOLDER,
    TEST_RELAX_LATTICE_SCALE_DOWN,
    TEST_RELAX_LATTICE_SCALE_UP,
    TEST_RELAX_STEP_INDEX,
    TEST_RELAX_STRESS_VECTOR_ZERO,
    TEST_RELAX_ZERO_FORCE_VECTOR,
    TEST_SINGLE_RESULT_COUNT,
    TEST_SLAB_MILLER_INDICES,
    TEST_SLAB_REORIENT_LATTICE,
    TEST_SLAB_SCALE_FACTOR,
    TEST_SLAB_SHIFT,
    TEST_SLAB_TASK_DOC_COUNT,
)
from tests.helpers.jobflow import extract_outputs_of_type_from_run


@pytest.fixture
def adhesion_workflow_maker() -> AdhesionWorkflowMaker:
    slab_relax_maker_settings = build_qe_relax_maker_settings_for_test()
    interface_relax_maker_settings = build_qe_relax_maker_settings_for_test(
        constraints=(
            _build_external_faces_region_based_fixed_atom_selection_constraints_for_test()
        )
    )
    single_point_maker_settings = build_qe_single_point_maker_settings_for_test(
        qe_scf_simulation_settings=slab_relax_maker_settings.simulation_settings
    )
    adhesion_simulation_settings = AdhesionSimulationSettings(
        slab_relax_maker_settings=slab_relax_maker_settings,
        interface_relax_maker_settings=interface_relax_maker_settings,
        single_point_maker_settings=single_point_maker_settings,
    )
    return AdhesionWorkflowMaker(simulation_settings=adhesion_simulation_settings)


@pytest.fixture
def default_interface_assembly_settings() -> InterfaceAssemblySettings:
    return InterfaceAssemblySettings(
        offsets=DEFAULT_INTERFACE_OFFSETS,
        gaps=DEFAULT_INTERFACE_GAPS,
        vacuum=DEFAULT_INTERFACE_VACUUM,
    )


@pytest.fixture
def default_adhesion_inputs(
    default_interface_assembly_settings: InterfaceAssemblySettings,
) -> AdhesionWorkflowInputs:
    return AdhesionWorkflowInputs(
        top_slab=deepcopy(TEST_LINBO3_SLAB),
        bottom_slab=deepcopy(TEST_LINBO3_SLAB),
        assembly_settings=default_interface_assembly_settings,
    )


@pytest.fixture
def path_based_adhesion_inputs(
    default_interface_assembly_settings: InterfaceAssemblySettings,
) -> AdhesionWorkflowInputs:
    return AdhesionWorkflowInputs(
        top_slab=_load_test_slab_from_cif_path(SLAB_ASSEMBLY_TEST_TOP_PATH),
        bottom_slab=_load_test_slab_from_cif_path(SLAB_ASSEMBLY_TEST_BOTTOM_PATH),
        assembly_settings=default_interface_assembly_settings,
    )


@pytest.fixture
def mock_interface_objects() -> list[object]:
    return [object(), object()]


def _mock_qe_relax_properties(
    energy: float, structure: PMG_Structure
) -> RelaxProperties:
    step_structure = deepcopy(structure)
    step = SinglePointProperties(
        step=TEST_RELAX_STEP_INDEX,
        structure=step_structure,
        forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in step_structure],
        stress=TEST_RELAX_STRESS_VECTOR_ZERO,
        energy=energy,
    )
    return RelaxProperties(relax_steps=[step])


def _mock_qe_scf_properties(
    energy: float, structure: PMG_Structure
) -> SinglePointProperties:
    return SinglePointProperties(
        step=TEST_RELAX_STEP_INDEX,
        structure=deepcopy(structure),
        forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in structure],
        stress=TEST_RELAX_STRESS_VECTOR_ZERO,
        energy=energy,
    )


def _interface_area_in_angstrom_square_for_test(structure: PMG_Structure) -> float:
    lattice_vectors = np.array(structure.lattice.matrix, dtype=float)
    in_plane_vector_a = lattice_vectors[0]
    in_plane_vector_b = lattice_vectors[1]
    return float(np.linalg.norm(np.cross(in_plane_vector_a, in_plane_vector_b)))


def _process_run_info_success_for_test() -> ProcessRunInfo:
    return ProcessRunInfo(
        command_arguments=["true"],
        exit_code=PROCESS_EXIT_CODE_SUCCESS,
        standard_output_file=None,
        standard_error_file=None,
        error_message=None,
    )


def _structure_with_scaled_c_axis(scale_factor: float) -> PMG_Structure:
    lattice = np.array(TEST_LINBO3_SLAB.lattice.matrix, copy=True)
    lattice[TEST_C_AXIS_INDEX] = lattice[TEST_C_AXIS_INDEX] * scale_factor
    return PMG_Structure(
        lattice=Lattice(lattice),
        species=TEST_LINBO3_SLAB.species,
        coords=TEST_LINBO3_SLAB.frac_coords,
    )


def _write_mock_chargemol_net_atomic_charges_file(*, structure: PMG_Structure) -> None:
    file_path = Path.cwd() / CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME
    header_lines = [f"{len(structure)}", "Mock Chargemol net atomic charges"]
    atom_lines = [f"{site.specie.symbol} 0 0 0 0.0" for site in structure]
    file_path.write_text("\n".join([*header_lines, *atom_lines]) + "\n")
    cube_path = Path.cwd() / f"{QE_PP_CHARGE_DENSITY_BASENAME}.cube"
    cube_path.write_text("mock cube")


def _load_test_slab_from_cif_path(cif_path: Path) -> Slab:
    # We need to add a test that reads from the json schema also once we have that
    structure = PMG_Structure.from_file(cif_path)
    return Slab(
        lattice=structure.lattice,
        species=structure.species,
        coords=structure.frac_coords,
        miller_index=TEST_SLAB_MILLER_INDICES,
        oriented_unit_cell=structure,
        shift=TEST_SLAB_SHIFT,
        scale_factor=TEST_SLAB_SCALE_FACTOR,
        reorient_lattice=TEST_SLAB_REORIENT_LATTICE,
        site_properties=structure.site_properties,
        coords_are_cartesian=False,
    )


class _RelaxMakerForTest:
    def __init__(self, *, dir_name: str, energy: float) -> None:
        self.dir_name = dir_name
        self.energy = energy

    def make(self, structure: PMG_Structure) -> Job:
        return _mock_qe_relax_task_doc_job_with_dir_name(
            energy=self.energy, structure=deepcopy(structure), dir_name=self.dir_name
        )


class _QeSinglePointMakerForTest:
    def __init__(
        self, *, energy: float, dir_name_builder: Callable[[PMG_Structure], str]
    ) -> None:
        self.energy = energy
        self.dir_name_builder = dir_name_builder

    def make(self, structure: PMG_Structure) -> Job:
        return _mock_qe_scf_pp_chargemol_task_doc_job_with_dir_name(
            energy=self.energy,
            structure=deepcopy(structure),
            dir_name=self.dir_name_builder(structure),
        )


@job(output_schema=QeRelaxTaskDocument)
def _mock_qe_relax_task_doc_job(
    energy: float, structure: PMG_Structure | None = None
) -> QeRelaxTaskDocument:
    relax_structure = structure or TEST_LINBO3_SLAB
    return QeRelaxTaskDocument(
        dir_name=MOCK_TASK_DIR_NAME,
        properties=_mock_qe_relax_properties(energy, structure=relax_structure),
        errors="",
    )


@job(output_schema=QeRelaxTaskDocument)
def _mock_qe_relax_task_doc_job_with_dir_name(
    *, energy: float, dir_name: str, structure: PMG_Structure | None = None
) -> QeRelaxTaskDocument:
    relax_structure = structure or TEST_LINBO3_SLAB
    return QeRelaxTaskDocument(
        dir_name=dir_name,
        properties=_mock_qe_relax_properties(energy, structure=relax_structure),
        errors="",
    )


@job(output_schema=QeScfPpChargemolTaskDocument)
def _mock_qe_scf_pp_chargemol_task_doc_job(
    energy: float, structure: PMG_Structure | None = None
) -> QeScfPpChargemolTaskDocument:
    scf_structure = structure or TEST_LINBO3_SLAB
    return QeScfPpChargemolTaskDocument(
        dir_name=MOCK_TASK_DIR_NAME,
        errors="",
        properties=_mock_qe_scf_properties(energy, structure=scf_structure),
        structure=deepcopy(scf_structure),
        pp_run_info=_process_run_info_success_for_test(),
        chargemol_run_info=_process_run_info_success_for_test(),
    )


@job(output_schema=QeScfPpChargemolTaskDocument)
def _mock_qe_scf_pp_chargemol_task_doc_job_with_dir_name(
    *, energy: float, dir_name: str, structure: PMG_Structure | None = None
) -> QeScfPpChargemolTaskDocument:
    scf_structure = structure or TEST_LINBO3_SLAB
    return QeScfPpChargemolTaskDocument(
        dir_name=dir_name,
        errors="",
        properties=_mock_qe_scf_properties(energy, structure=scf_structure),
        structure=deepcopy(scf_structure),
        pp_run_info=_process_run_info_success_for_test(),
        chargemol_run_info=_process_run_info_success_for_test(),
    )


@job(output_schema=QeScfPpChargemolTaskDocument)
def _mock_failed_qe_scf_pp_chargemol_task_doc_job() -> QeScfPpChargemolTaskDocument:
    return QeScfPpChargemolTaskDocument(
        dir_name=MOCK_TASK_DIR_NAME,
        errors=MOCK_ERROR_LOG_MESSAGE,
        properties=None,
        structure=None,
        pp_run_info=None,
        chargemol_run_info=None,
    )


def _build_external_faces_region_based_fixed_atom_selection_constraints_for_test() -> (
    RelaxationConstraints
):
    return RelaxationConstraints(
        region_based_fixed_atom_selection_settings=ExternalFacesFreezeSettings(
            region_type=RegionBasedFixedAtomSelectionType.external_faces,
            thickness_in_angstrom=(
                TEST_INTERFACE_RELAX_EXTERNAL_FACES_FREEZE_THICKNESS_IN_ANGSTROM
            ),
        )
    )


def _mock_qe_relax_task_doc_job_for_call(*_args: object, **_kwargs: object) -> Job:
    return _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER)


def _mock_qe_scf_pp_chargemol_task_doc_job_for_call(
    *_args: object, **_kwargs: object
) -> Job:
    return _mock_qe_scf_pp_chargemol_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER)


def _extract_adhesion_workflow_result(
    run_flow: Mapping[str, Mapping[int, Response[Any]]],
) -> AdhesionWorkflowResult:
    results: list[AdhesionWorkflowResult] = []
    for responses_by_index in run_flow.values():
        results.extend(
            response.output
            for response in responses_by_index.values()
            if isinstance(response.output, AdhesionWorkflowResult)
        )
    assert len(results) == TEST_SINGLE_RESULT_COUNT
    result = results[TEST_FIRST_INDEX]
    assert isinstance(result, AdhesionWorkflowResult)
    return result


def test_adhesion_simulation_settings_uses_explicit_relax_maker_settings() -> None:
    slab_relax_constraints = RelaxationConstraints(
        fix_cell=False, fixed_atom_indices=[TEST_MANUAL_FIXED_ATOM_INDEX]
    )
    interface_relax_constraints = (
        _build_external_faces_region_based_fixed_atom_selection_constraints_for_test()
    )
    slab_relax_maker_settings = build_qe_relax_maker_settings_for_test(
        constraints=slab_relax_constraints
    )
    interface_relax_maker_settings = build_qe_relax_maker_settings_for_test(
        constraints=interface_relax_constraints
    )
    adhesion_simulation_settings = AdhesionSimulationSettings(
        slab_relax_maker_settings=slab_relax_maker_settings,
        interface_relax_maker_settings=interface_relax_maker_settings,
        single_point_maker_settings=(
            build_qe_single_point_maker_settings_for_test(
                qe_scf_simulation_settings=slab_relax_maker_settings.simulation_settings
            )
        ),
    )
    assert (
        adhesion_simulation_settings.slab_relax_maker_settings.simulation_settings.constraints
        == slab_relax_constraints
    )
    assert (
        adhesion_simulation_settings.interface_relax_maker_settings.simulation_settings.constraints
        == interface_relax_constraints
    )


def test_adhesion_simulation_settings_rejects_mlip_single_point_backend() -> None:
    with pytest.raises(ValidationError):
        AdhesionSimulationSettings(
            slab_relax_maker_settings=build_qe_relax_maker_settings_for_test(),
            interface_relax_maker_settings=build_qe_relax_maker_settings_for_test(),
            single_point_maker_settings=build_mlip_single_point_maker_settings_for_test(),
        )


@patch("crystal.flows.dft_adhesion_workflow_maker.build_single_point_maker")
@patch("crystal.flows.dft_adhesion_workflow_maker.build_relax_maker")
@patch(
    "crystal.flows.dft_adhesion_workflow_maker.InterfaceAssembler.generate_interfaces"
)
def test_workflow_uses_explicit_interface_relax_maker_settings_for_interface_relax_jobs(
    mock_generate_interfaces: MagicMock,
    mock_build_relax_maker: MagicMock,
    mock_build_single_point_maker: MagicMock,
    default_adhesion_inputs: AdhesionWorkflowInputs,
    tmp_path: Path,
) -> None:
    slab_relax_maker_settings = build_qe_relax_maker_settings_for_test()
    interface_relax_constraints = (
        _build_external_faces_region_based_fixed_atom_selection_constraints_for_test()
    )
    interface_relax_maker_settings = build_qe_relax_maker_settings_for_test(
        constraints=interface_relax_constraints
    )
    adhesion_simulation_settings = AdhesionSimulationSettings(
        slab_relax_maker_settings=slab_relax_maker_settings,
        interface_relax_maker_settings=interface_relax_maker_settings,
        single_point_maker_settings=(
            build_qe_single_point_maker_settings_for_test(
                qe_scf_simulation_settings=slab_relax_maker_settings.simulation_settings
            )
        ),
    )
    adhesion_workflow_maker = AdhesionWorkflowMaker(
        simulation_settings=adhesion_simulation_settings
    )
    mock_generate_interfaces.return_value = [deepcopy(TEST_LINBO3_SLAB)]

    def build_relax_maker_side_effect(*, relax_maker_settings: object) -> object:
        if relax_maker_settings == slab_relax_maker_settings:
            return _RelaxMakerForTest(
                dir_name="slab-relax", energy=TEST_RELAX_ENERGY_PLACEHOLDER
            )
        if relax_maker_settings == interface_relax_maker_settings:
            return _RelaxMakerForTest(
                dir_name="interface-relax-custom", energy=TEST_INTERFACE_ENERGY_SINGLE
            )
        pytest.fail("Unexpected relax maker settings passed to workflow")

    mock_build_relax_maker.side_effect = build_relax_maker_side_effect
    mock_build_single_point_maker.return_value = _QeSinglePointMakerForTest(
        energy=TEST_INTERFACE_ENERGY_SINGLE,
        dir_name_builder=lambda _structure: "single-point",
    )

    flow = adhesion_workflow_maker.make(default_adhesion_inputs)
    run_flow = run_locally(flow, root_dir=tmp_path, create_folders=True)
    result = _extract_adhesion_workflow_result(run_flow)

    assert result.slab_relax_dir_names == ["slab-relax", "slab-relax"]
    assert result.interface_relax_dir_names == ["interface-relax-custom"]
    assert result.interface_chargemol_dir_names == ["single-point"]


@patch("crystal.flows.dft_adhesion_workflow_maker.build_single_point_maker")
def test_build_slab_chargemol_jobs_before_interface_relax(
    mock_build_single_point_maker: MagicMock,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
    tmp_path: Path,
) -> None:
    top_structure = _structure_with_scaled_c_axis(TEST_RELAX_LATTICE_SCALE_UP)
    bottom_structure = _structure_with_scaled_c_axis(TEST_RELAX_LATTICE_SCALE_DOWN)
    base_c_axis_length = TEST_LINBO3_SLAB.lattice.c

    mock_build_single_point_maker.return_value = _QeSinglePointMakerForTest(
        energy=TEST_RELAX_ENERGY_PLACEHOLDER,
        dir_name_builder=lambda structure: (
            "top-single-point"
            if structure.lattice.c > base_c_axis_length
            else "bottom-single-point"
        ),
    )
    chargemol_executor_settings = ExecutorSettings(number_of_processes=6)
    slab_relax_maker_settings = (
        adhesion_workflow_maker.simulation_settings.slab_relax_maker_settings
    )
    assert isinstance(slab_relax_maker_settings, QeRelaxMakerSettings)
    single_point_maker_settings = build_qe_single_point_maker_settings_for_test(
        qe_scf_simulation_settings=slab_relax_maker_settings.simulation_settings,
        executor_settings=chargemol_executor_settings,
    )

    top_result, bottom_result = build_slab_chargemol_jobs_before_interface_relax(
        top_slab_relaxed_structure=top_structure,
        bottom_slab_relaxed_structure=bottom_structure,
        single_point_maker_settings=single_point_maker_settings,
    )

    run_flow = run_locally(
        Flow(
            jobs=[top_result, bottom_result],
            output=[top_result.output, bottom_result.output],
        ),
        root_dir=tmp_path,
        create_folders=True,
    )
    task_docs = extract_outputs_of_type_from_run(run_flow, QeScfPpChargemolTaskDocument)
    structures_by_dir_name: dict[str, PMG_Structure | None] = {}
    for task_doc in task_docs:
        assert task_doc.dir_name is not None
        structures_by_dir_name[task_doc.dir_name] = task_doc.structure

    assert sorted(structures_by_dir_name) == ["bottom-single-point", "top-single-point"]
    assert structures_by_dir_name["top-single-point"] is not None
    assert structures_by_dir_name["bottom-single-point"] is not None
    assert (
        structures_by_dir_name["top-single-point"].as_dict() == top_structure.as_dict()
    )
    assert (
        structures_by_dir_name["bottom-single-point"].as_dict()
        == bottom_structure.as_dict()
    )


def test_relax_properties_converged() -> None:
    relax_properties = _mock_qe_relax_properties(
        TEST_RELAX_ENERGY_PLACEHOLDER, TEST_LINBO3_SLAB
    )
    assert relax_properties_converged([relax_properties])
    assert not relax_properties_converged([None, relax_properties])


def test_extract_relax_final_structure_or_raise() -> None:
    relax_properties = _mock_qe_relax_properties(
        TEST_RELAX_ENERGY_PLACEHOLDER, TEST_LINBO3_SLAB
    )
    structure = extract_relax_final_structure_or_raise(
        relax_properties, SLAB_RELAX_NOT_CONVERGED_ERROR
    )
    assert structure.as_dict() == relax_properties.relaxed_structure.as_dict()


def test_extract_relax_final_structure_or_raise_raises_for_missing_properties() -> None:
    with pytest.raises(ValueError, match=SLAB_RELAX_NOT_CONVERGED_ERROR):
        _ = extract_relax_final_structure_or_raise(None, SLAB_RELAX_NOT_CONVERGED_ERROR)


def test_split_interfaces_into_batches() -> None:
    interfaces = [deepcopy(TEST_LINBO3_SLAB) for _ in range(TEST_INTERFACE_COUNT)]
    batches = split_items_into_batches(
        interfaces, items_per_batch=TEST_INTERFACE_JOBS_PER_BATCH
    )
    assert [len(batch) for batch in batches] == TEST_INTERFACE_BATCH_SIZES


def test_extract_relaxed_slabs_after_slab_relax() -> None:
    base_lattice = np.array(TEST_LINBO3_SLAB.lattice.matrix, copy=True)
    top_lattice = base_lattice.copy()
    top_lattice[TEST_C_AXIS_INDEX] = (
        top_lattice[TEST_C_AXIS_INDEX] * TEST_RELAX_LATTICE_SCALE_UP
    )
    bottom_lattice = base_lattice.copy()
    bottom_lattice[TEST_C_AXIS_INDEX] = (
        bottom_lattice[TEST_C_AXIS_INDEX] * TEST_RELAX_LATTICE_SCALE_DOWN
    )
    top_structure = PMG_Structure(
        lattice=Lattice(top_lattice),
        species=TEST_LINBO3_SLAB.species,
        coords=TEST_LINBO3_SLAB.frac_coords,
    )
    bottom_structure = PMG_Structure(
        lattice=Lattice(bottom_lattice),
        species=TEST_LINBO3_SLAB.species,
        coords=TEST_LINBO3_SLAB.frac_coords,
    )
    slab_relax_properties = [
        _mock_qe_relax_properties(TEST_RELAX_ENERGY_LOW, top_structure),
        _mock_qe_relax_properties(TEST_RELAX_ENERGY_HIGH, bottom_structure),
    ]

    top_relaxed, bottom_relaxed = extract_relaxed_slabs_after_slab_relax(
        slab_relax_properties
    )

    assert top_relaxed.as_dict() == top_structure.as_dict()
    assert bottom_relaxed.as_dict() == bottom_structure.as_dict()


@patch("crystal.flows.dft_adhesion_workflow_maker.build_single_point_maker")
@patch("crystal.flows.dft_adhesion_workflow_maker.build_relax_maker")
@patch(
    "crystal.flows.dft_adhesion_workflow_maker.InterfaceAssembler.generate_interfaces"
)
def test_workflow_passes_executor_settings_for_relax_and_chargemol(
    mock_generate_interfaces: MagicMock,
    mock_build_relax_maker: MagicMock,
    mock_build_single_point_maker: MagicMock,
    default_adhesion_inputs: AdhesionWorkflowInputs,
    tmp_path: Path,
) -> None:
    slab_relax_number_of_processes = 2
    interface_relax_number_of_processes = 3
    chargemol_number_of_processes = 4
    slab_relax_dir_name = f"slab-relax-{slab_relax_number_of_processes}"
    interface_relax_dir_name = f"interface-relax-{interface_relax_number_of_processes}"
    single_point_dir_name = f"single-point-{chargemol_number_of_processes}"
    slab_relax_maker_settings = build_qe_relax_maker_settings_for_test(
        executor_settings=ExecutorSettings(
            number_of_processes=slab_relax_number_of_processes
        )
    )
    interface_relax_maker_settings = build_qe_relax_maker_settings_for_test(
        constraints=(
            _build_external_faces_region_based_fixed_atom_selection_constraints_for_test()
        ),
        executor_settings=ExecutorSettings(
            number_of_processes=interface_relax_number_of_processes
        ),
    )
    single_point_maker_settings = build_qe_single_point_maker_settings_for_test(
        qe_scf_simulation_settings=slab_relax_maker_settings.simulation_settings,
        executor_settings=ExecutorSettings(
            number_of_processes=chargemol_number_of_processes
        ),
    )
    adhesion_workflow_maker = AdhesionWorkflowMaker(
        simulation_settings=AdhesionSimulationSettings(
            slab_relax_maker_settings=slab_relax_maker_settings,
            interface_relax_maker_settings=interface_relax_maker_settings,
            single_point_maker_settings=single_point_maker_settings,
        )
    )

    def build_relax_maker_side_effect(*, relax_maker_settings: object) -> object:
        if relax_maker_settings == slab_relax_maker_settings:
            return _RelaxMakerForTest(
                dir_name=slab_relax_dir_name, energy=TEST_RELAX_ENERGY_PLACEHOLDER
            )
        if relax_maker_settings == interface_relax_maker_settings:
            return _RelaxMakerForTest(
                dir_name=interface_relax_dir_name, energy=TEST_RELAX_ENERGY_PLACEHOLDER
            )
        pytest.fail("Unexpected relax maker settings passed to workflow")

    mock_generate_interfaces.return_value = [deepcopy(TEST_LINBO3_SLAB)]
    mock_build_relax_maker.side_effect = build_relax_maker_side_effect
    mock_build_single_point_maker.return_value = _QeSinglePointMakerForTest(
        energy=TEST_RELAX_ENERGY_PLACEHOLDER,
        dir_name_builder=lambda _structure: single_point_dir_name,
    )

    flow = adhesion_workflow_maker.make(default_adhesion_inputs)
    run_flow = run_locally(flow, root_dir=tmp_path, create_folders=True)
    result = _extract_adhesion_workflow_result(run_flow)

    assert result.slab_relax_dir_names == [slab_relax_dir_name, slab_relax_dir_name]
    assert result.interface_relax_dir_names == [interface_relax_dir_name]
    assert result.slab_chargemol_dir_names == [
        single_point_dir_name,
        single_point_dir_name,
    ]
    assert result.interface_chargemol_dir_names == [single_point_dir_name]


@patch("crystal.flows.dft_adhesion_workflow_maker.build_single_point_maker")
@patch("crystal.flows.dft_adhesion_workflow_maker.build_relax_maker")
@patch(
    "crystal.flows.dft_adhesion_workflow_maker.InterfaceAssembler.generate_interfaces"
)
def test_workflow_uses_interface_settings(
    mock_generate_interfaces: MagicMock,
    mock_build_relax_maker: MagicMock,
    mock_build_single_point_maker: MagicMock,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
    tmp_path: Path,
) -> None:
    offsets = TEST_INTERFACE_OFFSETS_CUSTOM
    gaps = TEST_INTERFACE_GAPS_CUSTOM
    gap_multipliers = TEST_INTERFACE_GAP_MULTIPLIERS_CUSTOM
    vacuum = TEST_INTERFACE_VACUUM_CUSTOM
    assembly_settings = InterfaceAssemblySettings(
        offsets=offsets, gaps=gaps, gap_multipliers=gap_multipliers, vacuum=vacuum
    )
    inputs = AdhesionWorkflowInputs(
        top_slab=deepcopy(TEST_LINBO3_SLAB),
        bottom_slab=deepcopy(TEST_LINBO3_SLAB),
        assembly_settings=assembly_settings,
    )

    def generate_interfaces_side_effect(**kwargs: object) -> list[PMG_Structure]:
        if kwargs == {
            "offsets": offsets,
            "gaps": gaps,
            "gap_multipliers": gap_multipliers,
            "vacuum": vacuum,
        }:
            return [deepcopy(TEST_LINBO3_SLAB)]
        return [deepcopy(TEST_LINBO3_SLAB), deepcopy(TEST_LINBO3_SLAB)]

    mock_generate_interfaces.side_effect = generate_interfaces_side_effect
    mock_build_relax_maker.side_effect = [
        _RelaxMakerForTest(dir_name="slab-relax", energy=TEST_RELAX_ENERGY_PLACEHOLDER),
        _RelaxMakerForTest(
            dir_name="interface-relax", energy=TEST_RELAX_ENERGY_PLACEHOLDER
        ),
    ]
    mock_build_single_point_maker.return_value = _QeSinglePointMakerForTest(
        energy=TEST_RELAX_ENERGY_PLACEHOLDER,
        dir_name_builder=lambda _structure: "single-point",
    )

    flow = adhesion_workflow_maker.make(inputs)
    run_flow = run_locally(flow, root_dir=tmp_path, create_folders=True)
    result = _extract_adhesion_workflow_result(run_flow)

    assert len(result.interface_relax_dir_names) == 1
    assert len(result.interface_chargemol_dir_names) == 1
    assert len(result.adhesion_energy_records) == 1


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.QeScfPpChargemolMaker.make")
@patch("crystal.calculators.qe.relax_maker.QeRelaxMaker.make")
@patch(
    "crystal.flows.dft_adhesion_workflow_maker.InterfaceAssembler.generate_interfaces"
)
@run_in_temp_dir
def test_adhesion_energy_formula(
    mock_generate_interfaces: MagicMock,
    mock_relax_make: MagicMock,
    mock_chargemol_make: MagicMock,
    default_adhesion_inputs: AdhesionWorkflowInputs,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
    mock_interface_objects: list[object],
) -> None:
    slab_energy_a = TEST_RELAX_ENERGY_LOW
    slab_energy_b = TEST_RELAX_ENERGY_HIGH
    interface_energy_a = TEST_INTERFACE_ENERGY_A
    interface_energy_b = TEST_INTERFACE_ENERGY_B
    relax_jobs = [
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
    ]
    chargemol_jobs = [
        _mock_qe_scf_pp_chargemol_task_doc_job(slab_energy_a),
        _mock_qe_scf_pp_chargemol_task_doc_job(slab_energy_b),
        _mock_qe_scf_pp_chargemol_task_doc_job(interface_energy_a),
        _mock_qe_scf_pp_chargemol_task_doc_job(interface_energy_b),
    ]
    mock_generate_interfaces.return_value = mock_interface_objects
    mock_relax_make.side_effect = relax_jobs
    mock_chargemol_make.side_effect = chargemol_jobs

    flow = adhesion_workflow_maker.make(default_adhesion_inputs)
    run_flow = run_locally(flow)
    result = _extract_adhesion_workflow_result(run_flow)
    energies = [record.adhesion_energy for record in result.adhesion_energy_records]
    expected_energies = [
        interface_energy_a - (slab_energy_a + slab_energy_b),
        interface_energy_b - (slab_energy_a + slab_energy_b),
    ]
    assert energies == pytest.approx(expected_energies)


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.QeScfPpChargemolMaker.make")
@patch("crystal.calculators.qe.relax_maker.QeRelaxMaker.make")
@patch(
    "crystal.flows.dft_adhesion_workflow_maker.InterfaceAssembler.generate_interfaces"
)
@run_in_temp_dir
def test_workflow_result_schema(
    mock_generate_interfaces: MagicMock,
    mock_relax_make: MagicMock,
    mock_chargemol_make: MagicMock,
    default_adhesion_inputs: AdhesionWorkflowInputs,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
) -> None:
    relax_jobs = [
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
    ]
    chargemol_jobs = [
        _mock_qe_scf_pp_chargemol_task_doc_job(TEST_RELAX_ENERGY_LOW),
        _mock_qe_scf_pp_chargemol_task_doc_job(TEST_RELAX_ENERGY_HIGH),
        _mock_qe_scf_pp_chargemol_task_doc_job(TEST_INTERFACE_ENERGY_SINGLE),
    ]
    mock_generate_interfaces.return_value = [object()]
    mock_relax_make.side_effect = relax_jobs
    mock_chargemol_make.side_effect = chargemol_jobs

    flow = adhesion_workflow_maker.make(default_adhesion_inputs)
    run_flow = run_locally(flow)
    result = _extract_adhesion_workflow_result(run_flow)
    assert len(result.slab_relax_dir_names) == TEST_SLAB_TASK_DOC_COUNT
    assert len(result.interface_relax_dir_names) == TEST_INTERFACE_TASK_DOC_COUNT_SINGLE
    assert len(result.slab_chargemol_dir_names) == TEST_SLAB_TASK_DOC_COUNT
    assert (
        len(result.interface_chargemol_dir_names)
        == TEST_INTERFACE_TASK_DOC_COUNT_SINGLE
    )
    assert all(
        dir_name == MOCK_TASK_DIR_NAME for dir_name in result.slab_relax_dir_names
    )
    assert all(
        dir_name == MOCK_TASK_DIR_NAME for dir_name in result.interface_relax_dir_names
    )
    assert all(
        dir_name == MOCK_TASK_DIR_NAME for dir_name in result.slab_chargemol_dir_names
    )
    assert all(
        dir_name == MOCK_TASK_DIR_NAME
        for dir_name in result.interface_chargemol_dir_names
    )
    assert len(result.adhesion_energy_records) == TEST_ADHESION_RECORD_COUNT_SINGLE


def test_adhesion_workflow_result_rejects_mismatched_interface_counts() -> None:
    with pytest.raises(ValueError, match=INTERFACE_RECORD_COUNT_MISMATCH_ERROR):
        _ = AdhesionWorkflowResult(
            slab_relax_dir_names=[MOCK_TASK_DIR_NAME] * TEST_SLAB_TASK_DOC_COUNT,
            interface_relax_dir_names=[MOCK_TASK_DIR_NAME]
            * TEST_INTERFACE_TASK_DOC_COUNT_MULTI,
            slab_chargemol_dir_names=[MOCK_TASK_DIR_NAME] * TEST_SLAB_TASK_DOC_COUNT,
            interface_chargemol_dir_names=[MOCK_TASK_DIR_NAME]
            * TEST_INTERFACE_TASK_DOC_COUNT_MULTI,
            adhesion_energy_records=[
                AdhesionEnergyRecord(
                    interface_index=TEST_FIRST_INDEX,
                    adhesion_energy=TEST_RELAX_ENERGY_PLACEHOLDER,
                    interface_area_in_angstrom_square=(
                        _interface_area_in_angstrom_square_for_test(TEST_LINBO3_SLAB)
                    ),
                )
            ],
        )


def test_extract_relaxed_slabs_after_slab_relax_uses_default_error_message() -> None:
    slab_relax_properties = [None, None]
    with pytest.raises(ValueError, match=SLAB_RELAX_NOT_CONVERGED_ERROR):
        extract_relaxed_slabs_after_slab_relax(slab_relax_properties)


def test_build_adhesion_energy_records_uses_default_slab_error_message() -> None:
    slab_task_docs = [
        QeScfPpChargemolTaskDocument(
            dir_name=MOCK_TASK_DIR_NAME,
            errors=MOCK_ERROR_LOG_MESSAGE,
            properties=None,
            structure=None,
            pp_run_info=None,
            chargemol_run_info=None,
        )
    ]
    interface_task_docs = [
        QeScfPpChargemolTaskDocument(
            dir_name=MOCK_TASK_DIR_NAME,
            errors="",
            properties=_mock_qe_scf_properties(0.0, TEST_LINBO3_SLAB),
            structure=deepcopy(TEST_LINBO3_SLAB),
            pp_run_info=_process_run_info_success_for_test(),
            chargemol_run_info=_process_run_info_success_for_test(),
        )
    ]
    with pytest.raises(ValueError, match=SLAB_CHARGEMOL_NOT_CONVERGED_ERROR):
        build_adhesion_energy_records_from_single_point_task_docs(
            slab_task_docs=slab_task_docs, interface_task_docs=interface_task_docs
        )


def test_build_adhesion_energy_records_sets_interface_area_and_energy_per_area() -> (
    None
):
    slab_structure = deepcopy(TEST_LINBO3_SLAB)
    interface_structure = deepcopy(TEST_LINBO3_SLAB)
    slab_task_docs = [
        QeScfPpChargemolTaskDocument(
            dir_name=MOCK_TASK_DIR_NAME,
            errors="",
            properties=_mock_qe_scf_properties(TEST_RELAX_ENERGY_LOW, slab_structure),
            structure=slab_structure,
            pp_run_info=_process_run_info_success_for_test(),
            chargemol_run_info=_process_run_info_success_for_test(),
        ),
        QeScfPpChargemolTaskDocument(
            dir_name=MOCK_TASK_DIR_NAME,
            errors="",
            properties=_mock_qe_scf_properties(TEST_RELAX_ENERGY_HIGH, slab_structure),
            structure=slab_structure,
            pp_run_info=_process_run_info_success_for_test(),
            chargemol_run_info=_process_run_info_success_for_test(),
        ),
    ]
    interface_energy = TEST_INTERFACE_ENERGY_SINGLE
    interface_task_docs = [
        QeScfPpChargemolTaskDocument(
            dir_name=MOCK_TASK_DIR_NAME,
            errors="",
            properties=_mock_qe_scf_properties(interface_energy, interface_structure),
            structure=interface_structure,
            pp_run_info=_process_run_info_success_for_test(),
            chargemol_run_info=_process_run_info_success_for_test(),
        )
    ]

    interface_area_in_angstrom_square = _interface_area_in_angstrom_square_for_test(
        interface_structure
    )
    slab_energy_sum = TEST_RELAX_ENERGY_LOW + TEST_RELAX_ENERGY_HIGH
    expected_adhesion_energy = interface_energy - slab_energy_sum
    expected_energy_per_area = (
        expected_adhesion_energy / interface_area_in_angstrom_square
    )

    records = build_adhesion_energy_records_from_single_point_task_docs(
        slab_task_docs=slab_task_docs, interface_task_docs=interface_task_docs
    )

    assert len(records) == 1
    record = records[0]
    assert np.isclose(
        record.interface_area_in_angstrom_square, interface_area_in_angstrom_square
    )
    assert np.isclose(record.adhesion_energy_per_area, expected_energy_per_area)


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.QeScfPpChargemolMaker.make")
@patch("crystal.calculators.qe.relax_maker.QeRelaxMaker.make")
@patch(
    "crystal.flows.dft_adhesion_workflow_maker.InterfaceAssembler.generate_interfaces"
)
@run_in_temp_dir
def test_interface_failure_handling(
    mock_generate_interfaces: MagicMock,
    mock_relax_make: MagicMock,
    mock_chargemol_make: MagicMock,
    default_adhesion_inputs: AdhesionWorkflowInputs,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
) -> None:
    relax_jobs = [
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
    ]
    chargemol_jobs = [
        _mock_qe_scf_pp_chargemol_task_doc_job(TEST_RELAX_ENERGY_LOW),
        _mock_qe_scf_pp_chargemol_task_doc_job(TEST_RELAX_ENERGY_HIGH),
        _mock_failed_qe_scf_pp_chargemol_task_doc_job(),
    ]
    mock_generate_interfaces.return_value = [object()]
    mock_relax_make.side_effect = relax_jobs
    mock_chargemol_make.side_effect = chargemol_jobs

    flow = adhesion_workflow_maker.make(default_adhesion_inputs)
    with pytest.raises(ValueError, match=INTERFACE_CHARGEMOL_NOT_CONVERGED_ERROR):
        run_locally(flow, raise_immediately=True)


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.QeScfPpChargemolMaker.make")
@patch("crystal.calculators.qe.relax_maker.QeRelaxMaker.make")
@patch(
    "crystal.flows.dft_adhesion_workflow_maker.InterfaceAssembler.generate_interfaces"
)
@run_in_temp_dir
def test_slab_failure_stops_interface_jobs(
    mock_generate_interfaces: MagicMock,
    mock_relax_make: MagicMock,
    mock_chargemol_make: MagicMock,
    default_adhesion_inputs: AdhesionWorkflowInputs,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
) -> None:
    relax_jobs = [
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
    ]
    chargemol_jobs = [
        _mock_failed_qe_scf_pp_chargemol_task_doc_job(),
        _mock_qe_scf_pp_chargemol_task_doc_job(TEST_RELAX_ENERGY_HIGH),
    ]
    mock_generate_interfaces.return_value = [object()]
    mock_relax_make.side_effect = relax_jobs
    mock_chargemol_make.side_effect = chargemol_jobs

    flow = adhesion_workflow_maker.make(default_adhesion_inputs)
    run_flow = run_locally(flow)

    adhesion_results = extract_outputs_of_type_from_run(
        run_flow, AdhesionWorkflowResult
    )
    assert adhesion_results == []


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.QeScfPpChargemolMaker.make")
@patch("crystal.calculators.qe.relax_maker.QeRelaxMaker.make")
@run_in_temp_dir
def test_workflow_with_real_interface_assembler(
    mock_relax_make: MagicMock,
    mock_chargemol_make: MagicMock,
    path_based_adhesion_inputs: AdhesionWorkflowInputs,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
) -> None:
    relax_jobs = [
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
        _mock_qe_relax_task_doc_job(TEST_RELAX_ENERGY_PLACEHOLDER),
    ]
    chargemol_jobs = [
        _mock_qe_scf_pp_chargemol_task_doc_job(TEST_RELAX_ENERGY_LOW),
        _mock_qe_scf_pp_chargemol_task_doc_job(TEST_RELAX_ENERGY_HIGH),
        _mock_qe_scf_pp_chargemol_task_doc_job(TEST_INTERFACE_ENERGY_SINGLE),
    ]
    mock_relax_make.side_effect = relax_jobs
    mock_chargemol_make.side_effect = chargemol_jobs

    flow = adhesion_workflow_maker.make(path_based_adhesion_inputs)
    run_flow = run_locally(flow)
    result = _extract_adhesion_workflow_result(run_flow)
    assert len(result.slab_relax_dir_names) == TEST_SLAB_TASK_DOC_COUNT
    assert len(result.interface_relax_dir_names) == TEST_INTERFACE_TASK_DOC_COUNT_SINGLE
    assert len(result.slab_chargemol_dir_names) == TEST_SLAB_TASK_DOC_COUNT
    assert (
        len(result.interface_chargemol_dir_names)
        == TEST_INTERFACE_TASK_DOC_COUNT_SINGLE
    )
    assert len(result.adhesion_energy_records) == TEST_ADHESION_RECORD_COUNT_SINGLE


@pytest.mark.needs_third_party_executables
@run_in_temp_dir
def test_adhesion_workflow_with_real_qe(
    path_based_adhesion_inputs: AdhesionWorkflowInputs,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
) -> None:
    qe_dir = CRYSTAL_CONFIG.programs.quantum_espresso_dir
    assert qe_dir is not None, QE_NOT_CONFIGURED_MESSAGE
    assert (qe_dir / QE_PW_EXECUTABLE_NAME).exists(), QE_NOT_CONFIGURED_MESSAGE
    assert (qe_dir / QE_PP_EXECUTABLE_NAME).exists(), QE_NOT_CONFIGURED_MESSAGE
    required_elements = {
        *{el.symbol for el in path_based_adhesion_inputs.top_slab.composition.elements},
        *{
            el.symbol
            for el in path_based_adhesion_inputs.bottom_slab.composition.elements
        },
    }
    slab_relax_maker_settings = (
        adhesion_workflow_maker.simulation_settings.slab_relax_maker_settings
    )
    assert isinstance(slab_relax_maker_settings, QeRelaxMakerSettings)
    provided_pseudos = set(
        slab_relax_maker_settings.simulation_settings.pseudopotentials
    )
    assert required_elements.issubset(provided_pseudos), QE_MISSING_PSEUDOS_MESSAGE
    chargemol_dir = CRYSTAL_CONFIG.programs.chargemol_dir
    assert chargemol_dir is not None, CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE
    chargemol_executable = chargemol_dir / CHARGEMOL_EXECUTABLE_RELATIVE_PATH
    assert chargemol_executable.exists(), CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE
    flow = adhesion_workflow_maker.make(path_based_adhesion_inputs)
    run_flow = run_locally(flow)
    result = _extract_adhesion_workflow_result(run_flow)
    assert len(result.slab_relax_dir_names) == TEST_SLAB_TASK_DOC_COUNT
    assert len(result.interface_relax_dir_names) == TEST_INTERFACE_TASK_DOC_COUNT_SINGLE
    assert len(result.slab_chargemol_dir_names) == TEST_SLAB_TASK_DOC_COUNT
    assert (
        len(result.interface_chargemol_dir_names)
        == TEST_INTERFACE_TASK_DOC_COUNT_SINGLE
    )
    assert len(result.adhesion_energy_records) == TEST_ADHESION_RECORD_COUNT_SINGLE


@patch("crystal.flows.dft_adhesion_workflow_maker.InterfaceAssembler")
@patch("crystal.calculators.qe.scf_pp_chargemol_maker.QeScfPpChargemolMaker.make")
@patch("crystal.calculators.qe.relax_maker.QeRelaxMaker.make")
@run_in_temp_dir
def test_interfaces_built_from_relaxed_slabs(
    mock_relax_make: MagicMock,
    mock_chargemol_make: MagicMock,
    mock_interface_assembler: MagicMock,
    default_adhesion_inputs: AdhesionWorkflowInputs,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
) -> None:
    base_lattice = np.array(TEST_LINBO3_SLAB.lattice.matrix, copy=True)
    top_relaxed_lattice_matrix = base_lattice.copy()
    top_relaxed_lattice_matrix[TEST_C_AXIS_INDEX] = (
        top_relaxed_lattice_matrix[TEST_C_AXIS_INDEX] * TEST_RELAX_LATTICE_SCALE_UP
    )
    bottom_relaxed_lattice_matrix = base_lattice.copy()
    bottom_relaxed_lattice_matrix[TEST_C_AXIS_INDEX] = (
        bottom_relaxed_lattice_matrix[TEST_C_AXIS_INDEX] * TEST_RELAX_LATTICE_SCALE_DOWN
    )
    top_relaxed_structure = PMG_Structure(
        lattice=Lattice(top_relaxed_lattice_matrix),
        species=TEST_LINBO3_SLAB.species,
        coords=TEST_LINBO3_SLAB.frac_coords,
    )
    bottom_relaxed_structure = PMG_Structure(
        lattice=Lattice(bottom_relaxed_lattice_matrix),
        species=TEST_LINBO3_SLAB.species,
        coords=TEST_LINBO3_SLAB.frac_coords,
    )
    top_chargemol_lattice_matrix = base_lattice.copy()
    top_chargemol_lattice_matrix[TEST_C_AXIS_INDEX] = top_chargemol_lattice_matrix[
        TEST_C_AXIS_INDEX
    ] * (TEST_RELAX_LATTICE_SCALE_UP + 0.01)
    bottom_chargemol_lattice_matrix = base_lattice.copy()
    bottom_chargemol_lattice_matrix[TEST_C_AXIS_INDEX] = (
        bottom_chargemol_lattice_matrix[TEST_C_AXIS_INDEX]
        * (TEST_RELAX_LATTICE_SCALE_DOWN - 0.01)
    )
    top_chargemol_structure = PMG_Structure(
        lattice=Lattice(top_chargemol_lattice_matrix),
        species=TEST_LINBO3_SLAB.species,
        coords=TEST_LINBO3_SLAB.frac_coords,
    )
    bottom_chargemol_structure = PMG_Structure(
        lattice=Lattice(bottom_chargemol_lattice_matrix),
        species=TEST_LINBO3_SLAB.species,
        coords=TEST_LINBO3_SLAB.frac_coords,
    )
    relax_jobs = [
        _mock_qe_relax_task_doc_job(
            TEST_RELAX_ENERGY_LOW, structure=top_relaxed_structure
        ),
        _mock_qe_relax_task_doc_job(
            TEST_RELAX_ENERGY_HIGH, structure=bottom_relaxed_structure
        ),
    ]
    chargemol_jobs = [
        _mock_qe_scf_pp_chargemol_task_doc_job(
            TEST_RELAX_ENERGY_LOW, structure=top_chargemol_structure
        ),
        _mock_qe_scf_pp_chargemol_task_doc_job(
            TEST_RELAX_ENERGY_HIGH, structure=bottom_chargemol_structure
        ),
    ]
    observed = {}
    mock_relax_make.side_effect = relax_jobs
    mock_chargemol_make.side_effect = chargemol_jobs

    class _RecordingAssembler:
        def __init__(self, top_slab: Slab, bottom_slab: Slab) -> None:
            observed[TEST_INTERFACE_ASSEMBLER_TOP_LATTICE_KEY] = top_slab.lattice.matrix
            observed[TEST_INTERFACE_ASSEMBLER_BOTTOM_LATTICE_KEY] = (
                bottom_slab.lattice.matrix
            )

        def generate_interfaces(self, **_kwargs: Any) -> list[object]:
            return []

    mock_interface_assembler.side_effect = _RecordingAssembler

    flow = adhesion_workflow_maker.make(default_adhesion_inputs)
    run_locally(flow)
    assert np.allclose(
        observed[TEST_INTERFACE_ASSEMBLER_TOP_LATTICE_KEY],
        top_chargemol_structure.lattice.matrix,
    )
    assert np.allclose(
        observed[TEST_INTERFACE_ASSEMBLER_BOTTOM_LATTICE_KEY],
        bottom_chargemol_structure.lattice.matrix,
    )


@patch("crystal.helpers.chargemol.chargemol_runner.ChargemolRunner.run")
@patch("crystal.calculators.qe.qe_pp_runner.QePpChargeDensityRunner.run")
@patch("crystal.calculators.qe.scf_pp_chargemol_maker.run_qe_scf")
@patch("crystal.calculators.qe.relax_maker.run_qe_relax")
@run_in_temp_dir
def test_adhesion_workflow_with_mocked_qe_and_chargemol_runners(
    mock_run_qe_relax: MagicMock,
    mock_run_qe_scf: MagicMock,
    mock_run_qe_pp: MagicMock,
    mock_run_chargemol: MagicMock,
    default_adhesion_inputs: AdhesionWorkflowInputs,
    adhesion_workflow_maker: AdhesionWorkflowMaker,
) -> None:
    relax_properties = _mock_qe_relax_properties(
        TEST_RELAX_ENERGY_PLACEHOLDER, TEST_LINBO3_SLAB
    )
    scf_properties = _mock_qe_scf_properties(
        TEST_INTERFACE_ENERGY_SINGLE, TEST_LINBO3_SLAB
    )
    mock_run_qe_relax.return_value = relax_properties
    mock_run_qe_scf.return_value = scf_properties
    mock_run_qe_pp.return_value = _process_run_info_success_for_test()

    def _mock_chargemol_run_with_file_write(
        *_args: object, **_kwargs: object
    ) -> ProcessRunInfo:
        _write_mock_chargemol_net_atomic_charges_file(structure=TEST_LINBO3_SLAB)
        return _process_run_info_success_for_test()

    mock_run_chargemol.side_effect = _mock_chargemol_run_with_file_write

    flow = adhesion_workflow_maker.make(default_adhesion_inputs)
    run_flow = run_locally(flow)
    result = _extract_adhesion_workflow_result(run_flow)
    assert len(result.slab_relax_dir_names) == TEST_SLAB_TASK_DOC_COUNT
    assert len(result.interface_relax_dir_names) == TEST_INTERFACE_TASK_DOC_COUNT_SINGLE
    assert len(result.slab_chargemol_dir_names) == TEST_SLAB_TASK_DOC_COUNT
    assert (
        len(result.interface_chargemol_dir_names)
        == TEST_INTERFACE_TASK_DOC_COUNT_SINGLE
    )
    assert len(result.adhesion_energy_records) == TEST_ADHESION_RECORD_COUNT_SINGLE
