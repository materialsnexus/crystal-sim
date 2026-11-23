from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jobflow.core.job import job
from jobflow.managers.local import run_locally
from pydantic import BaseModel
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.flows.dft_adhesion_workflow_maker import AdhesionWorkflowMaker
from crystal.schemas.adhesion import (
    AdhesionSimulationSettings,
    AdhesionWorkflowInputs,
    AdhesionWorkflowResult,
)
from crystal.schemas.backend import SinglePointBackend
from crystal.schemas.gpaw import (
    GpawChargemolTaskDocument,
    GpawScfChargemolMakerSettings,
)
from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from crystal.schemas.qe import QeScfPpChargemolTaskDocument
from crystal.schemas.qe_pp import QeScfPpChargemolMakerSettings
from crystal.schemas.relaxation import RelaxMakerSettings
from crystal.structure.interfaces.settings import InterfaceAssemblySettings
from tests.helpers.builders.process_run_info import (
    build_successful_process_run_info_for_test,
)
from tests.helpers.builders.properties import build_relax_properties_for_test
from tests.helpers.builders.relax_maker_settings import (
    build_gpaw_relax_maker_settings_for_test,
    build_mlip_relax_maker_settings_for_test,
    build_qe_relax_maker_settings_for_test,
)
from tests.helpers.builders.single_point_maker_settings import (
    build_gpaw_single_point_maker_settings_for_test,
    build_qe_single_point_maker_settings_for_test,
)
from tests.helpers.constants.structure import (
    DEFAULT_INTERFACE_GAPS,
    DEFAULT_INTERFACE_OFFSETS,
    DEFAULT_INTERFACE_VACUUM,
    TEST_ADHESION_RECORD_COUNT_SINGLE,
    TEST_FIRST_INDEX,
    TEST_INTERFACE_ENERGY_SINGLE,
    TEST_LINBO3_SLAB,
    TEST_RELAX_ENERGY_HIGH,
    TEST_RELAX_ENERGY_LOW,
    TEST_RELAX_STEP_INDEX,
    TEST_RELAX_STRESS_VECTOR_ZERO,
    TEST_RELAX_ZERO_FORCE_VECTOR,
)
from tests.helpers.jobflow import extract_outputs_of_type_from_run

TEST_SLAB_TOP_RELAX_DIR_NAME = "slab-top-relax"
TEST_SLAB_BOTTOM_RELAX_DIR_NAME = "slab-bottom-relax"
TEST_INTERFACE_RELAX_DIR_NAME = "interface-relax"
TEST_SLAB_TOP_SCF_AND_CHARGES_DIR_NAME = "slab-top-scf-and-charges"
TEST_SLAB_BOTTOM_SCF_AND_CHARGES_DIR_NAME = "slab-bottom-scf-and-charges"
TEST_INTERFACE_SCF_AND_CHARGES_DIR_NAME = "interface-scf-and-charges"
TEST_GPAW_SCF_AND_CHARGES_TOP_DIR_NAME = "gpaw-top"
TEST_GPAW_SCF_AND_CHARGES_BOTTOM_DIR_NAME = "gpaw-bottom"
TEST_GPAW_SCF_AND_CHARGES_INTERFACE_DIR_NAME = "gpaw-interface"


class RelaxTaskDocumentForTest(BaseModel):
    dir_name: str
    errors: str
    properties: RelaxProperties | None


def _build_default_adhesion_inputs_for_test() -> AdhesionWorkflowInputs:
    return AdhesionWorkflowInputs(
        top_slab=deepcopy(TEST_LINBO3_SLAB),
        bottom_slab=deepcopy(TEST_LINBO3_SLAB),
        assembly_settings=InterfaceAssemblySettings(
            offsets=DEFAULT_INTERFACE_OFFSETS,
            gaps=DEFAULT_INTERFACE_GAPS,
            vacuum=DEFAULT_INTERFACE_VACUUM,
        ),
    )


@job(output_schema=RelaxTaskDocumentForTest)
def _mock_relax_task_doc_job(
    *, structure: PMG_Structure, energy: float, dir_name: str
) -> RelaxTaskDocumentForTest:
    return RelaxTaskDocumentForTest(
        dir_name=dir_name,
        errors="",
        properties=build_relax_properties_for_test(structure=structure, energy=energy),
    )


@job(output_schema=QeScfPpChargemolTaskDocument)
def _mock_qe_scf_and_charges_task_doc_job(
    *, structure: PMG_Structure, energy: float, dir_name: str
) -> QeScfPpChargemolTaskDocument:
    single_point_properties = SinglePointProperties(
        step=TEST_RELAX_STEP_INDEX,
        structure=deepcopy(structure),
        forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in structure],
        stress=TEST_RELAX_STRESS_VECTOR_ZERO,
        energy=energy,
    )
    return QeScfPpChargemolTaskDocument(
        dir_name=dir_name,
        errors="",
        properties=single_point_properties,
        structure=deepcopy(structure),
        pp_run_info=build_successful_process_run_info_for_test(
            command_arguments=["pp.x"]
        ),
        chargemol_run_info=build_successful_process_run_info_for_test(),
    )


@job(output_schema=GpawChargemolTaskDocument)
def _mock_gpaw_scf_and_charges_task_doc_job(
    *, structure: PMG_Structure, energy: float, dir_name: str
) -> GpawChargemolTaskDocument:
    properties = SinglePointProperties(
        step=TEST_RELAX_STEP_INDEX,
        structure=deepcopy(structure),
        forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in structure],
        stress=TEST_RELAX_STRESS_VECTOR_ZERO,
        energy=energy,
    )
    return GpawChargemolTaskDocument(
        dir_name=dir_name,
        errors="",
        properties=properties,
        structure=deepcopy(structure),
        chargemol_run_info=build_successful_process_run_info_for_test(),
    )


@pytest.mark.parametrize(
    (
        "build_relax_maker_settings",
        "build_single_point_maker_settings",
        "single_point_backend",
        "single_point_task_doc_builder",
        "expected_slab_dir_names",
        "expected_interface_dir_names",
    ),
    [
        (
            build_qe_relax_maker_settings_for_test,
            build_qe_single_point_maker_settings_for_test,
            SinglePointBackend.qe,
            _mock_qe_scf_and_charges_task_doc_job,
            [
                TEST_SLAB_TOP_SCF_AND_CHARGES_DIR_NAME,
                TEST_SLAB_BOTTOM_SCF_AND_CHARGES_DIR_NAME,
            ],
            [TEST_INTERFACE_SCF_AND_CHARGES_DIR_NAME],
        ),
        (
            build_mlip_relax_maker_settings_for_test,
            build_qe_single_point_maker_settings_for_test,
            SinglePointBackend.qe,
            _mock_qe_scf_and_charges_task_doc_job,
            [
                TEST_SLAB_TOP_SCF_AND_CHARGES_DIR_NAME,
                TEST_SLAB_BOTTOM_SCF_AND_CHARGES_DIR_NAME,
            ],
            [TEST_INTERFACE_SCF_AND_CHARGES_DIR_NAME],
        ),
        (
            build_gpaw_relax_maker_settings_for_test,
            build_qe_single_point_maker_settings_for_test,
            SinglePointBackend.qe,
            _mock_qe_scf_and_charges_task_doc_job,
            [
                TEST_SLAB_TOP_SCF_AND_CHARGES_DIR_NAME,
                TEST_SLAB_BOTTOM_SCF_AND_CHARGES_DIR_NAME,
            ],
            [TEST_INTERFACE_SCF_AND_CHARGES_DIR_NAME],
        ),
        (
            build_qe_relax_maker_settings_for_test,
            build_gpaw_single_point_maker_settings_for_test,
            SinglePointBackend.gpaw,
            _mock_gpaw_scf_and_charges_task_doc_job,
            [
                TEST_GPAW_SCF_AND_CHARGES_TOP_DIR_NAME,
                TEST_GPAW_SCF_AND_CHARGES_BOTTOM_DIR_NAME,
            ],
            [TEST_GPAW_SCF_AND_CHARGES_INTERFACE_DIR_NAME],
        ),
    ],
    ids=[
        "qe-relax_qe-single-point",
        "mlip-relax_qe-single-point",
        "gpaw-relax_qe-single-point",
        "qe-relax_gpaw-single-point",
    ],
)
@patch("crystal.calculators.gpaw.scf_chargemol_maker.GpawScfChargemolMaker.make")
@patch("crystal.calculators.qe.scf_pp_chargemol_maker.QeScfPpChargemolMaker.make")
@patch(
    "crystal.flows.dft_adhesion_workflow_maker.InterfaceAssembler.generate_interfaces"
)
@patch("crystal.flows.dft_adhesion_workflow_maker.build_relax_maker")
def test_adhesion_workflow_runs_for_supported_backend_combinations(
    mock_build_relax_maker: MagicMock,
    mock_generate_interfaces: MagicMock,
    mock_qe_scf_and_charges_make: MagicMock,
    mock_gpaw_scf_and_charges_make: MagicMock,
    build_relax_maker_settings: Callable[[], RelaxMakerSettings],
    build_single_point_maker_settings: Callable[
        [], QeScfPpChargemolMakerSettings | GpawScfChargemolMakerSettings
    ],
    single_point_backend: SinglePointBackend,
    single_point_task_doc_builder: Callable[
        ..., QeScfPpChargemolTaskDocument | GpawChargemolTaskDocument
    ],
    expected_slab_dir_names: list[str],
    expected_interface_dir_names: list[str],
    tmp_path: Path,
) -> None:
    relax_maker_settings = build_relax_maker_settings()
    remaining_relax_job_sequences = [
        [
            _mock_relax_task_doc_job(
                structure=deepcopy(TEST_LINBO3_SLAB),
                energy=TEST_RELAX_ENERGY_LOW,
                dir_name=TEST_SLAB_TOP_RELAX_DIR_NAME,
            ),
            _mock_relax_task_doc_job(
                structure=deepcopy(TEST_LINBO3_SLAB),
                energy=TEST_RELAX_ENERGY_HIGH,
                dir_name=TEST_SLAB_BOTTOM_RELAX_DIR_NAME,
            ),
        ],
        [
            _mock_relax_task_doc_job(
                structure=deepcopy(TEST_LINBO3_SLAB),
                energy=TEST_INTERFACE_ENERGY_SINGLE,
                dir_name=TEST_INTERFACE_RELAX_DIR_NAME,
            )
        ],
    ]

    def _build_relax_maker_side_effect(**_kwargs: object) -> MagicMock:
        mock_relax_maker = MagicMock()
        mock_relax_maker.make.side_effect = remaining_relax_job_sequences.pop(0)
        return mock_relax_maker

    mock_build_relax_maker.side_effect = _build_relax_maker_side_effect
    mock_generate_interfaces.return_value = [deepcopy(TEST_LINBO3_SLAB)]
    single_point_jobs = [
        single_point_task_doc_builder(
            structure=deepcopy(TEST_LINBO3_SLAB),
            energy=TEST_RELAX_ENERGY_LOW,
            dir_name=expected_slab_dir_names[0],
        ),
        single_point_task_doc_builder(
            structure=deepcopy(TEST_LINBO3_SLAB),
            energy=TEST_RELAX_ENERGY_HIGH,
            dir_name=expected_slab_dir_names[1],
        ),
        single_point_task_doc_builder(
            structure=deepcopy(TEST_LINBO3_SLAB),
            energy=TEST_INTERFACE_ENERGY_SINGLE,
            dir_name=expected_interface_dir_names[0],
        ),
    ]
    if single_point_backend is SinglePointBackend.qe:
        mock_qe_scf_and_charges_make.side_effect = single_point_jobs
        mock_gpaw_scf_and_charges_make.side_effect = AssertionError(
            "GPAW single-point maker should not run for QE backend coverage"
        )
    else:
        mock_gpaw_scf_and_charges_make.side_effect = single_point_jobs
        mock_qe_scf_and_charges_make.side_effect = AssertionError(
            "QE single-point maker should not run for GPAW backend coverage"
        )

    adhesion_workflow_maker = AdhesionWorkflowMaker(
        simulation_settings=AdhesionSimulationSettings(
            slab_relax_maker_settings=relax_maker_settings,
            interface_relax_maker_settings=relax_maker_settings,
            single_point_maker_settings=build_single_point_maker_settings(),
        )
    )

    flow = adhesion_workflow_maker.make(_build_default_adhesion_inputs_for_test())
    run_flow = run_locally(flow, root_dir=tmp_path, create_folders=True)
    adhesion_results = extract_outputs_of_type_from_run(
        run_flow, AdhesionWorkflowResult
    )

    assert len(adhesion_results) == TEST_ADHESION_RECORD_COUNT_SINGLE
    assert len(adhesion_results[TEST_FIRST_INDEX].adhesion_energy_records) == (
        TEST_ADHESION_RECORD_COUNT_SINGLE
    )
    assert adhesion_results[TEST_FIRST_INDEX].slab_chargemol_dir_names == (
        expected_slab_dir_names
    )
    assert adhesion_results[TEST_FIRST_INDEX].interface_chargemol_dir_names == (
        expected_interface_dir_names
    )
