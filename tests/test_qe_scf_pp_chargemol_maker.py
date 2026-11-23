from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fireworks.core.firework import Firework
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow
from jobflow.managers.local import run_locally

from crystal import CRYSTAL_CONFIG
from crystal.calculators.qe.qe_pp_runner import QePpChargeDensityRunner
from crystal.calculators.qe.relax_maker import QeRelaxMaker
from crystal.calculators.qe.scf_pp_chargemol_maker import QeScfPpChargemolMaker
from crystal.constants.chargemol import CHARGEMOL_JOB_CONTROL_FILENAME
from crystal.constants.fireworks import (
    QUEUEADAPTER_MEM_KEY,
    QUEUEADAPTER_NTASKS_KEY,
    QUEUEADAPTER_SPEC_KEY,
    QUEUEADAPTER_WALLTIME_KEY,
)
from crystal.constants.qe_pp import QE_PP_CHARGE_DENSITY_BASENAME
from crystal.helpers.chargemol.charge_parser import (
    CHARGEMOL_NET_ATOMIC_CHARGES_PROPERTY,
    ChargemolNetAtomicChargeRecord,
    ChargemolNetAtomicCharges,
)
from crystal.helpers.chargemol.chargemol_runner import ChargemolRunner
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.process import ProcessRunInfo
from crystal.schemas.properties import SinglePointProperties
from crystal.schemas.pseudopotential import PseudopotentialConfig
from crystal.schemas.qe import (
    QeConfig,
    QeRelaxSimulationSettings,
    QeScfPpChargemolTaskDocument,
    QeScfSimulationSettings,
)
from crystal.schemas.qe_pp import (
    QePpInputSettings,
    QePpPlotSettings,
    QeScfPpChargemolSimulationSettings,
)
from crystal.schemas.relax_settings import OptimizerType
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.constants.process import (
    TEST_PROCESS_GENERIC_FAILURE_EXIT_CODE,
    TEST_PROCESS_SUCCESS_EXIT_CODE,
)
from tests.helpers.constants.qe_pp import QE_PP_DEFAULT_RUN_PREFIX, QE_PP_EXECUTABLE
from tests.helpers.constants.qe_scf_pp_chargemol import (
    CHARGEMOL_EXECUTABLE,
    TEST_FIRST_ATOM_INDEX,
    TEST_FIRST_CHARGE_INDEX,
    TEST_JOB_OUTPUT_INDEX,
    TEST_NET_ATOMIC_CHARGES,
    TEST_QE_SCF_ENERGY,
    TEST_QE_SCF_FORCE,
    TEST_QE_SCF_STEP,
    TEST_QE_SCF_STRESS,
    TEST_SECOND_ATOM_INDEX,
    TEST_SECOND_CHARGE_INDEX,
    TEST_SI_PSEUDOPOTENTIAL_FILENAME,
    TEST_SI_SYMBOL,
)
from tests.helpers.constants.queueadapter import (
    QUEUEADAPTER_EXPECTED_MEM_GB,
    QUEUEADAPTER_EXPECTED_WALLTIME,
    QUEUEADAPTER_TEST_MEM_GB,
    QUEUEADAPTER_TEST_WALLTIME_MIN,
)
from tests.helpers.constants.structure import MOCK_ERROR_MESSAGE, TEST_SI_STRUCTURE
from tests.helpers.test_chargemol_factories import (
    build_default_qe_pp_to_chargemol_workflow_inputs,
    build_test_pseudopotential_config,
)

PP_STDOUT_FILENAME = f"{QE_PP_CHARGE_DENSITY_BASENAME}.out"
CHARGEMOL_JOB_CONTROL_BASENAME = Path(CHARGEMOL_JOB_CONTROL_FILENAME).name
PP_RUN_INFO = ProcessRunInfo(
    command_arguments=[QE_PP_EXECUTABLE],
    exit_code=TEST_PROCESS_SUCCESS_EXIT_CODE,
    standard_output_file=Path(PP_STDOUT_FILENAME),
    standard_error_file=None,
)
CHARGEMOL_RUN_INFO = ProcessRunInfo(
    command_arguments=[CHARGEMOL_EXECUTABLE, CHARGEMOL_JOB_CONTROL_BASENAME],
    exit_code=TEST_PROCESS_SUCCESS_EXIT_CODE,
    standard_output_file=None,
    standard_error_file=None,
)


@pytest.fixture
def pseudopotential_config() -> PseudopotentialConfig:
    return build_test_pseudopotential_config(
        Path(__file__),
        pseudopotentials_file_map={TEST_SI_SYMBOL: TEST_SI_PSEUDOPOTENTIAL_FILENAME},
    )


@pytest.fixture
def simulation_settings(
    pseudopotential_config: PseudopotentialConfig,
) -> QeScfPpChargemolSimulationSettings:
    scf_settings = QeScfSimulationSettings(
        pseudopotentials={TEST_SI_SYMBOL: TEST_SI_PSEUDOPOTENTIAL_FILENAME},
        config=QeConfig(),
        set_pseudopotential_directory=pseudopotential_config.pseudopotential_dir,
    )
    return QeScfPpChargemolSimulationSettings(
        qe_scf_simulation_settings=scf_settings,
        qe_pp_input_settings=None,
        qe_pp_plot_settings=None,
    )


@pytest.fixture
def mock_scf_output() -> SinglePointProperties:
    return SinglePointProperties(
        step=TEST_QE_SCF_STEP,
        structure=deepcopy(TEST_SI_STRUCTURE),
        forces=TEST_QE_SCF_FORCE,
        stress=TEST_QE_SCF_STRESS,
        energy=TEST_QE_SCF_ENERGY,
    )


@pytest.fixture
def mock_net_atomic_charges() -> ChargemolNetAtomicCharges:
    records = [
        ChargemolNetAtomicChargeRecord(
            atom_index=TEST_FIRST_ATOM_INDEX,
            atom_symbol=TEST_SI_SYMBOL,
            net_atomic_charge=TEST_NET_ATOMIC_CHARGES[TEST_FIRST_CHARGE_INDEX],
        ),
        ChargemolNetAtomicChargeRecord(
            atom_index=TEST_SECOND_ATOM_INDEX,
            atom_symbol=TEST_SI_SYMBOL,
            net_atomic_charge=TEST_NET_ATOMIC_CHARGES[TEST_SECOND_CHARGE_INDEX],
        ),
    ]
    return ChargemolNetAtomicCharges(charge_records=records)


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.ChargemolNetAtomicChargesParser")
@patch.object(ChargemolRunner, "run", autospec=True)
@patch.object(QePpChargeDensityRunner, "run", autospec=True)
@patch("crystal.calculators.qe.scf_pp_chargemol_maker.run_qe_scf")
@run_in_temp_dir
def test_maker_defaults_prefix_and_outdir(
    mock_run_qe_scf: MagicMock,
    mock_run_qe_pp_charge_density: MagicMock,
    mock_run_chargemol: MagicMock,
    mock_parser_class: MagicMock,
    simulation_settings: QeScfPpChargemolSimulationSettings,
    mock_scf_output: SinglePointProperties,
    mock_net_atomic_charges: ChargemolNetAtomicCharges,
) -> None:
    mock_run_qe_scf.return_value = mock_scf_output
    mock_run_qe_pp_charge_density.return_value = PP_RUN_INFO
    mock_run_chargemol.return_value = CHARGEMOL_RUN_INFO
    parser_instance = MagicMock()
    parser_instance.get_chargemol_atomic_charges.return_value = mock_net_atomic_charges
    mock_parser_class.return_value = parser_instance

    executor_settings = ExecutorSettings(
        number_of_processes=2,
        total_memory_in_gb=QUEUEADAPTER_TEST_MEM_GB,
        maximum_walltime_in_minutes=QUEUEADAPTER_TEST_WALLTIME_MIN,
    )
    maker = QeScfPpChargemolMaker(
        simulation_settings=simulation_settings, executor_settings=executor_settings
    )
    job = maker.make(structure=TEST_SI_STRUCTURE)
    assert job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_NTASKS_KEY] == 2
    assert (
        job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_MEM_KEY]
        == QUEUEADAPTER_EXPECTED_MEM_GB
    )
    assert (
        job.metadata[QUEUEADAPTER_SPEC_KEY][QUEUEADAPTER_WALLTIME_KEY]
        == QUEUEADAPTER_EXPECTED_WALLTIME
    )
    run_locally(job, root_dir=Path.cwd(), create_folders=True)
    assert mock_run_qe_scf.call_args.kwargs["number_of_processes"] == 2

    runner_instance = mock_run_qe_pp_charge_density.call_args.args[0]
    input_settings = runner_instance.input_settings
    run_directory = runner_instance.run_directory
    assert input_settings.prefix == QE_PP_DEFAULT_RUN_PREFIX
    assert input_settings.outdir == run_directory.as_posix()


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.ChargemolNetAtomicChargesParser")
@patch.object(ChargemolRunner, "run", autospec=True)
@patch.object(QePpChargeDensityRunner, "run", autospec=True)
@patch("crystal.calculators.qe.scf_pp_chargemol_maker.run_qe_scf")
@run_in_temp_dir
def test_maker_runs_pp_and_chargemol_in_same_directory(
    mock_run_qe_scf: MagicMock,
    mock_run_qe_pp_charge_density: MagicMock,
    mock_run_chargemol: MagicMock,
    mock_parser_class: MagicMock,
    simulation_settings: QeScfPpChargemolSimulationSettings,
    mock_scf_output: SinglePointProperties,
    mock_net_atomic_charges: ChargemolNetAtomicCharges,
) -> None:
    mock_run_qe_scf.return_value = mock_scf_output
    mock_run_qe_pp_charge_density.return_value = PP_RUN_INFO
    mock_run_chargemol.return_value = CHARGEMOL_RUN_INFO
    parser_instance = MagicMock()
    parser_instance.get_chargemol_atomic_charges.return_value = mock_net_atomic_charges
    mock_parser_class.return_value = parser_instance

    maker = QeScfPpChargemolMaker(simulation_settings=simulation_settings)
    job = maker.make(structure=TEST_SI_STRUCTURE)
    run_locally(job, root_dir=Path.cwd(), create_folders=True)

    pp_run_directory = mock_run_qe_pp_charge_density.call_args.args[0].run_directory
    chargemol_run_directory = mock_run_chargemol.call_args.args[0].run_directory
    assert pp_run_directory == chargemol_run_directory


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.ChargemolNetAtomicChargesParser")
@patch.object(ChargemolRunner, "run", autospec=True)
@patch.object(QePpChargeDensityRunner, "run", autospec=True)
@patch("crystal.calculators.qe.scf_pp_chargemol_maker.run_qe_scf")
@run_in_temp_dir
def test_maker_attaches_net_atomic_charges_and_outputs_energy_forces_stress(
    mock_run_qe_scf: MagicMock,
    mock_run_qe_pp_charge_density: MagicMock,
    mock_run_chargemol: MagicMock,
    mock_parser_class: MagicMock,
    simulation_settings: QeScfPpChargemolSimulationSettings,
    mock_scf_output: SinglePointProperties,
    mock_net_atomic_charges: ChargemolNetAtomicCharges,
) -> None:
    mock_run_qe_scf.return_value = mock_scf_output
    mock_run_qe_pp_charge_density.return_value = PP_RUN_INFO
    mock_run_chargemol.return_value = CHARGEMOL_RUN_INFO
    parser_instance = MagicMock()
    parser_instance.get_chargemol_atomic_charges.return_value = mock_net_atomic_charges
    mock_parser_class.return_value = parser_instance

    maker = QeScfPpChargemolMaker(simulation_settings=simulation_settings)
    job = maker.make(structure=TEST_SI_STRUCTURE)
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)

    output_doc = run_job[job.uuid][TEST_JOB_OUTPUT_INDEX].output
    assert isinstance(output_doc, QeScfPpChargemolTaskDocument)
    assert output_doc.structure is not None
    assert (
        output_doc.structure.site_properties[CHARGEMOL_NET_ATOMIC_CHARGES_PROPERTY]
        == TEST_NET_ATOMIC_CHARGES
    )
    assert isinstance(output_doc.properties, SinglePointProperties)
    assert output_doc.properties.charges == TEST_NET_ATOMIC_CHARGES
    assert output_doc.properties.structure == output_doc.structure
    assert output_doc.properties.energy == pytest.approx(TEST_QE_SCF_ENERGY)
    assert output_doc.properties.forces == TEST_QE_SCF_FORCE
    assert output_doc.properties.stress == TEST_QE_SCF_STRESS


def test_qe_pp_runner_returns_error_info_on_subprocess_failure(tmp_path: Path) -> None:
    input_settings = QePpInputSettings(
        prefix=QE_PP_DEFAULT_RUN_PREFIX, outdir=tmp_path.as_posix()
    )
    plot_settings = QePpPlotSettings()
    runner = QePpChargeDensityRunner(
        run_directory=tmp_path,
        input_settings=input_settings,
        plot_settings=plot_settings,
    )
    with (
        patch.object(
            QePpChargeDensityRunner,
            "_pp_executable",
            new_callable=PropertyMock,
            return_value=Path(QE_PP_EXECUTABLE),
        ),
        patch(
            "crystal.calculators.qe.qe_pp_runner.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                TEST_PROCESS_GENERIC_FAILURE_EXIT_CODE, [QE_PP_EXECUTABLE]
            ),
        ),
    ):
        run_info = runner.run()
    assert isinstance(run_info, ProcessRunInfo)
    assert run_info.exit_code != 0
    assert run_info.error_message is not None


def test_qe_pp_runner_returns_process_run_info(tmp_path: Path) -> None:
    input_settings = QePpInputSettings(
        prefix=QE_PP_DEFAULT_RUN_PREFIX, outdir=tmp_path.as_posix()
    )
    plot_settings = QePpPlotSettings()
    runner = QePpChargeDensityRunner(
        run_directory=tmp_path,
        input_settings=input_settings,
        plot_settings=plot_settings,
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        [QE_PP_EXECUTABLE], TEST_PROCESS_SUCCESS_EXIT_CODE
    )
    with (
        patch.object(
            QePpChargeDensityRunner,
            "_pp_executable",
            new_callable=PropertyMock,
            return_value=Path(QE_PP_EXECUTABLE),
        ),
        patch(
            "crystal.calculators.qe.qe_pp_runner.subprocess.run",
            return_value=completed_process,
        ),
    ):
        run_info = runner.run()
    assert isinstance(run_info, ProcessRunInfo)
    assert run_info.command_arguments == [QE_PP_EXECUTABLE]
    assert run_info.exit_code == TEST_PROCESS_SUCCESS_EXIT_CODE
    assert run_info.standard_output_file == runner.run_files.output_path
    assert run_info.standard_error_file is None


def test_chargemol_runner_returns_error_info_on_subprocess_failure(
    tmp_path: Path, pseudopotential_config: PseudopotentialConfig
) -> None:
    workflow_inputs = build_default_qe_pp_to_chargemol_workflow_inputs(
        cube_output_directory=tmp_path, pseudopotential_config=pseudopotential_config
    )
    runner = ChargemolRunner(run_directory=tmp_path, workflow_inputs=workflow_inputs)
    with (
        patch(
            "crystal.helpers.chargemol.chargemol_runner.ChargemolJobControlWriter.write",
            return_value=tmp_path / CHARGEMOL_JOB_CONTROL_FILENAME,
        ),
        patch.object(
            ChargemolRunner,
            "executable",
            new_callable=PropertyMock,
            return_value=Path(CHARGEMOL_EXECUTABLE),
        ),
        patch(
            "crystal.helpers.chargemol.chargemol_runner.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                TEST_PROCESS_GENERIC_FAILURE_EXIT_CODE, [CHARGEMOL_EXECUTABLE]
            ),
        ),
    ):
        run_info = runner.run()
    assert isinstance(run_info, ProcessRunInfo)
    assert run_info.exit_code != 0
    assert run_info.error_message is not None


def test_chargemol_runner_returns_process_run_info(
    tmp_path: Path, pseudopotential_config: PseudopotentialConfig
) -> None:
    workflow_inputs = build_default_qe_pp_to_chargemol_workflow_inputs(
        cube_output_directory=tmp_path, pseudopotential_config=pseudopotential_config
    )
    runner = ChargemolRunner(run_directory=tmp_path, workflow_inputs=workflow_inputs)
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        [CHARGEMOL_EXECUTABLE], TEST_PROCESS_SUCCESS_EXIT_CODE
    )
    with (
        patch(
            "crystal.helpers.chargemol.chargemol_runner.ChargemolJobControlWriter.write",
            return_value=tmp_path / CHARGEMOL_JOB_CONTROL_FILENAME,
        ),
        patch.object(
            ChargemolRunner,
            "executable",
            new_callable=PropertyMock,
            return_value=Path(CHARGEMOL_EXECUTABLE),
        ),
        patch(
            "crystal.helpers.chargemol.chargemol_runner.subprocess.run",
            return_value=completed_process,
        ),
    ):
        run_info = runner.run()
    assert isinstance(run_info, ProcessRunInfo)
    assert run_info.command_arguments == [
        CHARGEMOL_EXECUTABLE,
        CHARGEMOL_JOB_CONTROL_BASENAME,
    ]
    assert run_info.exit_code == TEST_PROCESS_SUCCESS_EXIT_CODE
    assert run_info.standard_output_file is None
    assert run_info.standard_error_file is None


@patch.object(QePpChargeDensityRunner, "run", autospec=True)
@patch("crystal.calculators.qe.scf_pp_chargemol_maker.run_qe_scf")
@run_in_temp_dir
def test_maker_records_pp_run_error_in_error_log(
    mock_run_qe_scf: MagicMock,
    mock_run_qe_pp_charge_density: MagicMock,
    simulation_settings: QeScfPpChargemolSimulationSettings,
    mock_scf_output: SinglePointProperties,
) -> None:
    mock_run_qe_scf.return_value = mock_scf_output
    mock_run_qe_pp_charge_density.return_value = ProcessRunInfo(
        command_arguments=[QE_PP_EXECUTABLE],
        exit_code=TEST_PROCESS_GENERIC_FAILURE_EXIT_CODE,
        standard_output_file=None,
        standard_error_file=None,
        error_message=MOCK_ERROR_MESSAGE,
    )
    maker = QeScfPpChargemolMaker(simulation_settings=simulation_settings)
    job = maker.make(structure=TEST_SI_STRUCTURE)
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    output_doc = run_job[job.uuid][TEST_JOB_OUTPUT_INDEX].output
    assert output_doc.errors == MOCK_ERROR_MESSAGE
    assert output_doc.properties is None
    assert output_doc.structure is None


@patch.object(ChargemolRunner, "run", autospec=True)
@patch.object(QePpChargeDensityRunner, "run", autospec=True)
@patch("crystal.calculators.qe.scf_pp_chargemol_maker.run_qe_scf")
@run_in_temp_dir
def test_maker_records_chargemol_run_error_in_error_log(
    mock_run_qe_scf: MagicMock,
    mock_run_qe_pp_charge_density: MagicMock,
    mock_run_chargemol: MagicMock,
    simulation_settings: QeScfPpChargemolSimulationSettings,
    mock_scf_output: SinglePointProperties,
) -> None:
    mock_run_qe_scf.return_value = mock_scf_output
    mock_run_qe_pp_charge_density.return_value = PP_RUN_INFO
    mock_run_chargemol.return_value = ProcessRunInfo(
        command_arguments=[CHARGEMOL_EXECUTABLE, CHARGEMOL_JOB_CONTROL_BASENAME],
        exit_code=TEST_PROCESS_GENERIC_FAILURE_EXIT_CODE,
        standard_output_file=None,
        standard_error_file=None,
        error_message=MOCK_ERROR_MESSAGE,
    )
    maker = QeScfPpChargemolMaker(simulation_settings=simulation_settings)
    job = maker.make(structure=TEST_SI_STRUCTURE)
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    output_doc = run_job[job.uuid][TEST_JOB_OUTPUT_INDEX].output
    assert output_doc.errors == MOCK_ERROR_MESSAGE
    assert output_doc.properties is None
    assert output_doc.structure is None
    assert output_doc.pp_run_info is None
    assert output_doc.chargemol_run_info is None


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.run_qe_scf")
@run_in_temp_dir
def test_maker_records_qe_scf_error_in_error_log(
    mock_run_qe_scf: MagicMock, simulation_settings: QeScfPpChargemolSimulationSettings
) -> None:
    mock_run_qe_scf.side_effect = RuntimeError(MOCK_ERROR_MESSAGE)
    maker = QeScfPpChargemolMaker(simulation_settings=simulation_settings)
    job = maker.make(structure=TEST_SI_STRUCTURE)
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)
    output_doc = run_job[job.uuid][TEST_JOB_OUTPUT_INDEX].output
    assert output_doc.errors == MOCK_ERROR_MESSAGE
    assert output_doc.properties is None
    assert output_doc.structure is None
    assert output_doc.pp_run_info is None
    assert output_doc.chargemol_run_info is None


@patch("crystal.calculators.qe.scf_pp_chargemol_maker.ChargemolNetAtomicChargesParser")
@patch.object(ChargemolRunner, "run", autospec=True)
@patch.object(QePpChargeDensityRunner, "run", autospec=True)
@patch("crystal.calculators.qe.scf_pp_chargemol_maker.run_qe_scf")
@run_in_temp_dir
def test_maker_stores_pp_and_chargemol_run_info(
    mock_run_qe_scf: MagicMock,
    mock_run_qe_pp_charge_density: MagicMock,
    mock_run_chargemol: MagicMock,
    mock_parser_class: MagicMock,
    simulation_settings: QeScfPpChargemolSimulationSettings,
    mock_scf_output: SinglePointProperties,
    mock_net_atomic_charges: ChargemolNetAtomicCharges,
) -> None:
    mock_run_qe_scf.return_value = mock_scf_output
    mock_run_qe_pp_charge_density.return_value = PP_RUN_INFO
    mock_run_chargemol.return_value = CHARGEMOL_RUN_INFO
    parser_instance = MagicMock()
    parser_instance.get_chargemol_atomic_charges.return_value = mock_net_atomic_charges
    mock_parser_class.return_value = parser_instance

    maker = QeScfPpChargemolMaker(simulation_settings=simulation_settings)
    job = maker.make(structure=TEST_SI_STRUCTURE)
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)

    output_doc = run_job[job.uuid][TEST_JOB_OUTPUT_INDEX].output
    assert output_doc.pp_run_info == mock_run_qe_pp_charge_density.return_value
    assert output_doc.chargemol_run_info == mock_run_chargemol.return_value


def test_qe_relax_and_chargemol_flow_firework_roundtrip(
    pseudopotential_config: PseudopotentialConfig,
    simulation_settings: QeScfPpChargemolSimulationSettings,
) -> None:
    relax_simulation_settings = QeRelaxSimulationSettings(
        config=QeConfig(),
        pseudopotentials={TEST_SI_SYMBOL: TEST_SI_PSEUDOPOTENTIAL_FILENAME},
        set_pseudopotential_directory=pseudopotential_config.pseudopotential_dir,
        optimizer_type=OptimizerType.BFGS,
        max_steps=2,
        force_threshold_in_ev_per_angstrom=2,
    )
    relax_maker = QeRelaxMaker(simulation_settings=relax_simulation_settings)
    chargemol_maker = QeScfPpChargemolMaker(simulation_settings=simulation_settings)
    relax_job = relax_maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    chargemol_job = chargemol_maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    flow = Flow(jobs=[relax_job, chargemol_job], output=chargemol_job.output)
    workflow = flow_to_workflow(flow)

    for firework in workflow.id_fw.values():
        roundtrip_firework = Firework.from_dict(firework.to_dict())
        assert roundtrip_firework.name == firework.name


@pytest.mark.needs_third_party_executables
@run_in_temp_dir
def test_run_qe_scf_pp_chargemol_maker_locally(
    simulation_settings: QeScfPpChargemolSimulationSettings,
) -> None:
    assert CRYSTAL_CONFIG.programs.quantum_espresso_dir is not None
    assert CRYSTAL_CONFIG.programs.chargemol_dir is not None
    maker = QeScfPpChargemolMaker(simulation_settings=simulation_settings)
    job = maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    run_job = run_locally(job, root_dir=Path.cwd(), create_folders=True)

    output_doc = run_job[job.uuid][TEST_JOB_OUTPUT_INDEX].output
    assert isinstance(output_doc, QeScfPpChargemolTaskDocument)
    assert output_doc.converged
    assert output_doc.properties is not None
    assert output_doc.structure is not None
    assert output_doc.pp_run_info is not None
    assert output_doc.chargemol_run_info is not None
    charges = output_doc.structure.site_properties.get(
        CHARGEMOL_NET_ATOMIC_CHARGES_PROPERTY
    )
    assert charges is not None
    assert len(charges) == len(output_doc.structure)
