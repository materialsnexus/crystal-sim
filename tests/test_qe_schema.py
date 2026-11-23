"""Unit tests for QE DFT input models."""

from pathlib import Path
from unittest.mock import patch

import pytest

from crystal import CRYSTAL_CONFIG, CRYSTAL_ROOT
from crystal.constants.qe import (
    DEFAULT_CHARGE_DENSITY_CUTOFF_ENERGY,
    DEFAULT_CHARGE_DENSITY_CUTOFF_IN_RYDBERG,
    DEFAULT_CONV_THR_ENERGY,
    DEFAULT_CONV_THR_IN_RYDBERG,
    DEFAULT_WAVEFUNCTION_CUTOFF_ENERGY,
    DEFAULT_WAVEFUNCTION_CUTOFF_IN_RYDBERG,
    RYDBERG_TO_EV,
)
from crystal.schemas.process import ProcessRunInfo
from crystal.schemas.qe import (
    EsmBoundaryCondition,
    IsolatedSystemPolicy,
    KPoints,
    Occupations,
    QeConfig,
    QeRelaxTaskDocument,
    QeScfPpChargemolTaskDocument,
    QeScfSimulationSettings,
    QeSystemKeyword,
    ScfElectronsParameters,
    ScfSystemParameters,
    SerializableEspressoProfile,
    Smearing,
    SpinPolarizationPolicy,
    StartingPotential,
    StartingWavefunction,
)
from tests.helpers.constants.properties import (
    MOCK_RELAX_PROPERTIES,
    MOCK_SCF_PROPERTIES,
)
from tests.helpers.constants.qe import (
    CONV_THR_FIELD_NAME,
    CUTOFF_SERIALIZATION_TOLERANCE,
    DEFAULT_DISK_IO,
    DEGAUSS_FIELD_NAME,
)
from tests.helpers.constants.structure import (
    MOCK_ERROR_LOG_MESSAGE,
    MOCK_TASK_DIR_NAME,
    TEST_SI_STRUCTURE,
)
from tests.helpers.test_paths import TEST_SI_PSEUDOPOTENTIAL_FILENAME

NUMBER_OF_PROCESSES = (1, 4)
CUSTOM_WAVEFUNCTION_CUTOFF_ENERGY = DEFAULT_WAVEFUNCTION_CUTOFF_ENERGY + 25.0
CUSTOM_CHARGE_DENSITY_CUTOFF_ENERGY = DEFAULT_CHARGE_DENSITY_CUTOFF_ENERGY + 250.0
EXPECTED_WAVEFUNCTION_CUTOFF_ENERGY = (
    DEFAULT_WAVEFUNCTION_CUTOFF_IN_RYDBERG * RYDBERG_TO_EV
)
EXPECTED_CHARGE_DENSITY_CUTOFF_ENERGY = (
    DEFAULT_CHARGE_DENSITY_CUTOFF_IN_RYDBERG * RYDBERG_TO_EV
)
WAVEFUNCTION_CUTOFF_FIELD_NAME = "wavefunction_cutoff_energy"
CHARGE_DENSITY_CUTOFF_FIELD_NAME = "charge_density_cutoff_energy"
DEFAULT_MIXING_NDIM = 12
DEFAULT_STARTING_WAVEFUNCTION = "atomic"
DEFAULT_STARTING_POTENTIAL = "atomic"


def test_scf_system_sets_default_smearing_for_smearing_occupations() -> None:
    """Smearing defaults to mv when not provided and occupations=smearing."""

    system = ScfSystemParameters(occupations=Occupations.smearing, smearing=None)
    assert system.smearing is Smearing.mv


def test_scf_system_esm_requires_assume_isolated_esm() -> None:
    """Setting esm_bc without assume_isolated=esm should raise ValueError."""

    with pytest.raises(ValueError, match="assume_isolated"):
        ScfSystemParameters(
            assume_isolated=IsolatedSystemPolicy.none, esm_bc=EsmBoundaryCondition.bc1
        )


def test_scf_system_allows_valid_esm_combo() -> None:
    """No error when esm_bc is paired with assume_isolated=esm."""

    system = ScfSystemParameters(
        assume_isolated=IsolatedSystemPolicy.esm, esm_bc=EsmBoundaryCondition.bc1
    )
    assert system.esm_bc is EsmBoundaryCondition.bc1


def test_scf_system_explicit_qe_export_converts_cutoff_energies_to_rydberg() -> None:
    system = ScfSystemParameters(
        wavefunction_cutoff_energy=EXPECTED_WAVEFUNCTION_CUTOFF_ENERGY,
        charge_density_cutoff_energy=EXPECTED_CHARGE_DENSITY_CUTOFF_ENERGY,
    )
    serialized_system = system.model_dump(mode="json", exclude_none=True)
    assert serialized_system[WAVEFUNCTION_CUTOFF_FIELD_NAME] == pytest.approx(
        EXPECTED_WAVEFUNCTION_CUTOFF_ENERGY, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )
    assert serialized_system[CHARGE_DENSITY_CUTOFF_FIELD_NAME] == pytest.approx(
        EXPECTED_CHARGE_DENSITY_CUTOFF_ENERGY, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )

    qe_system = system.build_qe_system_namelist_with_cutoffs_in_rydberg()
    assert qe_system[QeSystemKeyword.wavefunction_cutoff.value] == pytest.approx(
        DEFAULT_WAVEFUNCTION_CUTOFF_IN_RYDBERG, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )
    assert qe_system[QeSystemKeyword.charge_density_cutoff.value] == pytest.approx(
        DEFAULT_CHARGE_DENSITY_CUTOFF_IN_RYDBERG, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )
    assert WAVEFUNCTION_CUTOFF_FIELD_NAME not in qe_system
    assert CHARGE_DENSITY_CUTOFF_FIELD_NAME not in qe_system


def test_qe_input_data_explicit_export_uses_qe_system_cutoff_units() -> None:
    config = QeConfig()
    qe_input_data = (
        config.input_data.build_qe_input_data_namelists_with_cutoffs_in_rydberg()
    )
    qe_system = qe_input_data["system"]
    assert qe_system[QeSystemKeyword.wavefunction_cutoff.value] == pytest.approx(
        DEFAULT_WAVEFUNCTION_CUTOFF_IN_RYDBERG, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )
    assert qe_system[QeSystemKeyword.charge_density_cutoff.value] == pytest.approx(
        DEFAULT_CHARGE_DENSITY_CUTOFF_IN_RYDBERG, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )


def test_scf_system_export_converts_degauss_from_ev_to_rydberg() -> None:
    custom_degauss_in_ev = 0.35
    system = ScfSystemParameters(degauss=custom_degauss_in_ev)

    serialized_system = system.model_dump(mode="json", exclude_none=True)
    assert serialized_system[DEGAUSS_FIELD_NAME] == pytest.approx(
        custom_degauss_in_ev, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )

    qe_system = system.build_qe_system_namelist_with_cutoffs_in_rydberg()
    assert qe_system[DEGAUSS_FIELD_NAME] == pytest.approx(
        custom_degauss_in_ev / RYDBERG_TO_EV, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )


def test_scf_system_default_degauss_is_ev_but_exports_to_legacy_rydberg_default() -> (
    None
):
    system = ScfSystemParameters()
    serialized_system = system.model_dump(mode="json", exclude_none=True)
    assert serialized_system[DEGAUSS_FIELD_NAME] == pytest.approx(
        0.02 * RYDBERG_TO_EV, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )

    qe_system = system.build_qe_system_namelist_with_cutoffs_in_rydberg()
    assert qe_system[DEGAUSS_FIELD_NAME] == pytest.approx(
        0.02, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )


def test_scf_electrons_export_converts_conv_thr_from_ev_to_rydberg() -> None:
    custom_conv_thr_in_ev = 2.8e-4
    electrons = ScfElectronsParameters(conv_thr_energy=custom_conv_thr_in_ev)
    qe_electrons = electrons.build_qe_electrons_namelist_in_rydberg()
    assert qe_electrons[CONV_THR_FIELD_NAME] == pytest.approx(
        custom_conv_thr_in_ev / RYDBERG_TO_EV, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )


def test_scf_electrons_default_conv_thr_is_ev_but_exports_to_legacy_rydberg_default() -> (
    None
):
    electrons = ScfElectronsParameters()
    assert electrons.conv_thr_energy == pytest.approx(
        DEFAULT_CONV_THR_ENERGY, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )
    qe_electrons = electrons.build_qe_electrons_namelist_in_rydberg()
    assert qe_electrons[CONV_THR_FIELD_NAME] == pytest.approx(
        DEFAULT_CONV_THR_IN_RYDBERG, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )


def test_scf_electrons_default_starting_values_and_mixing_ndim() -> None:
    electrons = ScfElectronsParameters()
    assert electrons.mixing_ndim == DEFAULT_MIXING_NDIM
    assert electrons.startingwfc is StartingWavefunction.atomic
    assert electrons.startingpot is StartingPotential.atomic


def test_qe_input_data_export_includes_disk_io_starting_values_and_mixing_ndim() -> (
    None
):
    qe_input_data = (
        QeConfig().input_data.build_qe_input_data_namelists_with_cutoffs_in_rydberg()
    )
    assert qe_input_data["control"]["disk_io"] == DEFAULT_DISK_IO
    assert qe_input_data["electrons"]["mixing_ndim"] == DEFAULT_MIXING_NDIM
    assert qe_input_data["electrons"]["startingwfc"] == DEFAULT_STARTING_WAVEFUNCTION
    assert qe_input_data["electrons"]["startingpot"] == DEFAULT_STARTING_POTENTIAL


@pytest.mark.parametrize("startingwfc", ["atomic", "atomic+random", "random", "file"])
def test_scf_electrons_accepts_all_supported_startingwfc_values(
    startingwfc: str,
) -> None:
    electrons = ScfElectronsParameters(startingwfc=startingwfc)
    qe_electrons = electrons.build_qe_electrons_namelist_in_rydberg()
    assert qe_electrons["startingwfc"] == startingwfc


@pytest.mark.parametrize("startingpot", ["atomic", "file"])
def test_scf_electrons_accepts_all_supported_startingpot_values(
    startingpot: str,
) -> None:
    electrons = ScfElectronsParameters(startingpot=startingpot)
    qe_electrons = electrons.build_qe_electrons_namelist_in_rydberg()
    assert qe_electrons["startingpot"] == startingpot


@pytest.mark.parametrize(
    "disk_io", ["high", "medium", "low", "nowf", "minimal", "none"]
)
def test_scf_control_accepts_all_supported_disk_io_values(disk_io: str) -> None:
    control_payload = QeConfig().get_modified_instance(
        overrides={"input_data": {"control": {"disk_io": disk_io}}}
    )
    qe_input_data = control_payload.input_data.build_qe_input_data_namelists_with_cutoffs_in_rydberg()
    assert qe_input_data["control"]["disk_io"] == disk_io


def test_scf_system_missing_cutoff_fields_use_defaults() -> None:
    wavefunction_default_system = ScfSystemParameters.model_validate(
        {"charge_density_cutoff_energy": CUSTOM_CHARGE_DENSITY_CUTOFF_ENERGY}
    )
    assert (
        wavefunction_default_system.wavefunction_cutoff_energy
        == DEFAULT_WAVEFUNCTION_CUTOFF_ENERGY
    )
    assert (
        wavefunction_default_system.charge_density_cutoff_energy
        == CUSTOM_CHARGE_DENSITY_CUTOFF_ENERGY
    )

    charge_density_default_system = ScfSystemParameters.model_validate(
        {"wavefunction_cutoff_energy": CUSTOM_WAVEFUNCTION_CUTOFF_ENERGY}
    )
    assert (
        charge_density_default_system.wavefunction_cutoff_energy
        == CUSTOM_WAVEFUNCTION_CUTOFF_ENERGY
    )
    assert (
        charge_density_default_system.charge_density_cutoff_energy
        == DEFAULT_CHARGE_DENSITY_CUTOFF_ENERGY
    )


def test_kpoints_requires_one_mode() -> None:
    """KPoints enforces exclusive spacing or mesh selection."""

    with pytest.raises(ValueError, match="Must specify either"):
        KPoints(spacing_in_inverse_angstrom=None, mesh=None)

    with pytest.raises(ValueError, match="Cannot specify both"):
        KPoints(spacing_in_inverse_angstrom=0.2, mesh=(2, 2, 2))


def test_kpoints_accepts_spacing_or_mesh() -> None:
    """Valid spacing or mesh definitions should pass validation."""

    spacing_kpoints = KPoints(spacing_in_inverse_angstrom=0.15)
    assert spacing_kpoints.spacing_in_inverse_angstrom == 0.15
    mesh_kpoints = KPoints(mesh=(4, 4, 1))
    assert mesh_kpoints.mesh == (4, 4, 1)


def test_chargemol_task_document_is_success() -> None:
    run_info = ProcessRunInfo(command_arguments=["exec"], exit_code=0)
    success_doc = QeScfPpChargemolTaskDocument(
        dir_name=MOCK_TASK_DIR_NAME,
        errors="",
        properties=MOCK_SCF_PROPERTIES,
        structure=TEST_SI_STRUCTURE,
        pp_run_info=run_info,
        chargemol_run_info=run_info,
    )
    assert success_doc.is_success

    failure_doc = QeScfPpChargemolTaskDocument(
        dir_name=MOCK_TASK_DIR_NAME,
        errors=MOCK_ERROR_LOG_MESSAGE,
        properties=None,
        structure=None,
        pp_run_info=None,
        chargemol_run_info=None,
    )
    assert not failure_doc.is_success


def test_relax_task_document_accepts_relax_properties() -> None:
    relax_doc = QeRelaxTaskDocument(
        dir_name=MOCK_TASK_DIR_NAME, errors="", properties=MOCK_RELAX_PROPERTIES
    )
    assert relax_doc.properties == MOCK_RELAX_PROPERTIES


def test_relax_task_document_rejects_scf_properties() -> None:
    with pytest.raises(ValueError, match="properties"):
        _ = QeRelaxTaskDocument(
            dir_name=MOCK_TASK_DIR_NAME, errors="", properties=MOCK_SCF_PROPERTIES
        )


def test_spin_policy_maps_to_expected_nspin_values() -> None:
    """Spin polarization policy should expose QE nspin integers via property."""

    default_system = ScfSystemParameters()
    assert (
        default_system.spin_polarization_policy is SpinPolarizationPolicy.non_polarized
    )
    assert int(default_system.spin_polarization_policy) == 1

    polarized_system = ScfSystemParameters(
        spin_polarization_policy=SpinPolarizationPolicy.collinear
    )
    assert int(polarized_system.spin_polarization_policy) == 2


@pytest.mark.parametrize("number_of_processes", NUMBER_OF_PROCESSES)
def test_serializable_esspresso_profile_uses_multiple_processes_in_command(
    number_of_processes: int,
) -> None:
    espresso_profile = (
        SerializableEspressoProfile.from_number_of_processes_and_pseudo_dir(
            number_of_processes=number_of_processes, pseudopotential_dir=Path.cwd()
        )
    )
    assert str(number_of_processes) in espresso_profile.command


def test_serialization_of_serializable_esspresso_profile() -> None:
    espresso_profile = (
        SerializableEspressoProfile.from_number_of_processes_and_pseudo_dir(
            number_of_processes=1, pseudopotential_dir=Path()
        )
    )
    assert all(k in espresso_profile.as_dict() for k in ["command", "pseudo_dir"])


def test_qe_scf_simulation_settings() -> None:
    simulation_settings = QeScfSimulationSettings(
        pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME},
        config=QeConfig(),
        set_pseudopotential_directory=CRYSTAL_ROOT,
    )
    assert (
        simulation_settings.pseudopotential_directory.as_posix()
        == CRYSTAL_ROOT.as_posix()
    )

    with patch.object(CRYSTAL_CONFIG.programs, "pseudopotentials_dir", None):
        simulation_settings = QeScfSimulationSettings(
            pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME}, config=QeConfig()
        )
        with pytest.raises(TypeError, match="No Pseudopotential directory provided"):
            _ = simulation_settings.pseudopotential_directory
