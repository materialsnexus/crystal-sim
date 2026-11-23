"""Tests for mapping QE configs into ASE Espresso kwargs."""

import pytest

from crystal.calculators.qe.espresso_arguments_builder import EspressoArgumentsBuilder
from crystal.constants.qe import (
    DEFAULT_CHARGE_DENSITY_CUTOFF_IN_RYDBERG,
    DEFAULT_WAVEFUNCTION_CUTOFF_IN_RYDBERG,
    RYDBERG_TO_EV,
)
from crystal.schemas.qe import QeConfig, QeSystemKeyword
from tests.helpers.constants.qe import (
    CUTOFF_SERIALIZATION_TOLERANCE,
    DEGAUSS_FIELD_NAME,
    TEST_ESPRESSO_PROFILE,
)
from tests.helpers.test_paths import (
    TEST_O_PBE_USP_FILENAME,
    TEST_SI_PSEUDOPOTENTIAL_FILENAME,
)

EXPECTED_WAVEFUNCTION_CUTOFF_ENERGY = (
    DEFAULT_WAVEFUNCTION_CUTOFF_IN_RYDBERG * RYDBERG_TO_EV
)
EXPECTED_CHARGE_DENSITY_CUTOFF_ENERGY = (
    DEFAULT_CHARGE_DENSITY_CUTOFF_IN_RYDBERG * RYDBERG_TO_EV
)
WAVEFUNCTION_CUTOFF_FIELD_NAME = "wavefunction_cutoff_energy"
CHARGE_DENSITY_CUTOFF_FIELD_NAME = "charge_density_cutoff_energy"


def _config_dict_with_kpoints_only(
    kpoints_config: dict[
        str, float | tuple[int, int, int] | tuple[float, float, float]
    ],
) -> dict[str, object]:
    return {"kpoints": kpoints_config}


def test_builder_produces_spacing_kwargs() -> None:
    """Builder should emit ASE kwargs with kspacing when specified."""
    config = QeConfig.from_dict(
        config_dict=_config_dict_with_kpoints_only({"spacing_in_inverse_angstrom": 0.2})
    )
    builder = EspressoArgumentsBuilder(
        config=config,
        pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME},
        profile=TEST_ESPRESSO_PROFILE,
    )
    kwargs = builder.calculator_kwargs
    assert kwargs["input_data"]["control"]["calculation"] == "scf"
    assert kwargs["pseudopotentials"] == {"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME}
    assert kwargs["kspacing"] == 0.2
    assert "kpts" not in kwargs


def test_builder_produces_mesh_kwargs_with_offset() -> None:
    """Builder should convert KPoints mesh+offset to ASE format."""

    config = QeConfig.from_dict(
        config_dict=_config_dict_with_kpoints_only(
            {"mesh": (4, 4, 1), "offset": (0.5, 0.0, 0.5)}
        )
    )
    builder = EspressoArgumentsBuilder(
        config=config,
        pseudopotentials={
            "Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME,
            "O": TEST_O_PBE_USP_FILENAME,
        },
        profile=TEST_ESPRESSO_PROFILE,
    )
    kwargs = builder.calculator_kwargs
    assert kwargs["kpts"] == (4, 4, 1)
    assert kwargs["koffset"] == (1, 0, 1)


def test_builder_validates_pseudopotentials_cover_all_elements() -> None:
    """Missing pseudopotentials for elements should raise ValueError."""
    config = QeConfig.from_dict(
        config_dict=_config_dict_with_kpoints_only({"spacing_in_inverse_angstrom": 0.2})
    )
    builder = EspressoArgumentsBuilder(
        config=config,
        pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME},
        profile=TEST_ESPRESSO_PROFILE,
    )
    with pytest.raises(ValueError, match="Missing pseudopotentials for"):
        builder.validate_pseudopotentials({"Si", "O"})


def test_builder_serializes_cutoff_energies_in_rydberg() -> None:
    config = QeConfig.from_dict(
        config_dict=_config_dict_with_kpoints_only({"spacing_in_inverse_angstrom": 0.2})
    )
    builder = EspressoArgumentsBuilder(
        config=config,
        pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME},
        profile=TEST_ESPRESSO_PROFILE,
    )
    kwargs = builder.calculator_kwargs
    system_payload = kwargs["input_data"]["system"]
    assert system_payload[QeSystemKeyword.wavefunction_cutoff.value] == pytest.approx(
        DEFAULT_WAVEFUNCTION_CUTOFF_IN_RYDBERG, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )
    assert system_payload[QeSystemKeyword.charge_density_cutoff.value] == pytest.approx(
        DEFAULT_CHARGE_DENSITY_CUTOFF_IN_RYDBERG, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )
    assert WAVEFUNCTION_CUTOFF_FIELD_NAME not in system_payload
    assert CHARGE_DENSITY_CUTOFF_FIELD_NAME not in system_payload


def test_builder_serializes_degauss_from_ev_to_rydberg() -> None:
    custom_degauss_in_ev = 0.41
    config = QeConfig.from_dict(
        config_dict={
            "input_data": {"system": {"degauss": custom_degauss_in_ev}},
            "kpoints": {"spacing_in_inverse_angstrom": 0.2},
        }
    )
    builder = EspressoArgumentsBuilder(
        config=config,
        pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME},
        profile=TEST_ESPRESSO_PROFILE,
    )
    kwargs = builder.calculator_kwargs
    system_payload = kwargs["input_data"]["system"]
    assert system_payload[DEGAUSS_FIELD_NAME] == pytest.approx(
        custom_degauss_in_ev / RYDBERG_TO_EV, abs=CUTOFF_SERIALIZATION_TOLERANCE
    )
