"""Tests for Chargemol DDEC6 job_control.txt inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from crystal.helpers.chargemol.core_electrons import ChargemolCoreElectronCountBuilder
from crystal.schemas.chargemol import (
    ChargemolChargeType,
    ChargemolJobControlInputBuilder,
    ChargemolJobControlSettings,
)
from crystal.schemas.pseudopotential import (
    Functional,
    PseudopotentialInfo,
    PseudoPotentialType,
)
from tests.helpers.test_chargemol_job_control_helpers import (
    build_expected_chargemol_job_control_lines,
)
from tests.helpers.test_paths import (
    TEST_O_PBE_USP_FILENAME,
    TEST_SI_PSEUDOPOTENTIAL_FILENAME,
)


@pytest.fixture
def chargemol_base_settings(
    chargemol_input_cube_path: Path,
    chargemol_atomic_densities_directory: Path,
    chargemol_core_electron_mapping_for_si_and_o: dict[int, int],
) -> ChargemolJobControlSettings:
    """Base job_control settings shared across Chargemol tests."""
    return ChargemolJobControlSettings(
        input_filename=chargemol_input_cube_path,
        atomic_densities_directory=chargemol_atomic_densities_directory,
        periodic_a=True,
        periodic_b=True,
        periodic_c=False,
        charge_type=ChargemolChargeType.ddec6,
        compute_bond_orders=True,
        print_atomic_densities=False,
        core_electrons_by_atomic_number=chargemol_core_electron_mapping_for_si_and_o,
    )


@pytest.fixture
def chargemol_expected_job_control_lines(
    chargemol_input_cube_path: Path,
    chargemol_atomic_densities_directory: Path,
    chargemol_core_electron_mapping_for_si_and_o: dict[int, int],
) -> list[str]:
    """Canonical job_control lines for Chargemol DDEC6 cube inputs."""
    return build_expected_chargemol_job_control_lines(
        input_cube_path=chargemol_input_cube_path,
        atomic_densities_directory=chargemol_atomic_densities_directory,
        core_electrons_by_atomic_number=chargemol_core_electron_mapping_for_si_and_o,
    )


@pytest.fixture
def chargemol_valid_pseudopotential_infos() -> list[PseudopotentialInfo]:
    """Valid pseudopotential metadata for core-electron mapping."""
    return [
        PseudopotentialInfo(
            element="O",
            functional=Functional.pbe,
            type=PseudoPotentialType.ultrasoft,
            filename=TEST_O_PBE_USP_FILENAME,
            z_valence=6.0,
        ),
        PseudopotentialInfo(
            element="Si",
            functional=Functional.pbe,
            type=PseudoPotentialType.ultrasoft,
            filename=TEST_SI_PSEUDOPOTENTIAL_FILENAME,
            z_valence=4.0,
        ),
    ]


@pytest.fixture
def chargemol_fractional_valence_pseudopotential_info() -> PseudopotentialInfo:
    """Pseudopotential metadata with a fractional z_valence for validation tests."""
    return PseudopotentialInfo(
        element="Si",
        functional=Functional.pbe,
        type=PseudoPotentialType.ultrasoft,
        filename=TEST_SI_PSEUDOPOTENTIAL_FILENAME,
        z_valence=4.2,
    )


@pytest.fixture
def chargemol_missing_valence_pseudopotential_info() -> PseudopotentialInfo:
    """Pseudopotential metadata missing z_valence for validation tests."""
    return PseudopotentialInfo(
        element="Si",
        functional=Functional.pbe,
        type=PseudoPotentialType.ultrasoft,
        filename=TEST_SI_PSEUDOPOTENTIAL_FILENAME,
        z_valence=None,
    )


def test_job_control_input_builder_renders_closing_tag_format(
    chargemol_base_settings: ChargemolJobControlSettings,
    chargemol_expected_job_control_lines: list[str],
) -> None:
    """job_control.txt should use opening and closing tags with Fortran booleans."""
    builder = ChargemolJobControlInputBuilder(settings=chargemol_base_settings)
    expected = "\n".join(chargemol_expected_job_control_lines)
    assert builder.job_control_text == expected


def test_job_control_input_builder_writes_neutral_net_charge_tag(
    chargemol_base_settings: ChargemolJobControlSettings,
) -> None:
    """job_control.txt should always encode neutral system charge."""
    builder = ChargemolJobControlInputBuilder(settings=chargemol_base_settings)
    assert "<net charge>\n0.0\n</net charge>" in builder.job_control_text


def test_core_electron_count_builder_uses_pseudopotential_metadata(
    chargemol_valid_pseudopotential_infos: list[PseudopotentialInfo],
) -> None:
    """Core electron counts should come from Z - z_valence."""
    builder = ChargemolCoreElectronCountBuilder(
        pseudopotential_infos=chargemol_valid_pseudopotential_infos
    )
    assert builder.core_electrons_by_atomic_number == {8: 2, 14: 10}


def test_core_electron_builder_rejects_fractional_valence(
    chargemol_fractional_valence_pseudopotential_info: PseudopotentialInfo,
) -> None:
    """z_valence must be integer-like to compute core electrons."""
    with pytest.raises(ValueError, match="z_valence"):
        _ = ChargemolCoreElectronCountBuilder(
            pseudopotential_infos=[chargemol_fractional_valence_pseudopotential_info]
        ).core_electrons_by_atomic_number


def test_core_electron_builder_requires_valence_metadata(
    chargemol_missing_valence_pseudopotential_info: PseudopotentialInfo,
) -> None:
    """Missing z_valence should raise an error."""
    with pytest.raises(ValueError, match="z_valence"):
        _ = ChargemolCoreElectronCountBuilder(
            pseudopotential_infos=[chargemol_missing_valence_pseudopotential_info]
        ).core_electrons_by_atomic_number
