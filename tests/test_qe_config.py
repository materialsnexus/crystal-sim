"""Tests for QE config loading utilities."""

import pytest

from crystal.constants.qe import (
    DEFAULT_CHARGE_DENSITY_CUTOFF_ENERGY,
    DEFAULT_WAVEFUNCTION_CUTOFF_ENERGY,
)
from crystal.schemas.qe import DiskIO, QeConfig, Smearing

CUSTOM_CHARGE_DENSITY_CUTOFF_ENERGY = DEFAULT_CHARGE_DENSITY_CUTOFF_ENERGY + 250.0
DEFAULT_MIXING_NDIM = 12
DEFAULT_STARTING_WAVEFUNCTION = "atomic"
DEFAULT_STARTING_POTENTIAL = "atomic"


def test_config_uses_base_defaults_with_required_cutoffs() -> None:
    config = QeConfig()
    assert config.kpoints.spacing_in_inverse_angstrom == 0.25
    assert config.input_data.system.smearing is Smearing.mv
    assert config.input_data.control.disk_io is DiskIO.nowf
    assert config.input_data.electrons.mixing_ndim == DEFAULT_MIXING_NDIM
    assert (
        config.input_data.electrons.startingwfc.value == DEFAULT_STARTING_WAVEFUNCTION
    )
    assert config.input_data.electrons.startingpot.value == DEFAULT_STARTING_POTENTIAL
    assert (
        config.input_data.system.wavefunction_cutoff_energy
        == DEFAULT_WAVEFUNCTION_CUTOFF_ENERGY
    )
    assert (
        config.input_data.system.charge_density_cutoff_energy
        == DEFAULT_CHARGE_DENSITY_CUTOFF_ENERGY
    )


def test_config_partial_cutoff_settings_use_defaults() -> None:
    config = QeConfig.from_dict(
        config_dict={
            "input_data": {
                "system": {
                    "charge_density_cutoff_energy": CUSTOM_CHARGE_DENSITY_CUTOFF_ENERGY
                }
            },
            "kpoints": {"spacing_in_inverse_angstrom": 0.25},
        }
    )
    assert (
        config.input_data.system.wavefunction_cutoff_energy
        == DEFAULT_WAVEFUNCTION_CUTOFF_ENERGY
    )
    assert (
        config.input_data.system.charge_density_cutoff_energy
        == CUSTOM_CHARGE_DENSITY_CUTOFF_ENERGY
    )


def test_config_requires_kpoints() -> None:
    """If neither base nor user config define kpoints, raise ValueError."""

    with pytest.raises(ValueError, match="kpoints"):
        QeConfig.from_dict(config_dict={"input_data": {}})


def test_loader_validates_nested_models() -> None:
    """Invalid SCFSystem combinations (esm without assume_isolated) raise even if
    only override dictionary is illegal."""

    overrides = {
        "input_data": {"system": {"assume_isolated": "none", "esm_bc": "bc1"}},
        "kpoints": {"spacing_in_inverse_angstrom": 0.2},
    }
    config = QeConfig()
    with pytest.raises(ValueError, match="assume_isolated"):
        config.get_modified_instance(overrides=overrides)
