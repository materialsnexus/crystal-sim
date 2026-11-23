"""Unit tests for GPAW schema models."""

import pytest
from pydantic import ValidationError

from crystal.constants.gpaw import DEFAULT_GPAW_FERMI_DIRAC_WIDTH
from crystal.schemas.gpaw import GpawCalculatorSettings, GpawMixerSettings, GpawSettings
from tests.helpers.constants.gpaw import (
    GPAW_BULK_OR_SLAB_SLAB_INTERFACE_VALUE,
    SLAB_INTERFACE_GPAW_DIPOLELAYER,
    SLAB_INTERFACE_GPAW_PBC,
)


def test_gpaw_settings_rejects_pbc_for_bulk() -> None:
    with pytest.raises(ValueError, match="bulk_or_slab"):
        _ = GpawSettings(pbc=SLAB_INTERFACE_GPAW_PBC)


def test_gpaw_settings_rejects_poissonsolver_for_bulk() -> None:
    with pytest.raises(ValueError, match="bulk_or_slab"):
        _ = GpawSettings(poissonsolver={"dipolelayer": SLAB_INTERFACE_GPAW_DIPOLELAYER})


def test_gpaw_settings_require_pbc_for_slab_interface() -> None:
    with pytest.raises(ValueError, match="pbc"):
        _ = GpawSettings(
            bulk_or_slab=GPAW_BULK_OR_SLAB_SLAB_INTERFACE_VALUE,
            poissonsolver={"dipolelayer": SLAB_INTERFACE_GPAW_DIPOLELAYER},
        )


def test_gpaw_settings_require_poissonsolver_for_slab_interface() -> None:
    with pytest.raises(ValueError, match="poissonsolver"):
        _ = GpawSettings(
            bulk_or_slab=GPAW_BULK_OR_SLAB_SLAB_INTERFACE_VALUE,
            pbc=SLAB_INTERFACE_GPAW_PBC,
        )


def test_gpaw_settings_require_slab_interface_specific_pbc() -> None:
    with pytest.raises(ValueError, match="pbc"):
        _ = GpawSettings(
            bulk_or_slab=GPAW_BULK_OR_SLAB_SLAB_INTERFACE_VALUE,
            pbc=(True, True, True),
            poissonsolver={"dipolelayer": SLAB_INTERFACE_GPAW_DIPOLELAYER},
        )


def test_gpaw_settings_require_xy_dipolelayer_for_slab_interface() -> None:
    with pytest.raises(ValueError, match="dipolelayer"):
        _ = GpawSettings(
            bulk_or_slab=GPAW_BULK_OR_SLAB_SLAB_INTERFACE_VALUE,
            pbc=SLAB_INTERFACE_GPAW_PBC,
            poissonsolver={"dipolelayer": "xz"},
        )


def test_gpaw_settings_accept_spin_polarisation_and_export_spinpol() -> None:
    settings = GpawSettings(spin_polarisation=True)
    kwargs = settings.build_gpaw_calculator_kwargs()
    assert kwargs["spinpol"] is True
    assert "spin_polarisation" not in kwargs


def test_gpaw_settings_export_parallel_sl_auto_key_from_explicit_name() -> None:
    settings = GpawSettings(
        parallel={"auto_scalapack_parallelization": False, "use_elpa": True}
    )
    kwargs = settings.build_gpaw_calculator_kwargs()
    assert kwargs["parallel"]["sl_auto"] is False
    assert kwargs["parallel"]["use_elpa"] is True
    assert "auto_scalapack_parallelization" not in kwargs["parallel"]


def test_gpaw_settings_default_occupation_name_is_fixed_uniform() -> None:
    settings = GpawSettings()
    assert settings.build_gpaw_calculator_kwargs()["occupations"] == {
        "name": "fixed-uniform"
    }


@pytest.mark.parametrize(
    ("occupations", "expected_width"),
    [
        ({"name": "fermi-dirac"}, DEFAULT_GPAW_FERMI_DIRAC_WIDTH),
        ({"name": "fermi-dirac", "width": 0.02}, 0.02),
        ({"name": "fermi-dirac", "width": 0.0}, 0.0),
    ],
)
def test_gpaw_fermi_dirac_exports_expected_width(
    occupations: dict[str, object], expected_width: float
) -> None:
    settings = GpawSettings(occupations=occupations)

    assert settings.build_gpaw_calculator_kwargs()["occupations"] == {
        "name": "fermi-dirac",
        "width": expected_width,
    }


def test_gpaw_fermi_dirac_rejects_negative_width() -> None:
    with pytest.raises(ValidationError):
        GpawSettings(occupations={"name": "fermi-dirac", "width": -0.01})


def test_gpaw_slab_settings_export_poisson_solver() -> None:
    settings = GpawSettings(
        bulk_or_slab=GPAW_BULK_OR_SLAB_SLAB_INTERFACE_VALUE,
        pbc=SLAB_INTERFACE_GPAW_PBC,
        poissonsolver={"dipolelayer": SLAB_INTERFACE_GPAW_DIPOLELAYER},
    )

    assert settings.build_gpaw_calculator_kwargs()["poissonsolver"] == {
        "dipolelayer": SLAB_INTERFACE_GPAW_DIPOLELAYER
    }


def test_gpaw_calculator_settings_export_general_calculator_options() -> None:
    settings = GpawCalculatorSettings(
        kpoint_size=(1, 1, 1),
        charge=0.0,
        enable_hund_rule_initialisation=True,
        symmetry="off",
        number_of_bands=3,
    )

    kwargs = settings.build_gpaw_calculator_kwargs()

    assert kwargs["kpts"] == {"size": (1, 1, 1), "gamma": True}
    assert kwargs["charge"] == 0.0
    assert kwargs["hund"] is True
    assert kwargs["symmetry"] == "off"
    assert kwargs["nbands"] == 3


def test_gpaw_calculator_settings_reject_zero_band_count() -> None:
    with pytest.raises(ValidationError):
        GpawCalculatorSettings(number_of_bands=0)


def test_gpaw_calculator_settings_accept_negative_band_adjustments() -> None:
    settings = GpawCalculatorSettings(number_of_bands=-3)

    assert settings.number_of_bands == -3


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value"),
    [("beta", 0.1), ("nmaxold", 3), ("weight", 10.0)],
)
def test_gpaw_no_mixing_rejects_explicit_mixing_parameters(
    parameter_name: str, parameter_value: float
) -> None:
    with pytest.raises(ValueError, match="no-mixing density mixer backend"):
        GpawMixerSettings(backend="no-mixing", **{parameter_name: parameter_value})


def test_gpaw_no_mixing_accepts_omitted_mixing_parameters() -> None:
    settings = GpawMixerSettings(backend="no-mixing")

    assert settings.beta is None
    assert settings.nmaxold is None
    assert settings.weight is None


def test_gpaw_mixer_backend_maps_to_gpaw_backend_keyword() -> None:
    settings = GpawCalculatorSettings(mixer={"backend": "no-mixing"})

    assert settings.build_gpaw_calculator_kwargs()["mixer"] == {
        "backend": "no-mixing",
        "method": "fullspin",
    }


def test_gpaw_mixer_omits_numeric_defaults() -> None:
    settings = GpawCalculatorSettings()

    assert settings.mixer.beta is None
    assert settings.mixer.nmaxold is None
    assert settings.mixer.weight is None
    assert settings.build_gpaw_calculator_kwargs()["mixer"] == {
        "backend": "msr1",
        "method": "fullspin",
    }


def test_gpaw_mixer_keeps_explicit_mixing_parameters() -> None:
    settings = GpawCalculatorSettings(
        mixer={
            "backend": "msr1",
            "method": "fullspin",
            "beta": 0.1,
            "nmaxold": 3,
            "weight": 10.0,
        }
    )

    assert settings.build_gpaw_calculator_kwargs()["mixer"] == {
        "backend": "msr1",
        "method": "fullspin",
        "beta": 0.1,
        "nmaxold": 3,
        "weight": 10.0,
    }


def test_gpaw_default_eigensolver_is_ppcg() -> None:
    settings = GpawCalculatorSettings()

    assert settings.build_gpaw_calculator_kwargs()["eigensolver"] == {"name": "ppcg"}


@pytest.mark.parametrize(
    ("kpoint_density", "kpoint_size"), [(None, None), (6.0, (1, 1, 1))]
)
def test_gpaw_calculator_settings_require_one_kpoint_sampling_mode(
    kpoint_density: float | None, kpoint_size: tuple[int, int, int] | None
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        GpawCalculatorSettings(kpoint_density=kpoint_density, kpoint_size=kpoint_size)
