"""Tests for the standalone GPAW isolated-atom calculation settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crystal.constants.gpaw import (
    DEFAULT_GPAW_CUTOFF_EV,
    DEFAULT_GPAW_ISOLATED_ATOM_NUMBER_OF_BANDS,
    DEFAULT_GPAW_MAXITER,
    DEFAULT_GPAW_XC,
    GPAW_ISOLATED_ATOM_EIGENSOLVER_NAME,
    GPAW_ISOLATED_ATOM_MIXER_BACKEND,
)
from crystal.schemas.gpaw import (
    GpawIsolatedAtomCalculatorSettings,
    GpawIsolatedAtomConvergenceSettings,
    GpawIsolatedAtomSimulationSettings,
)


def _isolated_atom_calculator_settings(
    *,
    xc: str,
    cutoff_ev: float,
    convergence: GpawIsolatedAtomConvergenceSettings,
    maxiter: int,
) -> GpawIsolatedAtomCalculatorSettings:
    return GpawIsolatedAtomCalculatorSettings(
        xc=xc, cutoff_ev=cutoff_ev, convergence=convergence, maxiter=maxiter
    )


def test_gpaw_isolated_atom_settings_default_vacuum_and_grid() -> None:
    settings = GpawIsolatedAtomSimulationSettings()

    assert settings.vacuum_angstrom == 10.0
    assert settings.calculator_settings.kpoint_size == (1, 1, 1)


def test_gpaw_isolated_atom_settings_reject_periodic_gpaw_settings() -> None:
    with pytest.raises(ValueError, match="extra"):
        GpawIsolatedAtomSimulationSettings(
            calculator_settings={
                "bulk_or_slab": "slab_interface",
                "pbc": (True, True, False),
                "poissonsolver": {"dipolelayer": "xy"},
            }
        )


def test_gpaw_isolated_atom_settings_reject_non_neutral_calculator_recipe() -> None:
    with pytest.raises(ValueError, match="less than or equal to 0"):
        GpawIsolatedAtomSimulationSettings(calculator_settings={"charge": 1.0})


def test_gpaw_isolated_atom_calculator_settings_use_gamma_atom_grid() -> None:
    with pytest.raises(ValueError, match="Input should be True"):
        GpawIsolatedAtomCalculatorSettings(gamma_centered=False)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("spin_polarisation", False),
        ("convergence", {"forces": 0.01}),
        ("occupations", {"name": "fermi-dirac"}),
        ("occupations", {"name": "fixed-uniform", "width": 0.005}),
        ("eigensolver", {"name": "davidson"}),
        ("mixer", {"backend": "pulay"}),
        ("symmetry", "on"),
        ("number_of_bands", 3),
    ],
)
def test_gpaw_isolated_atom_calculator_settings_reject_fixed_policy_changes(
    field_name: str, field_value: object
) -> None:
    with pytest.raises(ValidationError):
        GpawIsolatedAtomCalculatorSettings(**{field_name: field_value})


def test_gpaw_isolated_atom_settings_default_kwargs_use_atom_recipe() -> None:
    kwargs = GpawIsolatedAtomSimulationSettings().build_gpaw_calculator_kwargs()

    assert kwargs["kpts"] == {"size": (1, 1, 1), "gamma": True}
    assert kwargs["xc"] == DEFAULT_GPAW_XC
    assert kwargs["mode"] == {"name": "pw", "ecut": DEFAULT_GPAW_CUTOFF_EV}
    assert kwargs["maxiter"] == DEFAULT_GPAW_MAXITER
    assert kwargs["charge"] == 0.0
    assert kwargs["spinpol"] is True
    assert kwargs["nbands"] == DEFAULT_GPAW_ISOLATED_ATOM_NUMBER_OF_BANDS
    assert kwargs["symmetry"] == "off"
    assert "forces" not in kwargs["convergence"]
    assert "parallel" not in kwargs
    assert kwargs["mixer"] == {
        "backend": GPAW_ISOLATED_ATOM_MIXER_BACKEND,
        "method": "fullspin",
    }
    assert kwargs["eigensolver"] == {"name": GPAW_ISOLATED_ATOM_EIGENSOLVER_NAME}


def test_gpaw_isolated_atom_calculator_settings_reject_parallel_settings() -> None:
    with pytest.raises(ValidationError):
        GpawIsolatedAtomCalculatorSettings(
            parallel={"auto_scalapack_parallelization": False, "use_elpa": False}
        )


def test_gpaw_isolated_atom_settings_build_exact_calculator_recipe() -> None:
    calculator_settings = _isolated_atom_calculator_settings(
        xc="RPBE",
        cutoff_ev=600.0,
        convergence=GpawIsolatedAtomConvergenceSettings(
            density=2e-4, eigenstates=2e-5, forces=None
        ),
        maxiter=400,
    )
    settings = GpawIsolatedAtomSimulationSettings(
        calculator_settings=calculator_settings, vacuum_angstrom=8.0
    )

    assert settings.build_gpaw_calculator_kwargs() == {
        "xc": "RPBE",
        "mode": {"name": "pw", "ecut": 600.0},
        "kpts": {"size": (1, 1, 1), "gamma": True},
        "charge": 0.0,
        "spinpol": True,
        "hund": True,
        "symmetry": "off",
        "nbands": DEFAULT_GPAW_ISOLATED_ATOM_NUMBER_OF_BANDS,
        "maxiter": 400,
        "convergence": {"density": 2e-4, "energy": 5e-4, "eigenstates": 2e-5},
        "occupations": {"name": "fixed-uniform"},
        "mixer": {"backend": GPAW_ISOLATED_ATOM_MIXER_BACKEND, "method": "fullspin"},
        "eigensolver": {"name": GPAW_ISOLATED_ATOM_EIGENSOLVER_NAME},
    }
