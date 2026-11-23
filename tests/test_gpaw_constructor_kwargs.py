"""Tests for GPAW constructor kwargs exported from schema settings."""

from __future__ import annotations

import pytest
from gpaw import GPAW

from crystal.constants.gpaw import DEFAULT_GPAW_FERMI_DIRAC_WIDTH
from crystal.schemas.gpaw import GpawSettings


@pytest.mark.gpaw_for_tests
def test_gpaw_constructor_accepts_exported_kwargs() -> None:
    settings = GpawSettings()
    kwargs = settings.build_gpaw_calculator_kwargs()
    calculator = GPAW(**kwargs)
    assert hasattr(calculator, "set")


@pytest.mark.gpaw_for_tests
def test_gpaw_constructor_accepts_fermi_dirac_kwargs() -> None:
    settings = GpawSettings(occupations={"name": "fermi-dirac"})
    kwargs = settings.build_gpaw_calculator_kwargs()

    calculator = GPAW(**kwargs)

    assert hasattr(calculator, "set")
    assert kwargs["occupations"] == {
        "name": "fermi-dirac",
        "width": DEFAULT_GPAW_FERMI_DIRAC_WIDTH,
    }
