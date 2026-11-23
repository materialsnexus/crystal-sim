"""Tests for property helpers."""

import pytest

from crystal.helpers.properties import (
    get_force_norms,
    get_maximum_force_norm,
    get_stress_norm,
    get_stress_norms,
)
from crystal.schemas.properties import SinglePointProperties
from tests.helpers.constants.structure import TEST_SI_STRUCTURE as MOCK_SI_STRUCTURE

MOCK_SI_STRUCTURE_ENERGY = -6.0
MOCK_SI_STRUCTURE_FORCES = [[0.3, 0.4, 0.0], [0.0, 0.0, 0.2]]
MOCK_SI_STRUCTURE_STRESS = [1.0, 2.0, 2.0, 0.0, 0.0, 0.0]
MOCK_SINGLE_POINT_PROPERTIES = SinglePointProperties(
    step=0,
    structure=MOCK_SI_STRUCTURE,
    energy=MOCK_SI_STRUCTURE_ENERGY,
    forces=MOCK_SI_STRUCTURE_FORCES,
    stress=MOCK_SI_STRUCTURE_STRESS,
)


def test_get_force_norms() -> None:
    expected_force_norms = [
        sum(component**2 for component in force) ** 0.5
        for force in MOCK_SI_STRUCTURE_FORCES
    ]

    assert get_force_norms(MOCK_SINGLE_POINT_PROPERTIES.forces) == pytest.approx(
        expected_force_norms
    )


def test_get_maximum_force_norm() -> None:
    expected_max_force_norm = max(
        sum(component**2 for component in force) ** 0.5
        for force in MOCK_SI_STRUCTURE_FORCES
    )

    assert get_maximum_force_norm(MOCK_SINGLE_POINT_PROPERTIES.forces) == pytest.approx(
        expected_max_force_norm
    )


def test_get_stress_norm_uses_voigt_vector_norm() -> None:
    expected_stress_norm = (
        sum(component**2 for component in MOCK_SI_STRUCTURE_STRESS) ** 0.5
    )

    assert get_stress_norm(MOCK_SINGLE_POINT_PROPERTIES.stress) == pytest.approx(
        expected_stress_norm
    )


def test_get_stress_norms_uses_voigt_vector_norms() -> None:
    stresses = [[0.0, 0.0, 0.0, 0.0, 0.0, 5.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    assert get_stress_norms(stresses) == pytest.approx([5.0, 6**0.5])
