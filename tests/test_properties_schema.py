from typing import Any

import pytest

from crystal.schemas.properties import (
    ForceComponent,
    SinglePointProperties,
    StressComponent,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE

VALID_SINGLE_POINT_FORCES = [[0.0, 0.0, 0.0] for _ in TEST_SI_STRUCTURE]
VALID_SINGLE_POINT_STRESS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def build_single_point_properties(**kwargs: Any) -> SinglePointProperties:
    properties_kwargs = {
        "step": 0,
        "structure": TEST_SI_STRUCTURE,
        "forces": VALID_SINGLE_POINT_FORCES,
        "stress": VALID_SINGLE_POINT_STRESS,
        "energy": 0.0,
        **kwargs,
    }
    return SinglePointProperties(**properties_kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        pytest.param(
            {"forces": [[1.0, 0.0, 0.0]]},
            "forces must have shape",
            id="bad-force-count",
        ),
        pytest.param(
            {"forces": [[1.0, 0.0], [0.0, 0.0, 0.0]]},
            "forces must have shape",
            id="bad-force-width",
        ),
        pytest.param(
            {"stress": [1.0, 0.0, 0.0]},
            "stress must be a Voigt-6 vector",
            id="bad-stress",
        ),
    ],
)
def test_single_point_properties_validates_force_and_stress_shapes(
    kwargs: dict[str, Any], expected_error: str
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        build_single_point_properties(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        pytest.param(
            {"energy": float("nan")},
            "energy must be a finite numeric scalar",
            id="non-finite-energy",
        ),
        pytest.param(
            {"forces": [[float("inf"), 0.0, 0.0], [0.0, 0.0, 0.0]]},
            "forces must contain only finite numeric values",
            id="non-finite-force",
        ),
        pytest.param(
            {"stress": [0.0, 0.0, 0.0, 0.0, 0.0, float("-inf")]},
            "stress must contain only finite numeric values",
            id="non-finite-stress",
        ),
        pytest.param(
            {"charges": [0.0]}, "charges must have shape", id="wrong-charge-count"
        ),
        pytest.param(
            {"charges": [0.0, float("nan")]},
            "charges must contain only finite numeric values",
            id="non-finite-charge",
        ),
    ],
)
def test_single_point_properties_validates_finite_labels_and_charge_shape(
    kwargs: dict[str, Any], expected_error: str
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        build_single_point_properties(**kwargs)


@pytest.mark.parametrize(
    ("component", "expected_ase_index"),
    [(ForceComponent.x, 0), (ForceComponent.y, 1), (ForceComponent.z, 2)],
)
def test_force_component_ase_indices_are_in_ase_force_vector_order(
    component: ForceComponent, expected_ase_index: int
) -> None:
    assert component.value == expected_ase_index


@pytest.mark.parametrize(
    ("component", "expected_ase_index"),
    [
        (StressComponent.xx, 0),
        (StressComponent.yy, 1),
        (StressComponent.zz, 2),
        (StressComponent.yz, 3),
        (StressComponent.xz, 4),
        (StressComponent.xy, 5),
    ],
)
def test_stress_component_ase_indices_are_in_ase_voigt_order(
    component: StressComponent, expected_ase_index: int
) -> None:
    assert component.value == expected_ase_index
