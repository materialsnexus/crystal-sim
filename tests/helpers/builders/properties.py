from __future__ import annotations

from copy import deepcopy

from pymatgen.core.structure import Structure as PMG_Structure

from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from tests.helpers.constants.structure import (
    TEST_RELAX_ENERGY_PLACEHOLDER,
    TEST_RELAX_STEP_INDEX,
    TEST_RELAX_STRESS_VECTOR_ZERO,
    TEST_RELAX_ZERO_FORCE_VECTOR,
)


def build_single_point_properties_for_test(
    structure: PMG_Structure, *, energy: float = TEST_RELAX_ENERGY_PLACEHOLDER
) -> SinglePointProperties:
    return SinglePointProperties(
        step=TEST_RELAX_STEP_INDEX,
        structure=deepcopy(structure),
        forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in range(len(structure))],
        stress=TEST_RELAX_STRESS_VECTOR_ZERO,
        energy=energy,
    )


def build_relax_properties_for_test(
    *, structure: PMG_Structure, energy: float
) -> RelaxProperties:
    return RelaxProperties(
        relax_steps=[
            SinglePointProperties(
                step=TEST_RELAX_STEP_INDEX,
                structure=deepcopy(structure),
                forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in structure],
                stress=TEST_RELAX_STRESS_VECTOR_ZERO,
                energy=energy,
            )
        ]
    )
