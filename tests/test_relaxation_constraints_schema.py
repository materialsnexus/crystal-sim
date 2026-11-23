"""Unit tests for relaxation constraint schemas."""

from __future__ import annotations

import pytest

from crystal.schemas import relaxation_constraints as relaxation_constraints_schema
from crystal.schemas.relaxation_constraints import (
    CenterFreezeSettings,
    ExternalFacesFreezeSettings,
    RegionBasedFixedAtomSelectionType,
    RelaxationConstraints,
)


def test_center_freeze_settings_requires_positive_thickness_in_angstrom() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        _ = CenterFreezeSettings(
            region_type=RegionBasedFixedAtomSelectionType.center_region,
            thickness_in_angstrom=0.0,
        )


def test_center_freeze_settings_accepts_thickness_in_angstrom() -> None:
    settings = CenterFreezeSettings(
        region_type=RegionBasedFixedAtomSelectionType.center_region,
        thickness_in_angstrom=0.5,
    )
    assert settings.region_type == RegionBasedFixedAtomSelectionType.center_region
    assert settings.thickness_in_angstrom == 0.5


def test_external_faces_freeze_settings_accept_symmetric_thickness() -> None:
    settings = ExternalFacesFreezeSettings(
        region_type=RegionBasedFixedAtomSelectionType.external_faces,
        thickness_in_angstrom=1.5,
    )
    assert settings.region_type == RegionBasedFixedAtomSelectionType.external_faces
    assert settings.thickness_in_angstrom == 1.5


def test_external_faces_freeze_settings_requires_positive_thickness_in_angstrom() -> (
    None
):
    with pytest.raises(ValueError, match="greater than 0"):
        _ = ExternalFacesFreezeSettings(
            region_type=RegionBasedFixedAtomSelectionType.external_faces,
            thickness_in_angstrom=0.0,
        )


def test_external_faces_freeze_settings_require_thickness_definition() -> None:
    with pytest.raises(ValueError, match="Field required"):
        _ = ExternalFacesFreezeSettings.model_validate(
            {"region_type": RegionBasedFixedAtomSelectionType.external_faces}
        )


def test_relaxation_constraints_accept_manual_and_region_based_fixed_atom_selection_settings() -> (
    None
):
    constraints = RelaxationConstraints(
        fix_cell=True,
        fixed_atom_indices=[0, 3],
        region_based_fixed_atom_selection_settings=CenterFreezeSettings(
            region_type=RegionBasedFixedAtomSelectionType.center_region,
            thickness_in_angstrom=2.0,
        ),
    )
    assert constraints.fix_cell
    assert constraints.fixed_atom_indices == [0, 3]
    assert constraints.region_based_fixed_atom_selection_settings is not None


def test_relaxation_constraints_schema_uses_region_based_naming() -> None:
    assert hasattr(relaxation_constraints_schema, "RegionBasedFixedAtomSelectionType")
    assert not hasattr(relaxation_constraints_schema, "AutoFreezeRegionType")
    assert hasattr(
        relaxation_constraints_schema,
        "MINIMUM_REGION_BASED_FIXED_ATOM_SELECTION_THICKNESS_IN_ANGSTROM",
    )
    assert not hasattr(
        relaxation_constraints_schema, "MINIMUM_AUTO_FREEZE_THICKNESS_IN_ANGSTROM"
    )
