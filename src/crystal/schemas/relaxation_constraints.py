"""Schemas for atom-freezing settings used during structural relaxation."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated, Literal, TypeAlias, cast

import ase
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

MINIMUM_REGION_BASED_FIXED_ATOM_SELECTION_THICKNESS_IN_ANGSTROM = 0.0
_C_AXIS_INDEX = 2
_SELECTION_TOLERANCE_IN_ANGSTROM = 1e-8
_FloatArray: TypeAlias = npt.NDArray[np.float64]


class RegionBasedFixedAtomSelectionType(StrEnum):
    """Supported regions for automatic atom freezing."""

    center_region = auto()
    external_faces = auto()


class CenterFreezeSettings(BaseModel):
    """Automatic atom-freeze settings for a centered region."""

    region_type: Literal[RegionBasedFixedAtomSelectionType.center_region] = (
        RegionBasedFixedAtomSelectionType.center_region
    )
    thickness_in_angstrom: float = Field(
        gt=MINIMUM_REGION_BASED_FIXED_ATOM_SELECTION_THICKNESS_IN_ANGSTROM
    )


class ExternalFacesFreezeSettings(BaseModel):
    """Automatic atom-freeze settings for top/bottom external faces."""

    region_type: Literal[RegionBasedFixedAtomSelectionType.external_faces] = (
        RegionBasedFixedAtomSelectionType.external_faces
    )
    thickness_in_angstrom: float = Field(
        gt=MINIMUM_REGION_BASED_FIXED_ATOM_SELECTION_THICKNESS_IN_ANGSTROM
    )


RegionBasedFixedAtomSelectionSettings = Annotated[
    CenterFreezeSettings | ExternalFacesFreezeSettings,
    Field(discriminator="region_type"),
]


class RelaxationConstraints(BaseModel):
    """Constraints applied during structural relaxations."""

    fix_cell: bool = True
    fixed_atom_indices: list[int] | None = None
    region_based_fixed_atom_selection_settings: (
        RegionBasedFixedAtomSelectionSettings | None
    ) = None

    @property
    def fix_indices(self) -> list[int]:
        """Return unique, sorted user-provided fixed atom indices."""
        return sorted(set(self.fixed_atom_indices or []))

    @staticmethod
    def compute_atom_coordinates_projected_along_c_axis(  # noqa: D102
        atoms: ase.Atoms,
    ) -> _FloatArray:
        c_vector = np.array(atoms.cell[_C_AXIS_INDEX], dtype=float)
        c_axis_norm = np.linalg.norm(c_vector)
        if c_axis_norm <= 0:
            error_msg = "C-axis vector length must be positive."
            raise ValueError(error_msg)
        c_axis_unit_vector = c_vector / c_axis_norm
        atom_cartesian_coordinates = np.array(atoms.positions, dtype=np.float64)
        return cast("_FloatArray", atom_cartesian_coordinates @ c_axis_unit_vector)

    @staticmethod
    def _select_atom_indices_in_closed_interval(
        projected_coordinates_along_c_axis: _FloatArray,
        lower_bound: float,
        upper_bound: float,
    ) -> list[int]:
        selected_indices: list[int] = []
        for atom_index, coordinate in enumerate(
            projected_coordinates_along_c_axis.tolist()
        ):
            if (
                coordinate >= lower_bound - _SELECTION_TOLERANCE_IN_ANGSTROM
                and coordinate <= upper_bound + _SELECTION_TOLERANCE_IN_ANGSTROM
            ):
                selected_indices.append(atom_index)
        return selected_indices

    @staticmethod
    def select_fixed_atom_indices_for_center_region(  # noqa: D102
        atoms: ase.Atoms, settings: CenterFreezeSettings
    ) -> list[int]:
        projected_coordinates_along_c_axis = (
            RelaxationConstraints.compute_atom_coordinates_projected_along_c_axis(atoms)
        )
        freeze_thickness_in_angstrom = settings.thickness_in_angstrom
        half_freeze_thickness_in_angstrom = 0.5 * freeze_thickness_in_angstrom
        structure_center_along_c_axis = float(
            0.5
            * (
                np.max(projected_coordinates_along_c_axis)
                + np.min(projected_coordinates_along_c_axis)
            )
        )
        lower_bound = structure_center_along_c_axis - half_freeze_thickness_in_angstrom
        upper_bound = structure_center_along_c_axis + half_freeze_thickness_in_angstrom
        return RelaxationConstraints._select_atom_indices_in_closed_interval(
            projected_coordinates_along_c_axis, lower_bound, upper_bound
        )

    @staticmethod
    def select_fixed_atom_indices_for_external_faces(  # noqa: D102
        atoms: ase.Atoms, settings: ExternalFacesFreezeSettings
    ) -> list[int]:
        projected_coordinates_along_c_axis = (
            RelaxationConstraints.compute_atom_coordinates_projected_along_c_axis(atoms)
        )
        minimum_coordinate = float(np.min(projected_coordinates_along_c_axis))
        maximum_coordinate = float(np.max(projected_coordinates_along_c_axis))
        external_faces_freeze_thickness_in_angstrom = settings.thickness_in_angstrom

        selected_indices_set: set[int] = set()
        bottom_upper_bound = (
            minimum_coordinate + external_faces_freeze_thickness_in_angstrom
        )
        selected_indices_set.update(
            RelaxationConstraints._select_atom_indices_in_closed_interval(
                projected_coordinates_along_c_axis,
                minimum_coordinate,
                bottom_upper_bound,
            )
        )
        top_lower_bound = (
            maximum_coordinate - external_faces_freeze_thickness_in_angstrom
        )
        selected_indices_set.update(
            RelaxationConstraints._select_atom_indices_in_closed_interval(
                projected_coordinates_along_c_axis, top_lower_bound, maximum_coordinate
            )
        )
        return sorted(selected_indices_set)

    @staticmethod
    def merge_fixed_atom_indices(  # noqa: D102
        user_defined_indices: list[int] | None, auto_selected_indices: list[int] | None
    ) -> list[int]:
        merged_indices: set[int] = set(user_defined_indices or [])
        merged_indices.update(auto_selected_indices or [])
        return sorted(merged_indices)

    @staticmethod
    def build_fixed_atom_indices_from_region_based_fixed_atom_selection_settings(  # noqa: D102
        atoms: ase.Atoms,
        region_based_fixed_atom_selection_settings: (
            RegionBasedFixedAtomSelectionSettings | None
        ),
    ) -> list[int]:
        if region_based_fixed_atom_selection_settings is None:
            return []
        if isinstance(region_based_fixed_atom_selection_settings, CenterFreezeSettings):
            return RelaxationConstraints.select_fixed_atom_indices_for_center_region(
                atoms, region_based_fixed_atom_selection_settings
            )
        return RelaxationConstraints.select_fixed_atom_indices_for_external_faces(
            atoms, region_based_fixed_atom_selection_settings
        )

    def build_region_based_fixed_atom_indices(self, atoms: ase.Atoms) -> list[int]:  # noqa: D102
        return RelaxationConstraints.build_fixed_atom_indices_from_region_based_fixed_atom_selection_settings(
            atoms, self.region_based_fixed_atom_selection_settings
        )

    def build_fixed_atom_indices(self, atoms: ase.Atoms) -> list[int]:  # noqa: D102
        return RelaxationConstraints.merge_fixed_atom_indices(
            self.fix_indices, self.build_region_based_fixed_atom_indices(atoms)
        )
