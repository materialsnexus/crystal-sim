# noqa: D100

import math
from collections.abc import Iterator
from enum import StrEnum, auto
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import numpy as np
from monty.json import MontyEncoder
from pydantic import Field, field_validator
from pymatgen.analysis.interfaces.coherent_interfaces import (
    from_2d_to_3d,
    get_2d_transform,
)
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, ZSLMatch
from pymatgen.core.elasticity.strain import Deformation, Strain
from pymatgen.core.interface import Interface as PMG_Interface
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.exceptions import NoInterfaceMatchException, OverlappingAtomsException
from crystal.helpers.json import read_json_to_dict, write_data_to_json
from crystal.schemas.serialization import MSONableBaseModel
from crystal.structure.interfaces.utils import compute_estimated_vacuum_gap_in_angstrom
from crystal.structure.slabs.builder import Slab, Termination

DEFAULT_VACUUM_IN_ANGSTROM = 2
MULTIPLIER_FOR_ESTIMATED_VACUUM_GAP = 1.1
DEFAULT_PLACEHOLDER_MILLER_INDICES = (0, 0, 1)
DEFAULT_SCALE_FACTOR = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
TOP_SLAB_KEY = "film"
BOTTOM_SLAB_KEY = "substrate"


class RequiredInterfaceFieldsInStructure(StrEnum):  # noqa: D101
    gap_angstrom = auto()
    in_plane_offset = auto()
    vacuum_over_film = auto()

    @classmethod
    def values(cls) -> tuple[str, ...]:  # noqa: D102
        return tuple(field.value for field in cls)


class Interface(PMG_Interface):
    """Interface object with helpers for extracting reusable slab components.

    This type extends `pymatgen`'s `Interface` with methods used by the
    structure-generation workflow to recover centered top and bottom slabs for
    subsequent interface assembly sweeps.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D107
        super().__init__(*args, **kwargs)

        existing_properties = dict(getattr(self, "interface_properties", {}) or {})

        # Add only missing keys; do not overwrite existing values
        for key, value in self.additional_interface_properties_for_structure.items():
            existing_properties.setdefault(key, value)

        self.interface_properties = existing_properties

        # add also in structure.properties so that all properties translate into atoms objects
        existing_structure_properties = dict(getattr(self, "properties", {}) or {})
        for key, value in self.interface_properties.items():
            existing_structure_properties.setdefault(key, value)
        self.properties = existing_structure_properties

    @cached_property
    def additional_interface_properties_for_structure(  # noqa: D102
        self,
    ) -> dict[RequiredInterfaceFieldsInStructure, Any]:
        return {
            RequiredInterfaceFieldsInStructure.gap_angstrom: self.gap,
            RequiredInterfaceFieldsInStructure.in_plane_offset: self.in_plane_offset,
            RequiredInterfaceFieldsInStructure.vacuum_over_film: self.vacuum_over_film,
        }

    @classmethod
    def from_structure(cls, structure: PMG_Structure) -> "Interface":
        """To create an interface, the following must exist.

        1. structure.properties should contain all interface properties
        2. properties must contain:
            - in_plane_offset
            - gap_angstom
            - vacuum_over_film
        3. structure.site_properties contains:
            - interface_label
        with "substrate"/"film" tag for each site
        """
        interface_properties = getattr(structure, "properties", None)

        if interface_properties is None:
            error_msg = "Cannot create Interface from Structure, it must contain properties with required fields"
            raise ValueError(error_msg)

        missing_fields = [
            field
            for field in RequiredInterfaceFieldsInStructure.values()
            if field not in interface_properties
        ]

        if missing_fields:
            error_msg = (
                "Cannot create Interface from Structure. "
                f"Missing required interface properties: {missing_fields}"
            )
            raise ValueError(error_msg)

        site_properties = getattr(structure, "site_properties", None)
        if (
            site_properties is None
            or "interface_label" not in site_properties
            or len(site_properties["interface_label"]) != len(structure.species)
        ):
            error_msg = (
                "Cannot create Interface from Structure. "
                "Missing required interface labels in site_properties"
            )
            raise ValueError(error_msg)

        interface_labels = site_properties["interface_label"]
        valid_labels = (TOP_SLAB_KEY, BOTTOM_SLAB_KEY)
        invalid_labels = [
            label
            for label in interface_labels
            if not isinstance(label, str) or label not in valid_labels
        ]
        if invalid_labels:
            error_msg = (
                "Cannot create Interface from Structure. Invalid interface_label "
                f"values: {invalid_labels!r}; expected only {valid_labels!r}"
            )
            raise ValueError(error_msg)

        missing_labels = [
            required_label
            for required_label in valid_labels
            if required_label not in interface_labels
        ]
        if missing_labels:
            error_msg = (
                "Cannot create Interface from Structure. Interface labels must "
                f"include both {valid_labels!r}; missing {missing_labels!r}"
            )
            raise ValueError(error_msg)

        return cls(
            lattice=structure.lattice,
            species=structure.species,
            coords=structure.cart_coords,
            site_properties=structure.site_properties,
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=True,
            in_plane_offset=interface_properties[
                RequiredInterfaceFieldsInStructure.in_plane_offset
            ],
            gap=interface_properties[RequiredInterfaceFieldsInStructure.gap_angstrom],
            vacuum_over_film=interface_properties[
                RequiredInterfaceFieldsInStructure.vacuum_over_film
            ],
            interface_properties=interface_properties,
        )

    @classmethod
    def read_from_json(cls, filepath: Path) -> Self:  # noqa: D102
        data_as_dict = read_json_to_dict(filepath=filepath)
        return cls.from_dict(data_as_dict)

    def write_to_json(self, filepath: Path) -> None:  # noqa: D102
        write_data_to_json(
            filepath=filepath, data=self.as_dict(), mode="w", encoder=MontyEncoder
        )

    def _get_individual_slab(self, structure: PMG_Structure) -> Slab:
        # NOTE: This includes geometric representation of the slab wrt the oriented unit cell and hence the miller index of the surface plane is always 001
        # The crystallographic miller index wrt the initial structure (conventional unit cells) can be obtained from interface_properties.top_slab_miller_indices or interface_properties.bottom_slab_miller_indices
        return Slab(
            lattice=structure.lattice,
            species=structure.species,
            coords=structure.frac_coords,
            miller_index=DEFAULT_PLACEHOLDER_MILLER_INDICES,
            oriented_unit_cell=structure,
            shift=0.0,
            scale_factor=DEFAULT_SCALE_FACTOR,
            reorient_lattice=False,
            site_properties=structure.site_properties,
            coords_are_cartesian=False,
        )

    def get_centered_top_slab(self, vacuum_in_angstrom: float | None = None) -> Slab:
        """Return the top slab extracted from the interface and centered in `c`.

        Args:
            vacuum_in_angstrom: Optional vacuum to add on each side while
                recentering the extracted slab.

        Returns:
            Slab: Centered top slab suitable for downstream assembly.
        """
        top_slab = self._get_individual_slab(structure=self.film)
        return top_slab.get_centered_slab(vacuum_in_angstrom=vacuum_in_angstrom)

    def get_centered_bottom_slab(self, vacuum_in_angstrom: float | None = None) -> Slab:
        """Return the bottom slab extracted from the interface and centered in `c`.

        Args:
            vacuum_in_angstrom: Optional vacuum to add on each side while
                recentering the extracted slab.

        Returns:
            Slab: Centered bottom slab suitable for downstream assembly.
        """
        bottom_slab = self._get_individual_slab(structure=self.substrate)
        return bottom_slab.get_centered_slab(vacuum_in_angstrom=vacuum_in_angstrom)


class InterfaceProperties(MSONableBaseModel):
    """Metadata recorded for each matched interface candidate."""

    green_lagrange_strain_tensor: Strain
    von_mises_strain: float
    terminations: tuple[Termination, Termination]
    top_slab_thickness_in_angstrom: float
    top_slab_miller_indices: tuple[int, int, int]
    bottom_slab_thickness_in_angstrom: float
    bottom_slab_miller_indices: tuple[int, int, int]
    match_area_in_angstrom_square: float


class InterfaceFilter(StrEnum):
    """Supported policies for selecting matches from generated ZSL candidates."""

    minimum_area = auto()


class InterfaceMatchPolicy(MSONableBaseModel):
    """Tolerances and filtering policy used during ZSL lattice matching.

    Attributes:
        maximum_area_ratio_tolerance: Relative tolerance on superlattice area
            ratio during matching.
        maximum_area_in_angstrom_square: Maximum allowed in-plane match area.
        length_tolerance_in_angstrom: Relative (unitless) tolerance on in-plane
            vector length matching. Field name is retained for schema
            compatibility.
        angle_tolerance_in_radian: Relative (unitless) tolerance on in-plane
            vector-angle matching. Field name is retained for schema
            compatibility.
        filter_by: Optional policy used to filter generated matches.
    """

    maximum_area_ratio_tolerance: float = Field(
        default=0.01, description="Maximum tolerance on ratio of slab superlattices"
    )
    maximum_area_in_angstrom_square: float = Field(
        default=100, description="Maximum superlattice area"
    )
    length_tolerance_in_angstrom: float = Field(
        default=0.01,
        description="Maximum length tolerance to check if two vectors are of equivalent length",
    )
    angle_tolerance_in_radian: float = Field(
        default=0.01,
        description="Maximum angle tolerance to check if two vectors have equivalent angle between them",
    )
    filter_by: InterfaceFilter | None = Field(default=InterfaceFilter.minimum_area)


class InterfaceFactory(MSONableBaseModel):
    """Generate orthogonal, strain-matched interfaces from top and bottom slabs.

    Input slabs are orthogonalized in-plane before matching. The matching stage
    uses Zur and McGill lattice matching through pymatgen's `ZSLGenerator`.

    Rigid-body rotation is intentionally not implemented because it can break
    periodicity in this interface-construction flow.
    """

    top_slab: Slab
    bottom_slab: Slab
    interface_match_policy: InterfaceMatchPolicy = Field(
        default_factory=InterfaceMatchPolicy
    )

    @field_validator("top_slab", "bottom_slab")
    @classmethod
    def _orthogonalize_slabs_in_xy_plane(cls, slab: Slab) -> Slab:
        """Ensure slabs used by the factory are orthogonal in-plane."""
        return slab.orthogonalize_in_xy_plane

    @property
    def estimated_vacuum_gap_in_angstrom(self) -> float:
        """Estimate a safe inter-slab gap from nearest-neighbor distances."""
        assert isinstance(self.top_slab, Slab)
        assert isinstance(self.bottom_slab, Slab)
        return compute_estimated_vacuum_gap_in_angstrom(
            top_slab=self.top_slab,
            bottom_slab=self.bottom_slab,
            safety_multiplier=MULTIPLIER_FOR_ESTIMATED_VACUUM_GAP,
        )

    @cached_property
    def _zsl_generator(self) -> ZSLGenerator:
        return ZSLGenerator(  # type: ignore[no-untyped-call]
            max_area_ratio_tol=self.interface_match_policy.maximum_area_ratio_tolerance,
            max_area=self.interface_match_policy.maximum_area_in_angstrom_square,
            max_length_tol=self.interface_match_policy.length_tolerance_in_angstrom,
            max_angle_tol=self.interface_match_policy.angle_tolerance_in_radian,
            bidirectional=True,
        )

    @cached_property
    def _get_matches_for_top_and_bottom_slabs(self) -> list[ZSLMatch]:
        assert isinstance(self.top_slab, Slab)
        assert isinstance(self.bottom_slab, Slab)
        matches = list(
            self._zsl_generator(
                film_vectors=self.top_slab.lattice.matrix[:2],
                substrate_vectors=self.bottom_slab.lattice.matrix[:2],
                lowest=False,
            )
        )
        if not matches:
            message = (
                "No ZSL interface match was found for slab pair "
                f"{self.top_slab.miller_index}/{self.bottom_slab.miller_index}. "
                "Try loosening the interface matching tolerances or increasing "
                "the maximum match area."
            )
            raise NoInterfaceMatchException(message)
        return matches

    @property
    def _filtered_matches(self) -> list[ZSLMatch]:
        all_matches = self._get_matches_for_top_and_bottom_slabs
        assert isinstance(self.interface_match_policy, InterfaceMatchPolicy)
        assert self.interface_match_policy.filter_by is not None

        match self.interface_match_policy.filter_by:
            case InterfaceFilter.minimum_area:
                [match_with_minimum_area, *_] = sorted(
                    all_matches, key=lambda x: x.match_area
                )
                minimum_match_area = match_with_minimum_area.match_area
                return [
                    m
                    for m in all_matches
                    if math.isclose(m.match_area, minimum_match_area, abs_tol=1e-2)
                ]
            case _:
                raise NotImplementedError

    def _build_slab_for_interface(
        self, initial_slab: Slab, match_superlattice_vectors: list[list[int]]
    ) -> Slab:
        assert isinstance(initial_slab, Slab)
        superlattice = np.round(
            from_2d_to_3d(
                get_2d_transform(
                    initial_slab.lattice.matrix[:2].tolist(), match_superlattice_vectors
                )
            )
        ).astype(int)
        slab_in_interface_structure = initial_slab.copy()
        slab_in_interface_structure.make_supercell(superlattice)
        return slab_in_interface_structure

    def _get_interface_properties(self, match: ZSLMatch) -> dict[str, Any]:
        assert isinstance(self.top_slab, Slab)
        assert isinstance(self.bottom_slab, Slab)

        deformation = Deformation(match.match_transformation)
        strain = deformation.green_lagrange_strain
        interface_properties = InterfaceProperties(
            green_lagrange_strain_tensor=strain,
            von_mises_strain=strain.von_mises_strain,
            terminations=(
                self.top_slab.bottom_layer_termination,
                self.bottom_slab.top_layer_termination,
            ),
            top_slab_thickness_in_angstrom=self.top_slab.height_in_angstrom_without_vacuum,
            top_slab_miller_indices=self.top_slab.miller_index,
            bottom_slab_thickness_in_angstrom=self.bottom_slab.height_in_angstrom_without_vacuum,
            bottom_slab_miller_indices=self.bottom_slab.miller_index,
            match_area_in_angstrom_square=match.match_area,
        )

        # Add extra info
        match_dict = match.as_dict()
        interface_properties_from_match = {
            k: match_dict[k] for k in match_dict if not k.startswith("@")
        }
        return {**interface_properties_from_match, **interface_properties.model_dump()}

    def _get_interfaces(
        self, gap: float, vacuum: float, in_plane_offset: tuple[float, float]
    ) -> Iterator[Interface]:
        assert isinstance(self.top_slab, Slab)
        assert isinstance(self.bottom_slab, Slab)

        filtered_matches = (
            self._filtered_matches
            if self.interface_match_policy.filter_by is not None
            else self._get_matches_for_top_and_bottom_slabs
        )

        for match in filtered_matches:
            top_slab_in_interface_structure = self._build_slab_for_interface(
                initial_slab=self.top_slab,
                match_superlattice_vectors=match.film_sl_vectors,
            )

            bottom_slab_in_interface_structure = self._build_slab_for_interface(
                initial_slab=self.bottom_slab,
                match_superlattice_vectors=match.substrate_sl_vectors,
            )

            interface_properties_dict = self._get_interface_properties(match=match)
            yield Interface.from_slabs(
                substrate_slab=bottom_slab_in_interface_structure,
                # From slabs method flips the film / top slab
                film_slab=top_slab_in_interface_structure.mirrored,
                gap=gap,
                vacuum_over_film=vacuum,
                in_plane_offset=in_plane_offset,
                interface_properties=interface_properties_dict,
            )

    def generate_interfaces(
        self,
        gap: float | None = None,
        vacuum: float = DEFAULT_VACUUM_IN_ANGSTROM,
        in_plane_offset: tuple[float, float] = (0, 0),
    ) -> Iterator[Interface]:
        """Yield matched interface candidates for the configured slab pair.

        Args:
            gap: Gap between substrate and film in angstrom. If `None`, a
                geometry-based estimate is used.
            vacuum: Vacuum above the film in angstrom.
            in_plane_offset: Fractional in-plane shift `(fx, fy)` applied during
                interface construction.

        Yields:
            Interface: Matched interface candidate with strain and matching
            metadata stored in `interface_properties`.

        Raises:
            OverlappingAtomsException: If generated geometry contains overlapping
                atoms under the requested gap or inferred default gap.
        """
        gap = gap or self.estimated_vacuum_gap_in_angstrom
        for interface in self._get_interfaces(
            vacuum=vacuum, gap=gap, in_plane_offset=in_plane_offset
        ):
            if not interface.is_valid():
                error_msg = "Overlapping atoms found in interface structure, gap should be set to None or provide appropriate interface gap"
                raise OverlappingAtomsException(error_msg)
            yield interface
