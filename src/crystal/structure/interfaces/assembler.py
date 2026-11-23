"""Assemble interfaces from prealigned slabs without ZSL transforms."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, model_validator
from pymatgen.core.lattice import Lattice

from crystal.structure.interfaces.builder import (
    BOTTOM_SLAB_KEY,
    TOP_SLAB_KEY,
    Interface,
)
from crystal.structure.interfaces.utils import (
    compute_estimated_vacuum_gap_in_angstrom,
    compute_interface_gaps,
)
from crystal.structure.slabs.builder import Slab

IN_PLANE_LATTICE_TOLERANCE = 1e-6
IN_PLANE_VECTOR_ABSOLUTE_TOLERANCE = 1e-6
MIN_INTERFACE_PADDING_ANGSTROM = 10.0


class InterfaceAssembler(BaseModel):
    """Assemble interfaces from orthogonal, prealigned slabs.

    Both slabs must already share the same in-plane lattice vectors and be
    orthogonal in the in-plane directions. Typical inputs are slabs extracted
    from a matched `Interface` object.
    """

    top_slab: Slab
    bottom_slab: Slab

    @property
    def estimated_vacuum_gap_in_angstrom(self) -> float:
        """Estimate a geometry-based gap from nearest-neighbor distances."""
        return compute_estimated_vacuum_gap_in_angstrom(
            top_slab=self.top_slab, bottom_slab=self.bottom_slab
        )

    @model_validator(mode="after")
    def _validate_orthogonal_slabs(self) -> "InterfaceAssembler":
        """Reject slabs that are not orthogonal in plane."""
        if not self.top_slab.is_orthogonal_in_plane:
            msg = "Top slab lattice is not orthogonal in-plane"
            raise ValueError(msg)
        if not self.bottom_slab.is_orthogonal_in_plane:
            msg = "Bottom slab lattice is not orthogonal in-plane"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_in_plane_vector_match(self) -> "InterfaceAssembler":
        """Reject slabs with mismatched in-plane lattice vectors."""
        top_in_plane_vectors = self.top_slab.lattice.matrix[:2]
        bottom_in_plane_vectors = self.bottom_slab.lattice.matrix[:2]
        if not np.allclose(
            top_in_plane_vectors,
            bottom_in_plane_vectors,
            rtol=IN_PLANE_LATTICE_TOLERANCE,
            atol=IN_PLANE_VECTOR_ABSOLUTE_TOLERANCE,
        ):
            msg = "Top and bottom slabs must share the same in-plane lattice"
            raise ValueError(msg)
        return self

    def _normalize_offset(self, offset: tuple[float, float]) -> tuple[float, float]:
        """Map an in-plane offset into the [0, 1) fractional range."""
        return (offset[0] % 1.0, offset[1] % 1.0)

    def _merge_site_properties(
        self, bottom_slab: Slab, top_slab: Slab
    ) -> dict[str, list[Any]]:
        bottom_properties = bottom_slab.site_properties or {}
        top_properties = top_slab.site_properties or {}
        property_keys = set(bottom_properties) | set(top_properties)
        merged_properties: dict[str, list[Any]] = {}
        for property_key in property_keys:
            bottom_values = bottom_properties.get(
                property_key, [None] * len(bottom_slab)
            )
            top_values = top_properties.get(property_key, [None] * len(top_slab))
            merged_properties[property_key] = [*bottom_values, *top_values]
        merged_properties["interface_label"] = [
            *[BOTTOM_SLAB_KEY] * len(bottom_slab),
            *[TOP_SLAB_KEY] * len(top_slab),
        ]
        return merged_properties

    def _validate_inputs(
        self, offsets: Sequence[tuple[float, float]], vacuum: float
    ) -> None:
        if len(offsets) == 0:
            msg = "At least one in-plane offset is required"
            raise ValueError(msg)
        if vacuum < 0:
            msg = "Vacuum must be non-negative"
            raise ValueError(msg)

    def _build_interface_properties(
        self, *, gap_in_angstrom: float, normalized_in_plane_offset: tuple[float, float]
    ) -> dict[str, Any]:
        return {
            "assembly_mode": "prealigned",
            "gap_angstrom": gap_in_angstrom,
            "in_plane_offset": normalized_in_plane_offset,
            "offset_target": "top",
        }

    @dataclass(frozen=True)
    class _PreparedAssembly:
        in_plane_a_vector: npt.NDArray[np.float64]
        in_plane_b_vector: npt.NDArray[np.float64]
        slab_normal_unit: npt.NDArray[np.float64]
        substrate_thickness: float
        film_thickness: float
        substrate_cart_coords: npt.NDArray[np.float64]
        film_cart_coords: npt.NDArray[np.float64]
        merged_site_properties: dict[str, list[Any]]
        all_species: list[Any]

    def _prepare_assembly(self) -> _PreparedAssembly:
        top_slab_without_vacuum = self.top_slab.strip_vacuum()
        bottom_slab_without_vacuum = self.bottom_slab.strip_vacuum()
        slab_normal_vector = top_slab_without_vacuum.lattice.matrix[2]
        slab_normal_unit = slab_normal_vector / np.linalg.norm(slab_normal_vector)
        in_plane_a_vector = top_slab_without_vacuum.lattice.matrix[0]
        in_plane_b_vector = top_slab_without_vacuum.lattice.matrix[1]
        substrate_thickness = (
            bottom_slab_without_vacuum.height_in_angstrom_without_vacuum
        )
        film_thickness = top_slab_without_vacuum.height_in_angstrom_without_vacuum
        substrate_cart_coords = np.array(bottom_slab_without_vacuum.cart_coords)
        film_cart_coords = np.array(top_slab_without_vacuum.cart_coords)
        merged_site_properties = self._merge_site_properties(
            bottom_slab_without_vacuum, top_slab_without_vacuum
        )
        all_species = [
            *bottom_slab_without_vacuum.species,
            *top_slab_without_vacuum.species,
        ]
        return self._PreparedAssembly(
            in_plane_a_vector=in_plane_a_vector,
            in_plane_b_vector=in_plane_b_vector,
            slab_normal_unit=slab_normal_unit,
            substrate_thickness=substrate_thickness,
            film_thickness=film_thickness,
            substrate_cart_coords=substrate_cart_coords,
            film_cart_coords=film_cart_coords,
            merged_site_properties=merged_site_properties,
            all_species=all_species,
        )

    def _build_interface(
        self,
        *,
        prepared: _PreparedAssembly,
        gap_in_angstrom: float,
        vacuum: float,
        normalized_in_plane_offset: tuple[float, float],
    ) -> Interface:
        total_c_without_vacuum = (
            prepared.substrate_thickness + prepared.film_thickness + gap_in_angstrom
        )
        total_c_with_vacuum = total_c_without_vacuum + vacuum
        assembled_lattice = Lattice(
            np.vstack(
                [
                    prepared.in_plane_a_vector,
                    prepared.in_plane_b_vector,
                    prepared.slab_normal_unit * total_c_with_vacuum,
                ]
            )
        )
        film_z_shift = (
            prepared.substrate_thickness + gap_in_angstrom
        ) * prepared.slab_normal_unit
        # Gap is applied along the slab normal (c-axis projection).
        in_plane_offset_vector = (
            normalized_in_plane_offset[0] * prepared.in_plane_a_vector
            + normalized_in_plane_offset[1] * prepared.in_plane_b_vector
        )
        adjusted_film_coords = (
            prepared.film_cart_coords + film_z_shift + in_plane_offset_vector
        )
        assembled_cart_coords = np.vstack(
            [prepared.substrate_cart_coords, adjusted_film_coords]
        )
        interface_properties = self._build_interface_properties(
            gap_in_angstrom=gap_in_angstrom,
            normalized_in_plane_offset=normalized_in_plane_offset,
        )
        return Interface(
            lattice=assembled_lattice,
            species=prepared.all_species,
            coords=assembled_cart_coords,
            coords_are_cartesian=True,
            site_properties=prepared.merged_site_properties,
            in_plane_offset=normalized_in_plane_offset,
            gap=gap_in_angstrom,
            vacuum_over_film=vacuum,
            interface_properties=interface_properties,
        )

    def generate_interfaces(
        self,
        *,
        offsets: Sequence[tuple[float, float]],
        gaps: Sequence[float] | None = None,
        gap_multipliers: Sequence[float] | None = None,
        vacuum: float = 0.0,
    ) -> Iterator[Interface]:
        """Yield assembled interface variants for each offset-gap combination.

        Args:
            offsets: Fractional in-plane translations `(fx, fy)`.
            gaps: Explicit interface gap values in angstrom.
            gap_multipliers: Multipliers applied to the estimated gap.
            # NOTE: Only one of `gaps`  and `gap multipliers` must be provided
            vacuum: Requested vacuum above the assembled interface in angstrom.
                Effective vacuum is lower-bounded internally.

        Yields:
            Interface: Assembled interface variant.
            NOTE: `interface_properties`
            includes assembly metadata (`assembly_mode`, `gap_angstrom`,
            `in_plane_offset`, `offset_target`).

        Raises:
            ValueError: If `offsets` is empty, `vacuum` is negative, or both
            `gaps` and `gap_multipliers` are provided.
        """
        self._validate_inputs(offsets=offsets, vacuum=vacuum)
        gap_values_in_angstrom = compute_interface_gaps(
            gaps=gaps,
            multipliers=gap_multipliers,
            estimated_gap=self.estimated_vacuum_gap_in_angstrom,
        )
        effective_vacuum = max(vacuum, MIN_INTERFACE_PADDING_ANGSTROM)
        prepared = self._prepare_assembly()

        for gap_in_angstrom in gap_values_in_angstrom:
            for in_plane_offset in offsets:
                normalized_in_plane_offset = self._normalize_offset(in_plane_offset)
                yield self._build_interface(
                    prepared=prepared,
                    gap_in_angstrom=gap_in_angstrom,
                    vacuum=effective_vacuum,
                    normalized_in_plane_offset=normalized_in_plane_offset,
                )
