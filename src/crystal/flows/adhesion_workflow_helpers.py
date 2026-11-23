"""Helpers for adhesion workflow orchestration."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.constants.adhesion import (
    INTERFACE_CHARGEMOL_NOT_CONVERGED_ERROR,
    SLAB_CHARGEMOL_NOT_CONVERGED_ERROR,
)
from crystal.schemas.adhesion import AdhesionEnergyRecord
from crystal.schemas.properties import RelaxProperties
from crystal.schemas.single_point import SinglePointWithChargesTaskDocument
from crystal.structure.slabs.builder import Slab


def relax_properties_converged(  # noqa: D103
    relax_properties: Iterable[RelaxProperties | None],
) -> bool:
    return all(properties is not None for properties in relax_properties)


def extract_relax_final_structure_or_raise(  # noqa: D103
    relax_properties: RelaxProperties | None, error_message: str
) -> PMG_Structure:
    if relax_properties is None:
        raise ValueError(error_message)
    return relax_properties.relaxed_structure


def build_slab_from_structure_with_slab_metadata(  # noqa: D103
    slab: Slab, structure: PMG_Structure
) -> Slab:
    slab_kwargs = {
        "lattice": structure.lattice,
        "species": structure.species,
        "coords": structure.frac_coords,
        "miller_index": slab.miller_index,
        "oriented_unit_cell": slab.oriented_unit_cell,
        "shift": slab.shift,
        "scale_factor": slab.scale_factor,
        "site_properties": slab.site_properties,
        "coords_are_cartesian": False,
    }
    if hasattr(slab, "reorient_lattice"):
        slab_kwargs["reorient_lattice"] = slab.reorient_lattice
    return type(slab)(**slab_kwargs)


def extract_single_point_energy_or_raise(  # noqa: D103
    task_doc: SinglePointWithChargesTaskDocument, error_message: str
) -> float:
    if not task_doc.is_success or task_doc.properties is None:
        raise ValueError(error_message)
    return float(task_doc.properties.energy)


def extract_interface_area_in_angstrom_square_or_raise(  # noqa: D103
    task_doc: SinglePointWithChargesTaskDocument, error_message: str
) -> float:
    if not task_doc.is_success or task_doc.structure is None:
        raise ValueError(error_message)
    lattice_vectors = np.array(task_doc.structure.lattice.matrix, dtype=float)
    in_plane_vector_a = lattice_vectors[0]
    in_plane_vector_b = lattice_vectors[1]
    return float(np.linalg.norm(np.cross(in_plane_vector_a, in_plane_vector_b)))


def build_adhesion_energy_records_from_single_point_task_docs(  # noqa: D103
    *,
    slab_task_docs: Iterable[SinglePointWithChargesTaskDocument],
    interface_task_docs: Iterable[SinglePointWithChargesTaskDocument],
    slab_error_message: str = SLAB_CHARGEMOL_NOT_CONVERGED_ERROR,
    interface_error_message: str = INTERFACE_CHARGEMOL_NOT_CONVERGED_ERROR,
) -> list[AdhesionEnergyRecord]:
    slab_energy_sum = sum(
        extract_single_point_energy_or_raise(doc, slab_error_message)
        for doc in slab_task_docs
    )
    adhesion_records = []
    for index, interface_task_doc in enumerate(interface_task_docs):
        interface_energy = extract_single_point_energy_or_raise(
            interface_task_doc, interface_error_message
        )
        interface_area_in_angstrom_square = (
            extract_interface_area_in_angstrom_square_or_raise(
                interface_task_doc, interface_error_message
            )
        )
        adhesion_records.append(
            AdhesionEnergyRecord(
                interface_index=index,
                adhesion_energy=interface_energy - slab_energy_sum,
                interface_area_in_angstrom_square=interface_area_in_angstrom_square,
            )
        )
    return adhesion_records
