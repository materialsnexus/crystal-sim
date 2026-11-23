"""MLIP runner utilities."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import ase
from ase.calculators.calculator import BaseCalculator
from pydantic import Field
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.base_atomistic_runner import BaseAtomisticRunner
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.helpers.mlip_helpers import prepare_atoms_for_matgl
from crystal.schemas.isolated_atom_energies import IsolatedAtomEnergies
from crystal.schemas.mlip import (
    MATGL_MISSING_ERROR_MESSAGE,
    MlipRelaxSimulationSettings,
    MlipRelaxTaskDocument,
    MlipSinglePointSimulationSettings,
    MlipSinglePointTaskDocument,
)
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.properties import (
    RelaxProperties,
    SinglePointProperties,
    StructureList,
)

if TYPE_CHECKING:
    from matgl.apps.pes import Potential
    from matgl.ext.ase import PESCalculator

logger = logging.getLogger(__name__)


class MlipModelError(TypeError):
    """Raised when an MLIP model does not provide required metadata."""


@runtime_checkable
class _ElementTypedMlipModel(Protocol):
    element_types: tuple[str, ...]


def _load_mlip_potential(
    model: MlipModelRef, *, move_to_default_device: bool = True
) -> Potential:
    try:
        import matgl  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from matgl.apps.pes import Potential  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(MATGL_MISSING_ERROR_MESSAGE) from exc

    potential = matgl.load_model(model.model_path)
    if not isinstance(potential, Potential):
        msg = (
            "MLIP calculations require a MatGL Potential: "
            f"loaded_type={type(potential).__name__}."
        )
        raise MlipModelError(msg)
    if move_to_default_device:
        potential.to(device=torch.get_default_device())
    return potential


def _load_mlip_pes_calculator(model: MlipModelRef) -> PESCalculator:
    from matgl.ext.ase import PESCalculator  # noqa: PLC0415

    return PESCalculator(potential=_load_mlip_potential(model), stress_unit="eV/A3")


def _get_mlip_model_element_types(
    *, model: object, model_path: str | Path
) -> tuple[str, ...]:
    msg = f"MLIP model {model_path} does not contain element types."
    if not isinstance(model, _ElementTypedMlipModel):
        raise MlipModelError(msg)
    element_types = model.element_types
    if not element_types:
        raise MlipModelError(msg)
    return element_types


def get_mlip_isolated_atom_energies(model: MlipModelRef) -> IsolatedAtomEnergies:
    """Return the model's per-element energy references."""
    potential = _load_mlip_potential(model, move_to_default_device=False)
    element_types = _get_mlip_model_element_types(
        model=potential.model, model_path=model.model_path
    )
    element_refs = potential.element_refs
    if element_refs is None:
        msg = f"MLIP model {model.model_path} does not contain element references."
        raise MlipModelError(msg)
    reference_values = element_refs.property_offset.detach().cpu().tolist()
    if len(reference_values) != len(element_types):
        msg = (
            "MLIP model element reference count does not match its element type "
            f"count: {len(reference_values)} != {len(element_types)}."
        )
        raise MlipModelError(msg)
    return {
        element: float(reference_value)
        for element, reference_value in zip(
            element_types, reference_values, strict=True
        )
    }


class MlipRunner(BaseAtomisticRunner):  # noqa: D101
    model: MlipModelRef
    calculator: BaseCalculator | None = Field(default=None, exclude=True)

    @cached_property
    def _ase_atoms(self) -> ase.Atoms:
        atoms = super()._ase_atoms
        return prepare_atoms_for_matgl(atoms)

    @cached_property
    def _calculator(self) -> BaseCalculator:
        if self.calculator is not None:
            return self.calculator
        return _load_mlip_pes_calculator(model=self.model)


def run_mlip_single_point(  # noqa: D103
    structure: PMG_Structure, simulation_settings: MlipSinglePointSimulationSettings
) -> SinglePointProperties:
    runner = MlipRunner(structure=structure, model=simulation_settings.model)
    return runner.run_single_point()


def run_mlip_single_point_batch(
    structures: StructureList, simulation_settings: MlipSinglePointSimulationSettings
) -> list[MlipSinglePointTaskDocument]:
    """Run MLIP single points with one shared calculator and ordered outputs."""
    if not structures:
        return []

    try:
        calculator = _load_mlip_pes_calculator(model=simulation_settings.model)
    except Exception as exc:  # noqa: BLE001, RUF100
        error = get_diagnostic_exception_message(exc)
        logger.exception("MLIP batch calculator loading failed.")
        return [
            MlipSinglePointTaskDocument(properties=None, errors=error)
            for _ in structures
        ]

    task_documents: list[MlipSinglePointTaskDocument] = []
    for structure in structures:
        try:
            properties = MlipRunner(
                structure=structure,
                model=simulation_settings.model,
                calculator=calculator,
            ).run_single_point()
            task_documents.append(MlipSinglePointTaskDocument(properties=properties))
        except Exception as exc:  # noqa: BLE001
            error = get_diagnostic_exception_message(exc)
            task_documents.append(
                MlipSinglePointTaskDocument(properties=None, errors=error)
            )

    return task_documents


def run_mlip_relax(
    structure: PMG_Structure, simulation_settings: MlipRelaxSimulationSettings
) -> RelaxProperties:
    """Run an MLIP relaxation and return relaxation properties."""
    runner = MlipRunner(structure=structure, model=simulation_settings.model)
    properties = runner.run_relax(
        optimizer_settings=simulation_settings,
        constraints=simulation_settings.constraints,
    )
    if simulation_settings.store_final_step_properties_only:
        return RelaxProperties(relax_steps=[properties.relax_steps[-1]])
    return properties


def run_mlip_relax_batch(
    structures: StructureList,
    simulation_settings: MlipRelaxSimulationSettings,
    *,
    run_directories: list[Path],
) -> list[MlipRelaxTaskDocument]:
    """Run MLIP relaxations with one shared calculator for all structures.

    The returned documents preserve input order and represent per-structure
    failures without aborting the rest of the batch.
    """
    if len(run_directories) != len(structures):
        msg = "MLIP relaxation batch requires one run directory per structure."
        raise ValueError(msg)

    if not structures:
        return []

    try:
        calculator = _load_mlip_pes_calculator(model=simulation_settings.model)
    except Exception as exc:  # noqa: BLE001, RUF100
        error = get_diagnostic_exception_message(exc)
        logger.exception("MLIP batch calculator loading failed.")
        return [
            MlipRelaxTaskDocument(
                dir_name=run_directory.as_posix(), properties=None, errors=error
            )
            for run_directory in run_directories
        ]

    task_documents: list[MlipRelaxTaskDocument] = []
    for structure, run_directory in zip(structures, run_directories, strict=True):
        try:
            properties = MlipRunner(
                structure=structure,
                model=simulation_settings.model,
                calculator=calculator,
            ).run_relax(
                optimizer_settings=simulation_settings,
                constraints=simulation_settings.constraints,
                store_trajectory=False,
                output_directory=run_directory,
            )
            task_documents.append(
                MlipRelaxTaskDocument(
                    dir_name=run_directory.as_posix(), properties=properties
                )
            )
        except Exception as exc:  # noqa: BLE001
            error = get_diagnostic_exception_message(exc)
            task_documents.append(
                MlipRelaxTaskDocument(
                    dir_name=run_directory.as_posix(), properties=None, errors=error
                )
            )

    return task_documents
