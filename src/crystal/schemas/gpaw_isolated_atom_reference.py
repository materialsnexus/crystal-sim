"""Domain contracts for isolated-atom DFT energy references."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.elements import ElementStr
from crystal.schemas.gpaw import (
    GpawEigensolverName,
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
)
from crystal.schemas.serialization import MSONableBaseModel

if TYPE_CHECKING:
    from crystal.calculators.gpaw.isolated_atom_runner import GpawIsolatedAtomRunResult


class IsolatedAtomEnergy(MSONableBaseModel):
    """One successful GPAW isolated-atom energy and runtime provenance."""

    element: ElementStr
    energy: float
    eigensolver: GpawEigensolverName = GpawEigensolverName.etdm_fdpw
    runtime_provenance: GpawRuntimeProvenance

    @property
    def atomic_data(self) -> GpawAtomicDataInfo:
        """Return the selected atomic data for this isolated atom."""
        return self.runtime_provenance.atomic_data_by_element[self.element]

    @model_validator(mode="after")
    def _validate_energy_and_atomic_data(self) -> Self:
        if not isfinite(self.energy):
            msg = "Isolated-atom energy must be finite."
            raise ValueError(msg)
        if self.runtime_provenance.elements != frozenset({self.element}):
            msg = "An isolated-atom calculation must contain atomic data for its element only."
            raise ValueError(msg)
        return self


class IsolatedAtomEnergyReference(MSONableBaseModel):
    """Complete, provenance-consistent GPAW isolated-atom energy reference."""

    simulation_settings: GpawIsolatedAtomSimulationSettings
    isolated_atom_energies: tuple[IsolatedAtomEnergy, ...] = Field(min_length=1)

    @property
    def energies_by_element(self) -> dict[ElementStr, float]:
        """Return physical isolated-atom energies indexed by element."""
        return {
            isolated_atom_energy.element: isolated_atom_energy.energy
            for isolated_atom_energy in self.isolated_atom_energies
        }

    @property
    def atomic_data_by_element(self) -> dict[ElementStr, GpawAtomicDataInfo]:
        """Return reference atomic data indexed by element."""
        return {
            isolated_atom_energy.element: isolated_atom_energy.atomic_data
            for isolated_atom_energy in self.isolated_atom_energies
        }

    @property
    def elements(self) -> frozenset[ElementStr]:
        """Return the chemical elements covered by this reference."""
        return frozenset(self.energies_by_element)

    @model_validator(mode="after")
    def _validate_complete_consistent_entries(self) -> Self:
        if len(self.energies_by_element) != len(self.isolated_atom_energies):
            msg = "Duplicate isolated-atom energy reference elements are not allowed."
            raise ValueError(msg)
        gpaw_versions = {
            isolated_atom_energy.runtime_provenance.gpaw_version
            for isolated_atom_energy in self.isolated_atom_energies
        }
        if len(gpaw_versions) != 1:
            msg = "All isolated-atom energy references must use the same GPAW version."
            raise ValueError(msg)
        return self


class GpawIsolatedAtomTaskDocument(BaseModel):
    """Energy-only result and diagnostics for one GPAW isolated-atom task."""

    model_config = ConfigDict(extra="forbid")

    dir_name: str = Field(description="The directory for this isolated-atom task.")
    element: ElementStr
    errors: str = Field(default="", description="Error log for this task.")
    energy: float | None = None
    magnetic_moment: float | None = None
    eigensolver: GpawEigensolverName | None = None
    runtime_provenance: GpawRuntimeProvenance | None = None

    @property
    def is_success(self) -> bool:
        """Whether the isolated-atom calculation completed successfully."""
        return self.errors == ""

    @property
    def isolated_atom_energy(self) -> IsolatedAtomEnergy:
        """Return the validated energy reference entry for a successful task."""
        return self._build_isolated_atom_energy()

    def _build_isolated_atom_energy(self) -> IsolatedAtomEnergy:
        """Build the energy reference entry after task-output validation."""
        if not self.is_success:
            msg = (
                "A failed GPAW isolated-atom task has no energy reference entry. "
                f"Task error: {self.errors}"
            )
            raise ValueError(msg)
        if self.energy is None or self.runtime_provenance is None:
            msg = (
                "A successful GPAW isolated-atom task requires energy and runtime "
                "provenance."
            )
            raise ValueError(msg)
        return IsolatedAtomEnergy(
            element=self.element,
            energy=self.energy,
            eigensolver=self.eigensolver,
            runtime_provenance=self.runtime_provenance,
        )

    @classmethod
    def from_directory_and_run_result(
        cls, dir_path: Path, run_result: GpawIsolatedAtomRunResult
    ) -> Self:
        """Build a successful task document from an external GPAW run result."""
        return cls(
            dir_name=dir_path.as_posix(),
            element=run_result.element,
            energy=run_result.energy,
            magnetic_moment=run_result.magnetic_moment,
            eigensolver=run_result.eigensolver,
            runtime_provenance=run_result.runtime_provenance,
        )

    @classmethod
    def from_directory_and_error(
        cls, dir_path: Path, error: str, element: ElementStr
    ) -> Self:
        """Build a failed task document while preserving its execution directory."""
        return cls(dir_name=dir_path.as_posix(), element=element, errors=error)

    @model_validator(mode="after")
    def _validate_errors_and_outputs(self) -> Self:
        magnetic_moment = self.magnetic_moment
        if self.is_success:
            required_outputs = {
                "energy": self.energy,
                "magnetic_moment": magnetic_moment,
                "eigensolver": self.eigensolver,
                "runtime_provenance": self.runtime_provenance,
            }
            missing_outputs = [
                name for name, value in required_outputs.items() if value is None
            ]
            if missing_outputs:
                msg = (
                    "A successful GPAW isolated-atom task is missing required "
                    f"outputs: {', '.join(missing_outputs)}."
                )
                raise ValueError(msg)
            if magnetic_moment is None:
                raise AssertionError
            if not isfinite(magnetic_moment):
                msg = "A successful GPAW isolated-atom task requires a finite magnetic moment."
                raise ValueError(msg)
            self._build_isolated_atom_energy()
            return self

        if not self.errors.strip():
            msg = "A failed GPAW isolated-atom task requires a non-blank error message."
            raise ValueError(msg)
        if (
            self.energy is not None
            or magnetic_moment is not None
            or self.eigensolver is not None
            or self.runtime_provenance is not None
        ):
            msg = "A failed GPAW isolated-atom task cannot include computed outputs."
            raise ValueError(msg)
        return self


class GpawIsolatedAtomReferenceCollection(BaseModel):
    """Complete isolated-atom result set and its usable reference when available."""

    model_config = ConfigDict(extra="forbid")

    task_documents: tuple[GpawIsolatedAtomTaskDocument, ...] = Field(min_length=1)
    reference: IsolatedAtomEnergyReference | None = None

    @property
    def is_success(self) -> bool:
        """Whether every atom task completed and produced a usable reference."""
        return self.reference is not None

    @classmethod
    def from_task_documents(
        cls,
        task_documents: tuple[GpawIsolatedAtomTaskDocument, ...],
        simulation_settings: GpawIsolatedAtomSimulationSettings,
    ) -> Self:
        """Aggregate successful atom tasks into a reference without losing failures."""
        reference = None
        if all(task_document.is_success for task_document in task_documents):
            reference = IsolatedAtomEnergyReference(
                simulation_settings=simulation_settings,
                isolated_atom_energies=tuple(
                    task_document.isolated_atom_energy
                    for task_document in task_documents
                ),
            )
        return cls(task_documents=task_documents, reference=reference)

    @model_validator(mode="after")
    def _validate_reference_matches_task_documents(self) -> Self:
        task_elements = {task_document.element for task_document in self.task_documents}
        if len(task_elements) != len(self.task_documents):
            msg = "Duplicate GPAW isolated-atom task document elements are not allowed."
            raise ValueError(msg)
        every_task_succeeded = all(
            task_document.is_success for task_document in self.task_documents
        )
        if every_task_succeeded and self.reference is None:
            msg = "Successful GPAW isolated-atom tasks require an energy reference."
            raise ValueError(msg)
        if not every_task_succeeded and self.reference is not None:
            msg = "A partial GPAW isolated-atom result cannot include an energy reference."
            raise ValueError(msg)
        if self.reference is not None:
            task_energies = tuple(
                task_document.isolated_atom_energy
                for task_document in self.task_documents
            )
            if self.reference.isolated_atom_energies != task_energies:
                msg = "The isolated-atom reference must match its task documents."
                raise ValueError(msg)
        return self


class GpawManualIsolatedAtomReferenceSource(MSONableBaseModel):
    """Use an explicit isolated-atom reference without accessing the registry."""

    source: Literal["manual"] = "manual"
    isolated_atom_reference: IsolatedAtomEnergyReference


class GpawRegistryIsolatedAtomReferenceSource(MSONableBaseModel):
    """Resolve a compatible isolated-atom reference from the GPAW registry."""

    source: Literal["registry"] = "registry"
    simulation_settings: GpawIsolatedAtomSimulationSettings
    eigensolver: GpawEigensolverName | None = None


GpawIsolatedAtomReferenceSource = Annotated[
    GpawManualIsolatedAtomReferenceSource | GpawRegistryIsolatedAtomReferenceSource,
    Field(discriminator="source"),
]
