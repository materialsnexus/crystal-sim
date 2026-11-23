"""Domain contracts for reusable GPAW isolated-atom energy registry entries."""

from __future__ import annotations

from math import isfinite
from typing import Final, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from crystal.helpers.job import hash_object
from crystal.schemas.atomic_data import SHA256_DIGEST_PATTERN, GpawAtomicDataInfo
from crystal.schemas.elements import ElementStr
from crystal.schemas.gpaw import (
    GpawEigensolverName,
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
)
from crystal.schemas.serialization import MSONableBaseModel

GPAW_ISOLATED_ATOM_REGISTRY_SCHEMA_VERSION: Final = "2"
GPAW_ISOLATED_ATOM_REGISTRY_RECIPE_VERSION: Final = "1"
GPAW_ISOLATED_ATOM_RECIPE_FINGERPRINT_KIND: Final = "gpaw-isolated-atom-recipe"
GPAW_ISOLATED_ATOM_RUNTIME_FINGERPRINT_KIND: Final = "gpaw-isolated-atom-runtime"


class GpawIsolatedAtomRegistryIdentity(MSONableBaseModel):
    """Storage identity for reusable GPAW isolated-atom registry entries.

    The registry repository uses this identity to deduplicate successful
    isolated-atom calculations and retrieve compatible stored energies. It is the deterministic key for locating that
    result.
    """

    model_config = ConfigDict(extra="forbid")

    registry_schema_version: Literal["2"] = Field(
        default=GPAW_ISOLATED_ATOM_REGISTRY_SCHEMA_VERSION,
        description="Version of the GpawIsolatedAtomRegistryIdentity storage and lookup contract. Change when its fields change.",
    )
    recipe_version: Literal["1"] = Field(
        default=GPAW_ISOLATED_ATOM_REGISTRY_RECIPE_VERSION,
        description="Version of the GpawIsolatedAtomSimulationSettings fingerprint contract. Change when its inputs change.",
    )
    element: ElementStr = Field(
        description="Chemical element of the isolated atom represented by the entry."
    )
    eigensolver: GpawEigensolverName = Field(
        description="Eigensolver that produced the stored isolated-atom result."
    )
    gpaw_version: str = Field(
        min_length=1, description="GPAW version that generated the result."
    )
    recipe_fingerprint: str = Field(
        pattern=SHA256_DIGEST_PATTERN,
        description="SHA-256 hash of the complete isolated-atom simulation settings.",
    )
    runtime_fingerprint: str = Field(
        pattern=SHA256_DIGEST_PATTERN,
        description="SHA-256 hash of the resolved element-specific PAW setup.",
    )

    @property
    def entry_fingerprint(self) -> str:
        """Return the registry key combining all identity-defining components."""
        return hash_object(
            {
                "element": self.element,
                "eigensolver": self.eigensolver,
                "gpaw_version": self.gpaw_version,
                "recipe_fingerprint": self.recipe_fingerprint,
                "runtime_fingerprint": self.runtime_fingerprint,
                "registry_schema_version": self.registry_schema_version,
            }
        )

    @classmethod
    def recipe_fingerprint_for_simulation_settings(
        cls, simulation_settings: GpawIsolatedAtomSimulationSettings
    ) -> str:
        """Fingerprint the complete requested isolated-atom calculation recipe."""
        return hash_object(
            {
                "kind": GPAW_ISOLATED_ATOM_RECIPE_FINGERPRINT_KIND,
                "recipe_version": GPAW_ISOLATED_ATOM_REGISTRY_RECIPE_VERSION,
                "registry_schema_version": GPAW_ISOLATED_ATOM_REGISTRY_SCHEMA_VERSION,
                "simulation_settings": simulation_settings.model_dump(
                    mode="json", exclude_none=True
                ),
            }
        )

    @classmethod
    def runtime_fingerprint_for_paw_data(cls, paw_data: GpawAtomicDataInfo) -> str:
        """Fingerprint PAW data selected for one isolated-atom calculation."""
        return hash_object(
            {
                "kind": GPAW_ISOLATED_ATOM_RUNTIME_FINGERPRINT_KIND,
                "registry_schema_version": GPAW_ISOLATED_ATOM_REGISTRY_SCHEMA_VERSION,
                "paw_setup_sha256_digest": paw_data.paw_setup_sha256_digest,
            }
        )

    @classmethod
    def from_simulation_settings_and_runtime_provenance(
        cls,
        *,
        element: ElementStr,
        simulation_settings: GpawIsolatedAtomSimulationSettings,
        runtime_provenance: GpawRuntimeProvenance,
        eigensolver: GpawEigensolverName = GpawEigensolverName.etdm_fdpw,
    ) -> Self:
        """Build the identity after a successful one-element GPAW calculation."""
        if runtime_provenance.elements != frozenset({element}):
            msg = "An isolated-atom registry identity requires runtime provenance for its element only."
            raise ValueError(msg)
        return cls.from_simulation_settings_and_paw_data(
            simulation_settings=simulation_settings,
            paw_data=runtime_provenance.atomic_data_by_element[element],
            gpaw_version=runtime_provenance.gpaw_version,
            eigensolver=eigensolver,
        )

    @classmethod
    def from_simulation_settings_and_paw_data(
        cls,
        *,
        simulation_settings: GpawIsolatedAtomSimulationSettings,
        paw_data: GpawAtomicDataInfo,
        gpaw_version: str,
        eigensolver: GpawEigensolverName = GpawEigensolverName.etdm_fdpw,
    ) -> Self:
        """Build an atom identity from label-resolved PAW data."""
        return cls(
            element=paw_data.element_symbol,
            eigensolver=eigensolver,
            gpaw_version=gpaw_version,
            recipe_fingerprint=cls.recipe_fingerprint_for_simulation_settings(
                simulation_settings
            ),
            runtime_fingerprint=cls.runtime_fingerprint_for_paw_data(paw_data),
        )


class GpawIsolatedAtomRegistryEntry(MSONableBaseModel):
    """One successful GPAW isolated-atom calculation available for reuse."""

    model_config = ConfigDict(extra="forbid")

    identity: GpawIsolatedAtomRegistryIdentity
    simulation_settings: GpawIsolatedAtomSimulationSettings
    energy: float
    magnetic_moment: float
    eigensolver: GpawEigensolverName
    runtime_provenance: GpawRuntimeProvenance
    source_job_uuid: str = Field(min_length=1)

    @property
    def element(self) -> ElementStr:
        """Return the atom whose energy this entry provides."""
        return self.identity.element

    def matches_scientific_result(self, other: Self) -> bool:
        """Whether entries agree on reusable results, excluding execution provenance."""
        return (
            self.identity == other.identity
            and self.simulation_settings == other.simulation_settings
            and self.energy == other.energy
            and self.magnetic_moment == other.magnetic_moment
            and self.eigensolver == other.eigensolver
        )

    @model_validator(mode="after")
    def _validate_successful_result_and_identity(self) -> Self:
        if not isfinite(self.energy):
            msg = "An isolated-atom registry entry requires a finite energy."
            raise ValueError(msg)
        if not isfinite(self.magnetic_moment):
            msg = "An isolated-atom registry entry requires a finite magnetic moment."
            raise ValueError(msg)
        expected_identity = GpawIsolatedAtomRegistryIdentity.from_simulation_settings_and_runtime_provenance(
            element=self.element,
            simulation_settings=self.simulation_settings,
            runtime_provenance=self.runtime_provenance,
            eigensolver=self.eigensolver,
        )
        if self.identity != expected_identity:
            msg = "The isolated-atom registry entry identity must match its settings and runtime provenance."
            raise ValueError(msg)
        return self
