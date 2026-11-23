"""MongoDB repository for reusable GPAW isolated-atom energies."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Final

from maggma.stores import MongoStore
from pymongo import ASCENDING, ReturnDocument
from pymongo.collection import Collection as MongoCollection
from pymongo.errors import DuplicateKeyError

from crystal.schemas.gpaw_isolated_atom_registry import (
    GpawIsolatedAtomRegistryEntry,
    GpawIsolatedAtomRegistryIdentity,
)

GPAW_ISOLATED_ATOM_REGISTRY_UNIQUE_INDEX_NAME: Final = (
    "unique_gpaw_isolated_atom_registry_identity"
)
GPAW_ISOLATED_ATOM_REGISTRY_UNIQUE_INDEX_FIELDS: Final = (
    ("identity.registry_schema_version", ASCENDING),
    ("identity.recipe_version", ASCENDING),
    ("identity.recipe_fingerprint", ASCENDING),
    ("identity.element", ASCENDING),
    ("identity.eigensolver", ASCENDING),
    ("identity.gpaw_version", ASCENDING),
    ("identity.runtime_fingerprint", ASCENDING),
)


class GpawIsolatedAtomRegistryConflictError(ValueError):
    """Raised when one registry identity has conflicting scientific results."""


class GpawIsolatedAtomRegistryRepository:
    """Store and retrieve successful GPAW isolated-atom results."""

    def __init__(self, registry_store: MongoStore) -> None:
        """Use the connected MongoDB collection configured in the JobStore."""
        self._collection: MongoCollection[dict[str, Any]] = registry_store._collection  # noqa: SLF001
        self._collection.create_index(
            GPAW_ISOLATED_ATOM_REGISTRY_UNIQUE_INDEX_FIELDS,
            name=GPAW_ISOLATED_ATOM_REGISTRY_UNIQUE_INDEX_NAME,
            unique=True,
        )

    def upsert_success(
        self, entry: GpawIsolatedAtomRegistryEntry
    ) -> GpawIsolatedAtomRegistryEntry:
        """Atomically insert an entry or return its matching first-written result."""
        entry_identity_filter = self._build_registry_identity_filter(entry.identity)
        stored_registry_document: dict[str, Any] | None
        try:
            stored_registry_document = self._collection.find_one_and_update(
                entry_identity_filter,
                {"$setOnInsert": entry.model_dump(mode="json")},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            stored_registry_document = self._collection.find_one(entry_identity_filter)

        if stored_registry_document is None:
            msg = "The GPAW isolated-atom registry did not return an inserted entry."
            raise RuntimeError(msg)

        stored_entry = self._entry_from_document(stored_registry_document)
        if not stored_entry.matches_scientific_result(entry):
            msg = (
                "A GPAW isolated-atom registry entry already exists with conflicting "
                "scientific results."
            )
            raise GpawIsolatedAtomRegistryConflictError(msg)
        return stored_entry

    def get_entries(
        self, registry_identities: Collection[GpawIsolatedAtomRegistryIdentity]
    ) -> dict[str, GpawIsolatedAtomRegistryEntry]:
        """Return entries matching the requested registry identities exactly."""
        registry_identity_filters = [
            self._build_registry_identity_filter(registry_identity)
            for registry_identity in registry_identities
        ]
        if not registry_identity_filters:
            return {}
        registry_documents = self._collection.find({"$or": registry_identity_filters})
        registry_entries = (
            self._entry_from_document(registry_document)
            for registry_document in registry_documents
        )
        return {
            registry_entry.identity.entry_fingerprint: registry_entry
            for registry_entry in registry_entries
        }

    @staticmethod
    def _build_registry_identity_filter(
        registry_identity: GpawIsolatedAtomRegistryIdentity,
    ) -> dict[str, str]:
        return {
            "identity.registry_schema_version": registry_identity.registry_schema_version,
            "identity.recipe_version": registry_identity.recipe_version,
            "identity.recipe_fingerprint": registry_identity.recipe_fingerprint,
            "identity.element": registry_identity.element,
            "identity.eigensolver": registry_identity.eigensolver,
            "identity.gpaw_version": registry_identity.gpaw_version,
            "identity.runtime_fingerprint": registry_identity.runtime_fingerprint,
        }

    @staticmethod
    def _entry_from_document(document: dict[str, Any]) -> GpawIsolatedAtomRegistryEntry:
        document_without_mongo_id = dict(document)
        document_without_mongo_id.pop("_id", None)
        return GpawIsolatedAtomRegistryEntry.model_validate(document_without_mongo_id)
