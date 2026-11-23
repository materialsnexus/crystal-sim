"""Shared MongoDB clients for Jobflow stores."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import gridfs
from jobflow.core.store import JobStore
from maggma.core import DateTimeFormat, Sort, Store
from maggma.stores import GridFSStore, MongoStore
from maggma.stores.ssh_tunnel import SSHTunnel
from monty.serialization import dumpfn, loadfn
from pymongo import MongoClient

MongoClientFactory: TypeAlias = Callable[..., MongoClient[dict[str, Any]]]
FrozenValue: TypeAlias = Hashable
SHARED_MONGO_NOT_CONNECTED_ERROR = "The shared Mongo store is not connected"
SHARED_MONGO_SSH_TUNNEL_ERROR = "Shared Mongo stores do not support SSH tunnels"
SHARED_MONGO_STORE_LIFECYCLE_ERROR = (
    "Shared Mongo stores must be managed by SharedMongoJobStore"
)
SHARED_JOBSTORE_FILE_FORMAT_ERROR = "Unrecognised database file format."


class _MongoConnectionStore(Protocol):
    """Minimal Mongo-store connection attributes consumed by this module."""

    host: str
    port: int
    username: str
    password: str
    auth_source: str
    mongoclient_kwargs: dict[str, Any]


class _SerializableMongoStore(_MongoConnectionStore, Protocol):
    """Store attributes required to serialize a Mongo-backed store."""

    database: str
    collection_name: str
    ssh_tunnel: SSHTunnel | None
    kwargs: dict[str, Any]
    key: str
    last_updated_field: str

    @property
    def last_updated_type(self) -> DateTimeFormat | str: ...


def _freeze_value(value: Any) -> FrozenValue:
    """Convert PyMongo option values into a hashable client-key value.

    ``mongoclient_kwargs`` is intentionally typed as ``Any`` because PyMongo
    accepts many option-specific value types. Known containers are frozen
    recursively; unsupported values use ``repr`` so they cannot make the client
    key unhashable.
    """
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((str(key), _freeze_value(item)) for key, item in value.items()),
                key=lambda item: item[0],
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    return repr(value)


@dataclass(frozen=True)
class _MongoClientKey:
    host: str
    port: int
    username: str
    password: str
    auth_source: str | None
    mongoclient_kwargs: FrozenValue


def _client_key(store: _MongoConnectionStore) -> _MongoClientKey:
    return _MongoClientKey(
        host=store.host,
        port=store.port,
        username=store.username,
        password=store.password,
        auth_source=store.auth_source,
        mongoclient_kwargs=_freeze_value(store.mongoclient_kwargs),
    )


class _SharedClientStore:
    host: str
    port: int
    username: str
    password: str
    auth_source: str
    mongoclient_kwargs: dict[str, Any]

    def _initialize_shared_client(self) -> None:
        self._client: MongoClient[dict[str, Any]] | None = None

    @property
    def mongo_client(self) -> MongoClient[dict[str, Any]]:
        """Return the client bound by the owning JobStore."""
        if self._client is None:
            raise RuntimeError(SHARED_MONGO_NOT_CONNECTED_ERROR)
        return self._client

    def bind_client(self, client: MongoClient[dict[str, Any]]) -> None:
        """Bind a client and initialize this store's collection handles."""
        if getattr(self, "ssh_tunnel", None) is not None:
            raise ValueError(SHARED_MONGO_SSH_TUNNEL_ERROR)
        self._client = client
        try:
            self._bind_client(client)
        except Exception:
            self._client = None
            raise

    def _bind_client(self, client: MongoClient[dict[str, Any]]) -> None:
        pass

    def unbind_client(self) -> None:
        """Clear collection handles without closing the bound client."""
        self._unbind_client()
        self._client = None

    def _unbind_client(self) -> None:
        pass

    def connect(self, force_reset: bool = False) -> None:  # noqa: ARG002, FBT001, FBT002
        """Reject lifecycle management outside a shared JobStore."""
        raise RuntimeError(SHARED_MONGO_STORE_LIFECYCLE_ERROR)

    def close(self) -> None:
        """Clear this store's handles without closing the shared client."""
        self.unbind_client()


class SharedMongoStore(_SharedClientStore, MongoStore):
    """MongoStore that borrows a process-local shared MongoClient."""

    def __init__(
        self,
        database: str,
        collection_name: str,
        host: str,
        port: int,
        username: str = "",
        password: str = "",
        ssh_tunnel: SSHTunnel | None = None,
        safe_update: bool = False,  # noqa: FBT001, FBT002
        auth_source: str | None = None,
        mongoclient_kwargs: dict[str, Any] | None = None,
        default_sort: dict[str, Sort | int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a shared Mongo store.

        ``kwargs`` contains additional Maggma store options, while
        ``mongoclient_kwargs`` contains keyword options forwarded to PyMongo.
        ``ssh_tunnel`` retains Maggma's constructor shape but shared clients do
        not support tunnel-owned connection lifecycles.
        """
        super().__init__(
            database=database,
            collection_name=collection_name,
            host=host,
            port=port,
            username=username,
            password=password,
            ssh_tunnel=ssh_tunnel,
            safe_update=safe_update,
            auth_source=auth_source,
            mongoclient_kwargs=mongoclient_kwargs,
            default_sort=default_sort,
            **kwargs,
        )
        self._initialize_shared_client()
        self._coll: Any = None

    def _bind_client(self, client: MongoClient[dict[str, Any]]) -> None:
        self._coll = client[self.database][self.collection_name]

    def _unbind_client(self) -> None:
        self._coll = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize store configuration without runtime client state."""
        return _mongo_store_spec(self, type(self).__name__)


class SharedGridFSStore(_SharedClientStore, GridFSStore):
    """GridFSStore that borrows a process-local shared MongoClient."""

    def __init__(
        self,
        database: str,
        collection_name: str,
        host: str,
        port: int,
        username: str = "",
        password: str = "",
        compression: bool = False,  # noqa: FBT001, FBT002
        ensure_metadata: bool = False,  # noqa: FBT001, FBT002
        searchable_fields: list[str] | None = None,
        auth_source: str | None = None,
        mongoclient_kwargs: dict[str, Any] | None = None,
        ssh_tunnel: SSHTunnel | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a shared GridFS store.

        ``kwargs`` contains additional Maggma store options, and
        ``mongoclient_kwargs`` contains keyword options forwarded to PyMongo.
        The GridFS file and chunk collections are handles on the same shared
        client; the store does not create or close another client for them.
        """
        super().__init__(
            database=database,
            collection_name=collection_name,
            host=host,
            port=port,
            username=username,
            password=password,
            compression=compression,
            ensure_metadata=ensure_metadata,
            searchable_fields=searchable_fields,
            auth_source=auth_source,
            mongoclient_kwargs=mongoclient_kwargs,
            ssh_tunnel=ssh_tunnel,
            **kwargs,
        )
        self._initialize_shared_client()
        self._coll: Any = None
        self._fs: Any = None
        self._files_collection: Any = None
        self._chunks_collection: Any = None

    def _bind_client(self, client: MongoClient[dict[str, Any]]) -> None:
        database = client[self.database]
        self._coll = gridfs.GridFS(database, self.collection_name)
        self._files_collection = database[f"{self.collection_name}.files"]
        files_store: MongoStore = MongoStore.from_collection(  # type: ignore[no-untyped-call]
            self._files_collection
        )
        files_store.last_updated_field = f"metadata.{self.last_updated_field}"
        files_store.key = self.key
        self._fs = files_store
        self._chunks_collection = database[f"{self.collection_name}.chunks"]

    def _unbind_client(self) -> None:
        self._fs = None
        self._coll = None
        self._files_collection = None
        self._chunks_collection = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize GridFS configuration without runtime collection state."""
        spec = _mongo_store_spec(self, type(self).__name__)
        spec.update(
            {
                "compression": self.compression,
                "ensure_metadata": self.ensure_metadata,
                "searchable_fields": list(self.searchable_fields),
                "key": self.key,
            }
        )
        return spec


class SharedMongoJobStore(JobStore):
    """JobStore whose Mongo-backed stores share process-local clients."""

    docs_store: Store
    additional_stores: dict[str, Store]

    def __init__(
        self,
        docs_store: Store,
        additional_stores: dict[str, Store] | None = None,
        save: Any = None,
        load: Any = False,  # noqa: FBT002
        client_factory: MongoClientFactory = MongoClient,
    ) -> None:
        """Create a JobStore that owns clients for all shared Mongo stores."""
        docs_store = _transform_store_value(docs_store)
        if additional_stores is not None:
            additional_stores = {
                name: _transform_store_value(store)
                for name, store in additional_stores.items()
            }
        super().__init__(
            docs_store=docs_store,
            additional_stores=additional_stores or {},
            save=save,
            load=load,
        )
        self._client_factory = client_factory
        self._clients: dict[_MongoClientKey, MongoClient[dict[str, Any]]] = {}
        self._connected = False

    def _stores(self) -> list[Store]:
        return [self.docs_store, *self.additional_stores.values()]

    def _bind_store(self, store: _SharedClientStore) -> None:
        key = _client_key(store)
        client = self._clients.get(key)
        if client is None:
            client = self._create_client(store)
            self._clients[key] = client
        store.bind_client(client)

    def _create_client(self, store: _SharedClientStore) -> MongoClient[dict[str, Any]]:
        if store.username:
            return self._client_factory(
                host=store.host,
                port=store.port,
                username=store.username,
                password=store.password,
                authSource=store.auth_source,
                **store.mongoclient_kwargs,
            )
        return self._client_factory(store.host, store.port, **store.mongoclient_kwargs)

    def connect(self, force_reset: bool = False) -> None:  # noqa: FBT001, FBT002
        """Create shared clients and bind them to all child stores."""
        if self._connected and not force_reset:
            return
        if self._connected or self._clients:
            self.close()
        try:
            for store in self._stores():
                if isinstance(store, _SharedClientStore):
                    self._bind_store(store)
                else:
                    store.connect(force_reset=force_reset)
            self._connected = True
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Unbind child stores and close all clients owned by this JobStore."""
        try:
            for store in self._stores():
                store.close()  # type: ignore[no-untyped-call]
        finally:
            for client in self._clients.values():
                client.close()
            self._clients.clear()
            self._connected = False

    def _set_client_factory(self, client_factory: MongoClientFactory) -> None:
        self.close()
        self._client_factory = client_factory

    def as_dict(self) -> dict[str, Any]:
        """Serialize configuration without the runtime client factory."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "docs_store": self.docs_store.as_dict(),
            "additional_stores": {
                name: store.as_dict() for name, store in self.additional_stores.items()
            },
            "save": self.save,
            "load": self.load,
        }

    @classmethod
    def from_file(cls, db_file: str | Path, **kwargs: Any) -> SharedMongoJobStore:
        """Load standard Jobflow config as a shared Mongo JobStore.

        The input may be a normal Jobflow YAML specification or a Monty-serialized
        shared JobStore. Extra keyword arguments are forwarded to JobStore loading.
        """
        store_info = loadfn(db_file)
        if isinstance(store_info, cls):
            client_factory = kwargs.pop("client_factory", MongoClient)
            store_info._set_client_factory(client_factory)  # noqa: SLF001
            return store_info
        if isinstance(store_info, JobStore):
            store_info = store_info.as_dict()
        if not isinstance(store_info, dict):
            raise TypeError(SHARED_JOBSTORE_FILE_FORMAT_ERROR)
        return cls.from_dict_spec(store_info, **kwargs)

    @classmethod
    def from_dict_spec(cls, spec: dict[str, Any], **kwargs: Any) -> SharedMongoJobStore:
        """Construct a shared JobStore from a Jobflow dictionary specification.

        MongoStore and GridFSStore entries are replaced with their shared-client
        adapters. Other additional stores remain governed by Jobflow/Maggma.
        """
        client_factory = kwargs.pop("client_factory", MongoClient)
        transformed_spec = _transform_store_spec(spec)
        docs_store = transformed_spec.get("docs_store")
        if isinstance(docs_store, Store):
            additional_stores = dict(transformed_spec.get("additional_stores", {}))
            load = transformed_spec.get("load", False)
            jobstore = cls(
                docs_store=docs_store,
                additional_stores=additional_stores,
                save=transformed_spec.get("save"),
                load=load,
                client_factory=client_factory,
                **kwargs,
            )
        else:
            jobstore = super().from_dict_spec(
                transformed_spec, client_factory=client_factory, **kwargs
            )
        return jobstore

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SharedMongoJobStore:
        """Deserialize a shared JobStore from Monty data."""
        return cls.from_dict_spec(d)


def _transform_store_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Replace supported Mongo store specs with shared adapter type names."""
    transformed = dict(spec)
    if "docs_store" in transformed:
        transformed["docs_store"] = _transform_store_value(transformed["docs_store"])
    if "additional_stores" in transformed:
        transformed["additional_stores"] = {
            name: _transform_store_value(store_spec)
            for name, store_spec in transformed["additional_stores"].items()
        }
    return transformed


def _transform_store_value(store_spec: Any) -> Any:
    """Transform one decoded store object or one plain store specification."""
    if isinstance(store_spec, Store):
        if isinstance(store_spec, SharedMongoStore | SharedGridFSStore):
            return store_spec
        if isinstance(store_spec, MongoStore):
            return SharedMongoStore(**_mongo_store_constructor_arguments(store_spec))
        if isinstance(store_spec, GridFSStore):
            return SharedGridFSStore(**_gridfs_store_constructor_arguments(store_spec))
        return store_spec
    if not isinstance(store_spec, dict):
        return store_spec
    transformed = {
        key: _transform_store_value(value)
        for key, value in store_spec.items()
        if not key.startswith("@")
    }
    store_type = transformed.get("type") or store_spec.get("@class")
    store_type_map = {
        "MongoStore": "SharedMongoStore",
        "GridFSStore": "SharedGridFSStore",
        "SharedMongoStore": "SharedMongoStore",
        "SharedGridFSStore": "SharedGridFSStore",
    }
    if store_type in store_type_map:
        transformed["type"] = store_type_map[store_type]
    return transformed


def _mongo_store_constructor_arguments(store: MongoStore) -> dict[str, Any]:
    """Extract configuration needed to recreate a MongoStore adapter."""
    return {
        "database": store.database,
        "collection_name": store.collection_name,
        "host": store.host,
        "port": store.port,
        "username": store.username,
        "password": store.password,
        "ssh_tunnel": store.ssh_tunnel,
        "safe_update": store.safe_update,
        "auth_source": store.auth_source,
        "mongoclient_kwargs": store.mongoclient_kwargs,
        **store.kwargs,
        "default_sort": getattr(store, "default_sort", None),
        "key": store.key,
        "last_updated_field": store.last_updated_field,
        "last_updated_type": store.last_updated_type,
    }


def _gridfs_store_constructor_arguments(store: GridFSStore) -> dict[str, Any]:
    """Extract configuration needed to recreate a GridFSStore adapter."""
    return {
        "database": store.database,
        "collection_name": store.collection_name,
        "host": store.host,
        "port": store.port,
        "username": store.username,
        "password": store.password,
        "compression": store.compression,
        "ensure_metadata": store.ensure_metadata,
        "searchable_fields": list(store.searchable_fields),
        "auth_source": store.auth_source,
        "mongoclient_kwargs": store.mongoclient_kwargs,
        "ssh_tunnel": store.ssh_tunnel,
        **store.kwargs,
        "key": store.key,
        "last_updated_field": store.last_updated_field,
        "last_updated_type": store.last_updated_type,
    }


def _mongo_store_spec(
    store: _SerializableMongoStore, class_name: str
) -> dict[str, Any]:
    """Build Monty data without serializing live clients or collection handles."""
    spec = {
        "@module": "crystal.repositories.shared_mongo_jobstore",
        "@class": class_name,
        "database": store.database,
        "collection_name": store.collection_name,
        "host": store.host,
        "port": store.port,
        "username": store.username,
        "password": store.password,
        "ssh_tunnel": store.ssh_tunnel,
        "auth_source": store.auth_source,
        "mongoclient_kwargs": store.mongoclient_kwargs,
        **store.kwargs,
        "key": store.key,
        "last_updated_field": store.last_updated_field,
        "last_updated_type": DateTimeFormat(store.last_updated_type).value,
    }
    if class_name == "SharedMongoStore":
        spec["default_sort"] = getattr(store, "default_sort", None)
        spec["safe_update"] = getattr(store, "safe_update", False)
    return spec


def write_shared_jobstore_file(
    source_config_file: str | Path, destination_file: str | Path
) -> Path:
    """Convert a normal Jobflow store config to a worker-loadable shared file.

    The source may be the existing Jobflow YAML format. The destination is a Monty
    serialized ``SharedMongoJobStore`` suitable for ``JOB_STORE`` in a Jobflow
    configuration file. It should normally be generated in a site-specific excluded
    ``block_...`` directory.
    """
    shared_jobstore = SharedMongoJobStore.from_file(source_config_file)
    destination_path = Path(destination_file)
    dumpfn(shared_jobstore, destination_path)
    return destination_path
