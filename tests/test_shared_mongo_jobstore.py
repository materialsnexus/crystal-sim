from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from jobflow.core.job import job
from jobflow.managers.fireworks import flow_to_workflow
from maggma.core import DateTimeFormat, Store
from monty.json import MontyDecoder
from pymongo import MongoClient
from yaml import safe_dump

from crystal.helpers.fireworks.connections import connect_to_jobstore
from crystal.repositories.shared_mongo_jobstore import (
    MongoClientFactory,
    SharedGridFSStore,
    SharedMongoJobStore,
    SharedMongoStore,
    write_shared_jobstore_file,
)

TEST_MONGO_URI_HOST_PORT_ERROR = "Mongo URI must include a host and port"


class CountingMongoClient(MongoClient[dict[str, Any]]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def _mongo_endpoint(mongo_uri: str) -> tuple[str, int]:
    parsed_uri = urlsplit(mongo_uri)
    host = parsed_uri.hostname
    port = parsed_uri.port
    if host is None or port is None:
        raise ValueError(TEST_MONGO_URI_HOST_PORT_ERROR)
    return host, port


def _counting_factory(
    clients: list[CountingMongoClient],
) -> Callable[..., CountingMongoClient]:
    def create_client(*args: Any, **kwargs: Any) -> CountingMongoClient:
        client = CountingMongoClient(*args, **kwargs)
        clients.append(client)
        return client

    return create_client


def _build_shared_jobstore(
    *, mongo_uri: str, client_factory: MongoClientFactory = MongoClient
) -> SharedMongoJobStore:
    host, port = _mongo_endpoint(mongo_uri)
    database = f"shared_jobstore_test_{uuid4().hex}"
    return SharedMongoJobStore(
        docs_store=SharedMongoStore(
            database=database, collection_name="outputs", host=host, port=port
        ),
        additional_stores={
            "structures": SharedGridFSStore(
                database=database, collection_name="jobflow_blobs", host=host, port=port
            )
        },
        client_factory=client_factory,
    )


def _write_standard_jobstore_config(
    tmp_path: Path, database: str, mongo_uri: str
) -> Path:
    host, port = _mongo_endpoint(mongo_uri)
    config_path = tmp_path / "jobstore.yaml"
    config_path.write_text(
        safe_dump(
            {
                "docs_store": {
                    "type": "MongoStore",
                    "database": database,
                    "collection_name": "outputs",
                    "host": host,
                    "port": port,
                },
                "additional_stores": {
                    "structures": {
                        "type": "GridFSStore",
                        "database": database,
                        "collection_name": "jobflow_blobs",
                        "host": host,
                        "port": port,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.needs_mongodb
def test_shared_stores_round_trip_and_reset_real_jobflow_artifacts(
    mongo_uri: str,
) -> None:
    clients: list[CountingMongoClient] = []
    jobstore = _build_shared_jobstore(
        mongo_uri=mongo_uri, client_factory=_counting_factory(clients)
    )

    try:
        jobstore.connect()
        jobstore.connect()

        assert len(clients) == 1
        docs_store = jobstore.docs_store
        additional_stores: dict[str, Store] = jobstore.additional_stores
        structures_store = additional_stores["structures"]
        assert isinstance(docs_store, SharedMongoStore)
        assert isinstance(structures_store, SharedGridFSStore)
        assert docs_store.mongo_client is structures_store.mongo_client

        jobstore.connect(force_reset=True)

        assert len(clients) == 2
        assert clients[0].close_calls == 1
        assert clients[1].close_calls == 0
        assert docs_store.mongo_client is structures_store.mongo_client

        jobstore.update(
            {
                "uuid": "shared-client-job",
                "index": 1,
                "output": {"structure": {"lattice": [1, 2, 3]}},
            },
            save={"structures": [["output", "structure"]]},
        )
        stored_output = jobstore.get_output("shared-client-job", load=True)
        assert stored_output["structure"] == {"lattice": [1, 2, 3]}
    finally:
        jobstore.remove_docs({"uuid": "shared-client-job"})
        jobstore.close()

    assert clients[1].close_calls == 1


def test_jobflow_settings_load_shared_jobstore_from_config_file(
    tmp_path: Path, mongo_uri: str
) -> None:
    source_config_path = _write_standard_jobstore_config(
        tmp_path, "worker_loading_test", mongo_uri
    )
    jobstore_path = write_shared_jobstore_file(
        source_config_path, tmp_path / "worker_jobstore.json"
    )
    jobflow_config_path = tmp_path / "jobflow.json"
    jobflow_config_path.write_text(
        json.dumps({"JOB_STORE": jobstore_path.as_posix()}), encoding="utf-8"
    )
    environment = os.environ.copy()
    environment["JOBFLOW_CONFIG_FILE"] = jobflow_config_path.as_posix()

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import jobflow; "
                "from crystal.repositories.shared_mongo_jobstore "
                "import SharedGridFSStore, SharedMongoJobStore, SharedMongoStore; "
                "store = jobflow.SETTINGS.JOB_STORE; "
                "assert isinstance(store, SharedMongoJobStore); "
                "assert isinstance(store.docs_store, SharedMongoStore); "
                "assert isinstance(store.additional_stores['structures'], SharedGridFSStore)"
            ),
        ],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_connect_to_jobstore_uses_shared_store_path(
    tmp_path: Path, mongo_uri: str
) -> None:
    source_config_path = _write_standard_jobstore_config(
        tmp_path, "submission_loading_test", mongo_uri
    )
    clients: list[CountingMongoClient] = []

    jobstore = connect_to_jobstore(
        source_config_path, client_factory=_counting_factory(clients)
    )
    try:
        assert isinstance(jobstore, SharedMongoJobStore)
        assert isinstance(jobstore.docs_store, SharedMongoStore)
        assert isinstance(jobstore.additional_stores["structures"], SharedGridFSStore)
        assert (
            jobstore.docs_store.mongo_client
            is jobstore.additional_stores["structures"].mongo_client
        )
        assert len(clients) == 1
    finally:
        jobstore.close()


def test_fireworks_worker_receives_shared_jobstore_configuration(
    mongo_uri: str,
) -> None:
    @job
    def worker_job() -> int:
        return 1

    host, port = _mongo_endpoint(mongo_uri)
    jobstore = SharedMongoJobStore(
        docs_store=SharedMongoStore(
            database="worker_serialization_test",
            collection_name="outputs",
            host=host,
            port=port,
        ),
        additional_stores={
            "structures": SharedGridFSStore(
                database="worker_serialization_test",
                collection_name="jobflow_blobs",
                host=host,
                port=port,
            )
        },
    )

    workflow = flow_to_workflow(worker_job(), store=jobstore)
    task_dict = workflow.fws[0].tasks[0].to_dict()
    decoded_task = MontyDecoder().process_decoded(task_dict)
    worker_jobstore = decoded_task["store"]

    assert isinstance(worker_jobstore, SharedMongoJobStore)
    assert isinstance(worker_jobstore.docs_store, SharedMongoStore)
    assert isinstance(
        worker_jobstore.additional_stores["structures"], SharedGridFSStore
    )


def test_store_serialization_preserves_behavioral_configuration(mongo_uri: str) -> None:
    host, port = _mongo_endpoint(mongo_uri)
    mongo_store = SharedMongoStore(
        database="serialization_options_test",
        collection_name="outputs",
        host=host,
        port=port,
        safe_update=True,
        default_sort={"modified_at": -1},
        key="record_id",
        last_updated_field="modified_at",
        last_updated_type="isoformat",
        mongoclient_kwargs={"maxPoolSize": 3},
    )
    gridfs_store = SharedGridFSStore(
        database="serialization_options_test",
        collection_name="blobs",
        host=host,
        port=port,
        compression=True,
        ensure_metadata=True,
        searchable_fields=["job_uuid"],
        key="blob_id",
    )

    restored_mongo_store = MontyDecoder().process_decoded(mongo_store.as_dict())
    restored_gridfs_store = MontyDecoder().process_decoded(gridfs_store.as_dict())

    assert isinstance(restored_mongo_store, SharedMongoStore)
    assert restored_mongo_store.safe_update is True
    assert restored_mongo_store.default_sort == {"modified_at": -1}
    assert restored_mongo_store.key == "record_id"
    assert restored_mongo_store.last_updated_field == "modified_at"
    assert (
        DateTimeFormat(restored_mongo_store.last_updated_type)
        == DateTimeFormat.IsoFormat
    )
    assert restored_mongo_store.mongoclient_kwargs == {"maxPoolSize": 3}
    assert isinstance(restored_gridfs_store, SharedGridFSStore)
    assert restored_gridfs_store.compression is True
    assert restored_gridfs_store.ensure_metadata is True
    assert restored_gridfs_store.searchable_fields == ["job_uuid"]
    assert restored_gridfs_store.key == "blob_id"
