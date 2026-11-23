"""Shared pytest fixtures for QE-related tests."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest
from ase import Atoms
from fireworks import LaunchPad
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from fireworks.utilities.fw_serializers import load_object_from_file
from jobflow.core.store import JobStore
from maggma.stores import MongoStore

from crystal import SLURM_QADAPTER_YAML
from crystal.constants.gpaw import GPAW_ISOLATED_ATOM_REGISTRY_STORE_NAME
from crystal.constants.structure_labelling import (
    STRUCTURE_LABELLING_TASK_DOCUMENT_BLOB_STORE_NAME,
)
from tests.helpers.test_paths import (
    TEST_CHARGEMOL_ATOMIC_DENSITIES_DIR,
    TEST_FILE_HELPERS_DATA_DIRNAME,
    TEST_PW_CHARGE_DENSITY_CUBE_PATH,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture
def silicon_atoms() -> Atoms:
    """Reusable Si2 structure for workflow tests."""
    return Atoms("Si2", positions=[[0, 0, 0], [0, 0, 2]])


@pytest.fixture(scope="session")
def qe_data_dir() -> Path:
    """Path to test QE data assets (pseudopotentials, configs)."""

    return (Path(__file__).parent / TEST_FILE_HELPERS_DATA_DIRNAME).resolve()


@pytest.fixture(autouse=True, scope="session")
def mongo_uri(request: pytest.FixtureRequest) -> Generator[Any, Any, Any]:
    external = os.environ.get("CRYSTAL_MONGO_URI")
    if external:
        external = external.rstrip("/")
        parsed_uri = urlsplit(external)
        if parsed_uri.scheme and parsed_uri.path not in ("", "/"):
            uri = external
        elif parsed_uri.scheme:
            uri = urlunsplit(
                (
                    parsed_uri.scheme,
                    parsed_uri.netloc,
                    "/test",
                    parsed_uri.query,
                    parsed_uri.fragment,
                )
            )
        else:
            uri = f"{external}/test"
        yield uri
        return

    mongo_proc = request.getfixturevalue("mongo_proc")
    uri = f"mongodb://{mongo_proc.host}:{mongo_proc.port}/test"
    yield uri
    mongo_proc.stop()


@pytest.fixture
def mock_launchpad(mongo_uri: str) -> LaunchPad:
    launchpad = LaunchPad.from_dict(
        {"host": mongo_uri, "name": "fireworks_test", "uri_mode": True}
    )
    launchpad.reset("", require_password=False)
    return launchpad


@pytest.fixture
def gpaw_isolated_atom_registry_store(mongo_uri: str) -> MongoStore:
    """Provide the connected MongoDB store for GPAW atom registry entries."""
    registry_store = MongoStore(
        database="jobflow_test",
        collection_name=GPAW_ISOLATED_ATOM_REGISTRY_STORE_NAME,
        host=mongo_uri,
    )
    registry_store.connect()
    registry_store.remove_docs({})
    return registry_store


@pytest.fixture
def mock_jobstore(
    mongo_uri: str, gpaw_isolated_atom_registry_store: MongoStore
) -> JobStore:
    docs_store = MongoStore(
        database="jobflow_test", collection_name="jobs", host=mongo_uri
    )
    structures_store = MongoStore(
        database="jobflow_test", collection_name="jobs_blobs", host=mongo_uri
    )
    mlip_relax_task_documents_store = MongoStore(
        database="jobflow_test",
        collection_name="mlip_relax_task_documents",
        host=mongo_uri,
    )
    structure_labelling_task_documents_store = MongoStore(
        database="jobflow_test",
        collection_name=STRUCTURE_LABELLING_TASK_DOCUMENT_BLOB_STORE_NAME,
        host=mongo_uri,
    )
    js = JobStore(
        docs_store=docs_store,
        additional_stores={
            "structures": structures_store,
            "mlip_relax_task_documents": mlip_relax_task_documents_store,
            STRUCTURE_LABELLING_TASK_DOCUMENT_BLOB_STORE_NAME: structure_labelling_task_documents_store,
            GPAW_ISOLATED_ATOM_REGISTRY_STORE_NAME: gpaw_isolated_atom_registry_store,
        },
    )
    js.connect(force_reset=True)
    return js


@pytest.fixture
def chargemol_atomic_densities_directory() -> Path:
    """Atomic densities directory for Chargemol job_control fixtures."""
    return TEST_CHARGEMOL_ATOMIC_DENSITIES_DIR


@pytest.fixture
def chargemol_input_cube_path() -> Path:
    """Cube path for Chargemol job_control fixtures."""
    return TEST_PW_CHARGE_DENSITY_CUBE_PATH


@pytest.fixture
def chargemol_core_electron_mapping_for_si_and_o() -> dict[int, int]:
    """Core electron mapping for Chargemol job_control fixtures."""
    return {8: 2, 14: 10}


@pytest.fixture
def default_slurm_queue_adapter() -> CommonAdapter:
    return load_object_from_file(SLURM_QADAPTER_YAML.as_posix())
