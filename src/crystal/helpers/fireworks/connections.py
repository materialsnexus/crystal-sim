"""Helpers for connecting to FireWorks and Jobflow stores."""

from __future__ import annotations

from pathlib import Path

from fireworks import LaunchPad
from pymongo import MongoClient

from crystal import LOCAL_JOBSTORE_YAML, LOCAL_LAUNCHPAD_YAML
from crystal.repositories.shared_mongo_jobstore import (
    MongoClientFactory,
    SharedMongoJobStore,
)


def validate_config_file_exists(config_file: Path) -> Path:
    """Validate that a FireWorks/Jobflow config file exists on disk."""
    if config_file.is_file():
        return config_file
    error_message = f"{config_file.as_posix()} does not exist!"
    raise FileNotFoundError(error_message)


def run_launchpad(config_file: Path = LOCAL_LAUNCHPAD_YAML) -> LaunchPad:
    """Build a LaunchPad from a FireWorks YAML config file."""
    validate_config_file_exists(config_file)
    return LaunchPad.from_file(config_file.as_posix())


def connect_to_jobstore(
    config_file: Path = LOCAL_JOBSTORE_YAML,
    *,
    client_factory: MongoClientFactory = MongoClient,
) -> SharedMongoJobStore:
    """Build and connect a JobStore from a Jobflow YAML config file."""
    validate_config_file_exists(config_file)
    job_store = SharedMongoJobStore.from_file(
        config_file, client_factory=client_factory
    )
    job_store.connect()
    return job_store
