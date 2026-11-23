from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from jobflow.core.job import Response
from jobflow.core.store import JobStore
from maggma.stores import MemoryStore

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME

OutputT = TypeVar("OutputT")


def extract_outputs_of_type_from_run(
    run_flow: Mapping[str, Mapping[int, Response[object]]], output_type: type[OutputT]
) -> list[OutputT]:
    outputs: list[OutputT] = []
    for responses_by_index in run_flow.values():
        outputs.extend(
            response.output
            for response in responses_by_index.values()
            if isinstance(response.output, output_type)
        )
    return outputs


def make_blob_jobstore(
    *, additional_store_name: str = STRUCTURE_BLOB_STORE_NAME
) -> tuple[JobStore, MemoryStore]:
    blob_store = MemoryStore()
    jobstore = JobStore(
        docs_store=MemoryStore(), additional_stores={additional_store_name: blob_store}
    )
    jobstore.connect()
    return jobstore, blob_store
