# noqa: D100
from __future__ import annotations

import hashlib
import inspect
import json
import types
from collections.abc import Callable, Sequence
from typing import Any

from jobflow.core.job import Job
from jobflow.core.reference import OutputReference
from monty.json import jsanitize


def _normalize_for_deduplication(
    obj: Any, job_uuid_to_dedupe_mapping: dict[str, str] | None = None
) -> Any:
    mapping = job_uuid_to_dedupe_mapping or {}

    if isinstance(obj, OutputReference):
        return {
            "_id": mapping.get(obj.uuid, obj.uuid),
            "attributes": _normalize_for_deduplication(obj.attributes, mapping),
        }

    if isinstance(obj, dict):
        return {
            str(k): _normalize_for_deduplication(v, mapping) for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [_normalize_for_deduplication(v, mapping) for v in obj]

    if isinstance(obj, set):
        return sorted(_normalize_for_deduplication(v, mapping) for v in obj)

    to_dict = getattr(obj, "as_dict", None) or getattr(obj, "to_dict", None)
    if callable(to_dict):
        return _normalize_for_deduplication(to_dict(), mapping)

    if hasattr(obj, "model_dump"):
        return _normalize_for_deduplication(obj.model_dump(mode="python"), mapping)

    return jsanitize(obj, strict=True, allow_bson=True)


def _get_bound_object(job: Job) -> Any | None:
    """For bound methods, job_obj.function is typically a bound method; __self__ is the bound object."""
    bound = getattr(job.function, "__self__", None)
    if bound is None or isinstance(bound, types.ModuleType):
        return None
    return bound


def canonicalize_call(  # noqa: D103
    function: Callable[..., Any], args: Sequence[Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    signature = inspect.signature(function)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def compute_dedupe(
    job: Job,
    job_uuid_to_dedupe_mapping: dict[str, str] | None = None,
    dedupe_property: str = "dedupe_property",
) -> Any:
    """Returns a hash string for the following cases,
    1. If the job is bound to a self object, then looks for dedupe_property. If this is defined uses this for creating the hash with args and kwargs for the method excluding the self argument
    2. If property is not defined or job is not bound to self, then hashes the complete function signature.
    """
    bound = _get_bound_object(job)
    identifier_value = None
    if bound is not None and hasattr(bound, f"{dedupe_property}"):
        dedupe_property_value = getattr(bound, f"{dedupe_property}")
        identifier_value = _normalize_for_deduplication(
            dedupe_property_value, job_uuid_to_dedupe_mapping
        )
    function_value = (
        identifier_value
        if identifier_value is not None
        else _normalize_for_deduplication(
            getattr(job, "function", {}), job_uuid_to_dedupe_mapping
        )
    )
    cannonical_kwargs: dict[str, Any] = canonicalize_call(
        job.function,
        getattr(job, "function_args", []),
        getattr(job, "function_kwargs", {}),
    )

    normalized = _normalize_for_deduplication(
        cannonical_kwargs, job_uuid_to_dedupe_mapping
    )
    payload = {"function": function_value, "kwargs": normalized}
    return hash_object(payload)


def hash_object(obj: Any) -> str:  # noqa: D103
    obj_as_json_str = json.dumps(
        _normalize_for_deduplication(obj), sort_keys=True, default=str
    )
    return hashlib.sha256(obj_as_json_str.encode("utf-8")).hexdigest()
