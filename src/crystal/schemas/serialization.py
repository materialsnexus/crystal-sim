"""Shared serialization utilities for schema models."""

from __future__ import annotations

import json
from typing import Any, Literal, Self, TypeVar

from monty.json import MontyDecoder, MontyEncoder
from pydantic import BaseModel, ConfigDict

MSONABLE_JSON_MODE: Literal["json"] = "json"
ModelT = TypeVar("ModelT", bound=BaseModel)


def serialize_model_as_msonable(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model with Monty class metadata."""
    python_dump = model.model_dump(mode="python")
    decoded_object = json.loads(json.dumps(python_dump, cls=MontyEncoder))
    if not isinstance(decoded_object, dict):
        error_msg = "Serialized model payload must be a dictionary."
        raise TypeError(error_msg)
    decoded_object.update(
        {"@module": model.__class__.__module__, "@class": model.__class__.__name__}
    )
    return decoded_object


def deserialize_model_from_msonable(
    model_type: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Deserialize a Pydantic model from a Monty representation."""
    model_payload = dict(payload)
    model_payload.pop("@module", None)
    model_payload.pop("@class", None)
    decoded = MontyDecoder().process_decoded(model_payload)
    if not isinstance(decoded, dict):
        error_msg = "Decoded MSON payload must be a dictionary."
        raise TypeError(error_msg)
    return model_type.model_validate(decoded)


class MSONableBaseModel(BaseModel):  # noqa: D101
    model_config = ConfigDict(extra="ignore")

    def as_dict(self) -> dict[str, Any]:  # noqa: D102
        return serialize_model_as_msonable(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:  # noqa: D102
        return deserialize_model_from_msonable(cls, d)
