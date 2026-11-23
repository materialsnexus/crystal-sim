# noqa: D100
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

from jobflow.core.maker import Maker
from pydantic import BaseModel, Field, model_validator

from crystal.helpers.fireworks.slurm_settings import SlurmSettingsGenerator
from crystal.schemas.fireworks import SlurmSettings
from crystal.schemas.maker_names import MakerName
from crystal.schemas.properties_union import PropertiesDiscriminatedUnion
from crystal.schemas.serialization import (
    deserialize_model_from_msonable,
    serialize_model_as_msonable,
)
from crystal.schemas.settings import ExecutorSettings, SimulationSettings


@dataclass
class BaseMaker(Maker):  # noqa: D101
    name: MakerName = MakerName.base
    simulation_settings: SimulationSettings | None = None
    executor_settings: ExecutorSettings = field(default_factory=ExecutorSettings)

    def __post_init__(self) -> None:  # noqa: D105
        if self.simulation_settings is None:
            msg = "simulation settings cannot be None"
            raise ValueError(msg)

    @property
    def slurm_settings(self) -> SlurmSettings:  # noqa: D102
        return SlurmSettingsGenerator(executor_settings=self.executor_settings).settings


class BaseTaskDocument(BaseModel):  # noqa: D101
    dir_name: str | None = Field(
        description="The directory for this task", default=None
    )
    errors: str = Field(description="Error log", default="")
    properties: PropertiesDiscriminatedUnion | None = Field(
        description="Computed properties from a task"
    )

    @model_validator(mode="after")
    def _validate_properties_or_errors_present(self) -> Self:
        if self.errors != "" and not self.errors.strip():
            msg = "Error message cannot be whitespace only."
            raise ValueError(msg)

        has_properties = self.properties is not None
        has_errors = self.errors != ""
        if has_properties == has_errors:
            msg = "Task document must include either properties or a non-empty error message."
            raise ValueError(msg)
        return self


class MSONableModelMixin:
    """Serialize MLIP task documents for Jobflow blob storage.

    Jobflow serializes output-schema classes before running a job. If
    BaseTaskDocument exposed ``as_dict``, Jobflow would call that instance method
    on every output-schema class and raise TypeError. This mixin confines
    ``as_dict`` to concrete MLIP task-document classes, which Jobflow stores in blobs.

    It must precede MSONable in class inheritance so this class's Pydantic schema
    hook overrides MSONable's schema hook and preserves the task-document schema.
    """

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Callable[[Any], dict[str, Any]]
    ) -> dict[str, Any]:
        """Use Pydantic's normal schema generation for this MSONable model."""
        return handler(core_schema)

    def as_dict(self: BaseModel) -> dict[str, Any]:
        """Return a Monty-JSON representation of the model."""
        return serialize_model_as_msonable(self)

    @classmethod
    def from_dict(cls: type[Self], d: dict[str, Any]) -> Self:
        """Construct a model from a Monty-JSON representation."""
        return deserialize_model_from_msonable(  # type: ignore[type-var]
            cls, d
        )
