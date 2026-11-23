# noqa: D100
from pydantic import BaseModel, ConfigDict


class Properties(BaseModel):
    """Base class to keep all the calculated properties from a job."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str
