# noqa: D100
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import validate_call

DEFAULT_JSON_INDENT = 4


@validate_call
def read_json_to_dict(filepath: Path) -> Any:  # noqa: D103
    with filepath.open("r") as f:
        return json.load(f)


def write_data_to_json(  # noqa: D103
    filepath: Path,
    data: Any,
    mode: Literal["w", "a"] = "w",
    indent: int = DEFAULT_JSON_INDENT,
    encoder: type[json.JSONEncoder] | None = None,
) -> None:
    with filepath.open(mode=mode) as of:
        json.dump(data, of, indent=indent, cls=encoder)

    if not filepath.is_file():
        msg = f"Expected file {filepath} to exist after writing."
        raise FileNotFoundError(msg)
