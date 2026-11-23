# noqa: D100
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def read_file_lines(  # noqa: D103
    file: Path, open_func: Callable[[Path], IO[str]] = open
) -> list[str]:
    if not file.is_file():
        msg = "Path does not point to a file."
        raise ValueError(msg)
    with open_func(file) as f:
        return f.readlines()


def validate_dir_exists(path: Path) -> None:
    """Validate that a path exists and points to a directory."""
    if not path.is_dir():
        error_msg = f"Directory {path.as_posix()} does not exist."
        raise ValueError(error_msg)


def count_args_not_none(*args: Any) -> int:
    """Counts the number of non-None arguments."""
    return sum(arg is not None for arg in args)
