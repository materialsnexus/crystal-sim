# noqa: D100
import functools
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def run_in_temp_dir(func: F) -> Any:  # noqa: D103
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        original_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                result = func(*args, **kwargs)
            finally:
                os.chdir(str(original_dir))
            return result

    return wrapper
