from contextlib import nullcontext
from pathlib import Path

import pytest
from monty.json import MontyEncoder

from crystal.helpers.dict import count_existing_keys
from crystal.helpers.file import read_file_lines, validate_dir_exists
from crystal.helpers.json import read_json_to_dict, write_data_to_json
from crystal.helpers.temp import run_in_temp_dir
from tests.helpers.test_paths import (
    TEST_EXAMPLE_JSON_FILENAME,
    TEST_EXAMPLE_WRITE_JSON_FILENAME,
    TEST_EXAMPLE_WRITE_JSON_WITH_MONTY_FILENAME,
    TEST_FILE_HELPERS_DATA_DIRNAME,
    TEST_MISSING_TEXT_FILENAME,
    TEST_TEXT_FILENAME,
)

EXAMPLE_JSON = (
    Path(__file__).parent.resolve()
    / TEST_FILE_HELPERS_DATA_DIRNAME
    / TEST_EXAMPLE_JSON_FILENAME
)


def test_read_json_to_dict() -> None:
    data = read_json_to_dict(EXAMPLE_JSON)
    assert data == {"example": "calculation", "number": 1}


@run_in_temp_dir
def test_write_json_to_dict() -> None:
    data = read_json_to_dict(EXAMPLE_JSON)
    file_to_write = Path(TEST_EXAMPLE_WRITE_JSON_FILENAME)
    write_data_to_json(file_to_write, data)
    assert file_to_write.is_file()

    # with MontyEncoder
    file_to_write_with_monty = Path(TEST_EXAMPLE_WRITE_JSON_WITH_MONTY_FILENAME)
    write_data_to_json(file_to_write_with_monty, data, encoder=MontyEncoder)

    data_from_monty_decoder = read_json_to_dict(file_to_write_with_monty)
    assert data_from_monty_decoder == data


def test_count_existing_keys_in_dict() -> None:
    sample_dict = {"a": 1, "b": 2, "c": 3}
    assert count_existing_keys(sample_dict, keys=("a", "b")) == 2
    assert count_existing_keys(sample_dict, keys=("d",)) == 0
    assert count_existing_keys(sample_dict, keys=("a", "a")) == 1


@run_in_temp_dir
def test_read_file_lines() -> None:
    file = Path.cwd() / TEST_TEXT_FILENAME
    file.write_text("line1\nline2\n")
    lines = read_file_lines(file)
    assert lines == ["line1\n", "line2\n"]


@run_in_temp_dir
def test_read_file_lines_raises_for_non_file() -> None:
    not_a_file = Path.cwd() / TEST_MISSING_TEXT_FILENAME
    with pytest.raises(ValueError, match="Path does not point to a file"):
        read_file_lines(not_a_file)


@run_in_temp_dir
def test_validate_dir_exists_passes_for_existing_directory() -> None:
    directory = Path.cwd()

    with nullcontext():
        validate_dir_exists(directory)


@run_in_temp_dir
def test_validate_dir_exists_raises_for_missing_path() -> None:
    missing_directory = Path.cwd() / "missing-directory"

    with pytest.raises(
        ValueError, match=f"Directory {missing_directory.as_posix()} does not exist."
    ):
        validate_dir_exists(missing_directory)


@run_in_temp_dir
def test_validate_dir_exists_raises_for_file_path() -> None:
    file_path = Path.cwd() / TEST_TEXT_FILENAME
    file_path.write_text("line1\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match=f"Directory {file_path.as_posix()} does not exist."
    ):
        validate_dir_exists(file_path)
