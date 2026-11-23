"""Tests for pseudopotential library helpers."""

from pathlib import Path

import pytest

from crystal.helpers.qe.pseudopotentials.library import PseudopotentialLibrary
from tests.helpers.test_paths import (
    TEST_FILE_HELPERS_DATA_DIRNAME,
    TEST_MISSING_PSEUDOPOTENTIAL_FILENAME,
    TEST_SI_PSEUDOPOTENTIAL_FILENAME,
)

DATA_DIR = Path(__file__).parent / TEST_FILE_HELPERS_DATA_DIRNAME


def test_library_loads_metadata_case_insensitively() -> None:
    """Library should load files regardless of element case in config."""

    library = PseudopotentialLibrary(pseudo_dir=DATA_DIR)
    files = {"si": TEST_SI_PSEUDOPOTENTIAL_FILENAME}
    mapping = library.load_metadata(files)
    assert mapping["Si"].filename == TEST_SI_PSEUDOPOTENTIAL_FILENAME
    assert mapping["Si"].functional == "pbe"


def test_library_reports_missing_files() -> None:
    """Missing files raise ValueError with the offending element."""

    library = PseudopotentialLibrary(pseudo_dir=DATA_DIR)
    with pytest.raises(ValueError, match="Missing pseudopotential file"):
        library.load_metadata({"Si": TEST_MISSING_PSEUDOPOTENTIAL_FILENAME})


def test_library_requires_all_elements() -> None:
    """If some elements are missing from the mapping, raise ValueError."""

    library = PseudopotentialLibrary(pseudo_dir=DATA_DIR)
    with pytest.raises(ValueError, match="Missing pseudopotentials for"):
        library.validate_required_elements(
            {"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME}, {"Si", "O"}
        )
