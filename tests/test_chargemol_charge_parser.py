from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from crystal.helpers.chargemol.charge_parser import (
    CHARGEMOL_NET_ATOMIC_CHARGES_PROPERTY,
    ChargemolNetAtomicCharges,
    ChargemolNetAtomicChargesParser,
    attach_chargemol_net_atomic_charges_to_structure,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.test_data_paths import resolve_tests_data_dir
from tests.helpers.test_paths import (
    TEST_CHARGEMOL_DATA_DIRNAME,
    TEST_DDEC6_INCOMPLETE_ATOM_BLOCK_FILENAME,
    TEST_DDEC6_INCOMPLETE_ATOM_LINE_FILENAME,
    TEST_DDEC6_INVALID_CHARGE_FILENAME,
    TEST_DDEC6_INVALID_HEADER_FILENAME,
    TEST_DDEC6_MISMATCHED_SYMBOL_FILENAME,
    TEST_DDEC6_MISSING_FILENAME,
    TEST_DDEC6_VALID_FILENAME,
)


def _chargemol_test_data_dir() -> Path:
    return resolve_tests_data_dir(Path(__file__)) / TEST_CHARGEMOL_DATA_DIRNAME


def _chargemol_fixture_path(fixture_name: Path) -> Path:
    return _chargemol_test_data_dir() / fixture_name


def test_parses_net_atomic_charges_file_successfully() -> None:
    source_path = _chargemol_fixture_path(TEST_DDEC6_VALID_FILENAME)
    result = ChargemolNetAtomicChargesParser(
        chargemol_net_atomic_charges_path=source_path
    ).get_chargemol_atomic_charges()
    assert isinstance(result, ChargemolNetAtomicCharges)
    assert result.atom_count == 2
    assert result.atom_symbols == ["Si", "Si"]
    assert result.net_atomic_charges == [0.1, -0.1]
    assert result.total_charge == pytest.approx(0.0)
    assert result.charge_records[0].atom_index == 1
    assert result.charge_records[0].atom_symbol == "Si"
    assert result.charge_records[0].net_atomic_charge == pytest.approx(0.1)


def test_missing_net_atomic_charges_file_raises(tmp_path: Path) -> None:
    with pytest.raises(
        ValidationError, match="Missing Chargemol DDEC6 net atomic charges file"
    ):
        ChargemolNetAtomicChargesParser(
            chargemol_net_atomic_charges_path=tmp_path / TEST_DDEC6_MISSING_FILENAME
        ).get_chargemol_atomic_charges()


def test_invalid_header_raises_value_error() -> None:
    net_atomic_charges_path = _chargemol_fixture_path(
        TEST_DDEC6_INVALID_HEADER_FILENAME
    )
    with pytest.raises(ValueError, match="atom count header"):
        ChargemolNetAtomicChargesParser(
            chargemol_net_atomic_charges_path=net_atomic_charges_path
        ).get_chargemol_atomic_charges()


def test_incomplete_atom_block_raises_value_error() -> None:
    net_atomic_charges_path = _chargemol_fixture_path(
        TEST_DDEC6_INCOMPLETE_ATOM_BLOCK_FILENAME
    )
    with pytest.raises(ValueError, match="missing atom entries"):
        ChargemolNetAtomicChargesParser(
            chargemol_net_atomic_charges_path=net_atomic_charges_path
        ).get_chargemol_atomic_charges()


def test_incomplete_atom_line_raises_value_error() -> None:
    net_atomic_charges_path = _chargemol_fixture_path(
        TEST_DDEC6_INCOMPLETE_ATOM_LINE_FILENAME
    )
    with pytest.raises(ValueError, match="entry 2 is incomplete"):
        ChargemolNetAtomicChargesParser(
            chargemol_net_atomic_charges_path=net_atomic_charges_path
        ).get_chargemol_atomic_charges()


def test_invalid_charge_value_raises_value_error() -> None:
    net_atomic_charges_path = _chargemol_fixture_path(
        TEST_DDEC6_INVALID_CHARGE_FILENAME
    )
    with pytest.raises(ValueError, match="charge value missing for atom 1"):
        ChargemolNetAtomicChargesParser(
            chargemol_net_atomic_charges_path=net_atomic_charges_path
        ).get_chargemol_atomic_charges()


def test_attach_charges_validates_structure_symbols() -> None:
    net_atomic_charges_path = _chargemol_fixture_path(
        TEST_DDEC6_MISMATCHED_SYMBOL_FILENAME
    )
    charges = ChargemolNetAtomicChargesParser(
        chargemol_net_atomic_charges_path=net_atomic_charges_path
    ).get_chargemol_atomic_charges()
    with pytest.raises(
        ValueError, match="Structure species do not match DDEC6 atom symbols"
    ):
        attach_chargemol_net_atomic_charges_to_structure(
            structure=TEST_SI_STRUCTURE, net_atomic_charges=charges
        )


def test_attach_charges_adds_site_properties() -> None:
    net_atomic_charges_path = _chargemol_fixture_path(TEST_DDEC6_VALID_FILENAME)
    charges = ChargemolNetAtomicChargesParser(
        chargemol_net_atomic_charges_path=net_atomic_charges_path
    ).get_chargemol_atomic_charges()
    structure_with_charges = attach_chargemol_net_atomic_charges_to_structure(
        structure=TEST_SI_STRUCTURE, net_atomic_charges=charges
    )
    assert structure_with_charges.site_properties[
        CHARGEMOL_NET_ATOMIC_CHARGES_PROPERTY
    ] == [0.1, -0.1]
