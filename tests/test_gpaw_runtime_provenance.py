"""Tests for GPAW runtime provenance payload construction."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from crystal.calculators.gpaw.external_runtime.gpaw_runtime_provenance import (
    build_gpaw_atomic_data_payloads,
    build_gpaw_runtime_provenance_payload,
)

SILICON_SETUP_FILENAME = "Si.PBE.gz"
ALTERNATE_SILICON_SETUP_FILENAME = "Si.alternate.PBE.gz"
OXYGEN_SETUP_FILENAME = "O.PBE.gz"
SILICON_SETUP_CONTENT = b"silicon PAW setup"
ALTERNATE_SILICON_SETUP_CONTENT = b"alternate silicon PAW setup"
OXYGEN_SETUP_CONTENT = b"oxygen PAW setup"
SILICON_VALENCE_CHARGE = 4.0
OXYGEN_VALENCE_CHARGE = 6.0
PAW_DATA_KIND = "paw"
GPAW_VERSION = "26.7.0"


def _setup(
    *, element: str, filename: Path, valence_electron_count: float
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=element, filename=filename.as_posix(), Nv=valence_electron_count
    )


def test_runtime_payload_hashes_resolved_selected_paw_setup_files(
    tmp_path: Path,
) -> None:
    silicon_setup_path = tmp_path / SILICON_SETUP_FILENAME
    oxygen_setup_path = tmp_path / OXYGEN_SETUP_FILENAME
    silicon_setup_path.write_bytes(SILICON_SETUP_CONTENT)
    oxygen_setup_path.write_bytes(OXYGEN_SETUP_CONTENT)

    atomic_data_payloads = build_gpaw_atomic_data_payloads(
        setups=(
            _setup(
                element="Si",
                filename=silicon_setup_path,
                valence_electron_count=SILICON_VALENCE_CHARGE,
            ),
            _setup(
                element="O",
                filename=oxygen_setup_path,
                valence_electron_count=OXYGEN_VALENCE_CHARGE,
            ),
        )
    )

    assert atomic_data_payloads == [
        {
            "element_symbol": "Si",
            "paw_setup_sha256_digest": sha256(SILICON_SETUP_CONTENT).hexdigest(),
            "paw_setup_filename": SILICON_SETUP_FILENAME,
            "paw_setup_kind": PAW_DATA_KIND,
            "valence_electron_count": int(SILICON_VALENCE_CHARGE),
        },
        {
            "element_symbol": "O",
            "paw_setup_sha256_digest": sha256(OXYGEN_SETUP_CONTENT).hexdigest(),
            "paw_setup_filename": OXYGEN_SETUP_FILENAME,
            "paw_setup_kind": PAW_DATA_KIND,
            "valence_electron_count": int(OXYGEN_VALENCE_CHARGE),
        },
    ]


def test_runtime_provenance_payload_contains_only_version_and_resolved_atomic_data(
    tmp_path: Path,
) -> None:
    silicon_setup_path = tmp_path / SILICON_SETUP_FILENAME
    silicon_setup_path.write_bytes(SILICON_SETUP_CONTENT)

    runtime_provenance_payload = build_gpaw_runtime_provenance_payload(
        gpaw_version=GPAW_VERSION,
        setups=(
            _setup(
                element="Si",
                filename=silicon_setup_path,
                valence_electron_count=SILICON_VALENCE_CHARGE,
            ),
        ),
    )

    assert runtime_provenance_payload == {
        "gpaw_version": GPAW_VERSION,
        "atomic_data": [
            {
                "element_symbol": "Si",
                "paw_setup_sha256_digest": sha256(SILICON_SETUP_CONTENT).hexdigest(),
                "paw_setup_filename": SILICON_SETUP_FILENAME,
                "paw_setup_kind": PAW_DATA_KIND,
                "valence_electron_count": int(SILICON_VALENCE_CHARGE),
            }
        ],
    }


def test_runtime_payload_collapses_repeated_identical_resolved_paw_entries(
    tmp_path: Path,
) -> None:
    silicon_setup_path = tmp_path / SILICON_SETUP_FILENAME
    silicon_setup_path.write_bytes(SILICON_SETUP_CONTENT)

    atomic_data_payloads = build_gpaw_atomic_data_payloads(
        setups=(
            _setup(
                element="Si",
                filename=silicon_setup_path,
                valence_electron_count=SILICON_VALENCE_CHARGE,
            ),
            _setup(
                element="Si",
                filename=silicon_setup_path,
                valence_electron_count=SILICON_VALENCE_CHARGE,
            ),
        )
    )

    assert atomic_data_payloads == [
        {
            "element_symbol": "Si",
            "paw_setup_sha256_digest": sha256(SILICON_SETUP_CONTENT).hexdigest(),
            "paw_setup_filename": SILICON_SETUP_FILENAME,
            "paw_setup_kind": PAW_DATA_KIND,
            "valence_electron_count": int(SILICON_VALENCE_CHARGE),
        }
    ]


def test_runtime_payload_rejects_inconsistent_resolved_paw_entries(
    tmp_path: Path,
) -> None:
    silicon_setup_path = tmp_path / SILICON_SETUP_FILENAME
    alternate_silicon_setup_path = tmp_path / ALTERNATE_SILICON_SETUP_FILENAME
    silicon_setup_path.write_bytes(SILICON_SETUP_CONTENT)
    alternate_silicon_setup_path.write_bytes(ALTERNATE_SILICON_SETUP_CONTENT)

    with pytest.raises(RuntimeError, match="inconsistent atomic data for element Si"):
        build_gpaw_atomic_data_payloads(
            setups=(
                _setup(
                    element="Si",
                    filename=silicon_setup_path,
                    valence_electron_count=SILICON_VALENCE_CHARGE,
                ),
                _setup(
                    element="Si",
                    filename=alternate_silicon_setup_path,
                    valence_electron_count=SILICON_VALENCE_CHARGE,
                ),
            )
        )
