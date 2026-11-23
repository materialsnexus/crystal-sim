"""Runtime provenance helpers executed inside the external GPAW environment."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

GPAW_PAW_DATA_KIND = "paw"
GPAW_SETUP_DATA_MISMATCH_MESSAGE_TEMPLATE = (
    "GPAW resolved inconsistent atomic data for element {element}."
)
SHA256_READ_BLOCK_SIZE_BYTES = 1024 * 1024


def build_gpaw_runtime_provenance_payload(
    *, gpaw_version: str, setups: Iterable[Any]
) -> dict[str, object]:
    """Build GPAW runtime-only provenance from resolved external state."""
    return {
        "gpaw_version": gpaw_version,
        "atomic_data": build_gpaw_atomic_data_payloads(setups=setups),
    }


def build_gpaw_atomic_data_payloads(
    *, setups: Iterable[Any]
) -> list[dict[str, object]]:
    """Build one selected-PAW payload per element from resolved GPAW setups."""
    atomic_data_by_element: dict[str, dict[str, object]] = {}
    for setup in setups:
        element = str(setup.symbol)
        setup_path = Path(str(setup.filename))
        atomic_data_payload = {
            "element_symbol": element,
            "paw_setup_sha256_digest": calculate_file_sha256(setup_path),
            "paw_setup_filename": setup_path.name,
            "paw_setup_kind": GPAW_PAW_DATA_KIND,
            "valence_electron_count": round(float(setup.Nv)),
        }
        existing_atomic_data_payload = atomic_data_by_element.get(element)
        if existing_atomic_data_payload is None:
            atomic_data_by_element[element] = atomic_data_payload
            continue
        if existing_atomic_data_payload != atomic_data_payload:
            raise RuntimeError(
                GPAW_SETUP_DATA_MISMATCH_MESSAGE_TEMPLATE.format(element=element)
            )
    return list(atomic_data_by_element.values())


def calculate_file_sha256(path: Path) -> str:
    """Return the SHA256 digest of one selected setup file."""
    file_hasher = sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(SHA256_READ_BLOCK_SIZE_BYTES), b""):
            file_hasher.update(block)
    return file_hasher.hexdigest()
