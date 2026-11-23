"""Tests for cohesive-energy reference helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from crystal.helpers.cohesive_energy import (
    CohesiveEnergiesCalculator,
    scale_total_energy_with_num_atoms,
)
from crystal.helpers.structure_summary import StructureSummary
from crystal.schemas.isolated_atom_energies import (
    IsolatedAtomEnergiesRef,
    IsolatedAtomEnergiesRegistry,
)
from tests.helpers.constants.structure import TEST_LINBO3_STRUCTURE, TEST_SI_STRUCTURE

LI_ELEMENT_STR = "Li"
NB_ELEMENT_STR = "Nb"
O_ELEMENT_STR = "O"
SI_ELEMENT_STR = "Si"
TEST_LINBO3_ELEMENT_SYMBOLS = [str(site.specie) for site in TEST_LINBO3_STRUCTURE]

LI_MOCK_ENERGY = -1.5
NB_MOCK_ENERGY = -3.0
O_MOCK_ENERGY = -0.5

MOCK_LI_EXTXYZ_STRUCTURE = (
    "1\n"
    f"Properties=species:S:1:pos:R:3 energy={LI_MOCK_ENERGY} "
    f"element={LI_ELEMENT_STR}\n"
    f"{LI_ELEMENT_STR} 0 0 0\n"
)
MOCK_NB_EXTXYZ_STRUCTURE = (
    "1\n"
    f"Properties=species:S:1:pos:R:3 energy={NB_MOCK_ENERGY} "
    f"element={NB_ELEMENT_STR}\n"
    f"{NB_ELEMENT_STR} 0 0 0\n"
)
MOCK_O_EXTXYZ_STRUCTURE = (
    "1\n"
    f"Properties=species:S:1:pos:R:3 energy={O_MOCK_ENERGY} "
    f"element={O_ELEMENT_STR}\n"
    f"{O_ELEMENT_STR} 0 0 0\n"
)

MOCK_LINBO3_REFERENCE_TOTAL = (
    4 * LI_MOCK_ENERGY + 4 * NB_MOCK_ENERGY + 12 * O_MOCK_ENERGY
)

MOCK_LINBO3_ISOLATED_ATOM_ENERGIES_EXTXYZ = (
    MOCK_LI_EXTXYZ_STRUCTURE + MOCK_NB_EXTXYZ_STRUCTURE + MOCK_O_EXTXYZ_STRUCTURE
)

MOCK_ISOLATED_ATOM_ENERGIES_DICT = {
    LI_ELEMENT_STR: LI_MOCK_ENERGY,
    NB_ELEMENT_STR: NB_MOCK_ENERGY,
    O_ELEMENT_STR: O_MOCK_ENERGY,
}

MOCK_ISOLATED_ATOM_ENERGIES_METADATA_YAML = (
    "name: MOCK\n"
    "published_on: '2026-05-27'\n"
    "description: Mock isolated atom references.\n"
)


def _write_mock_isolated_atom_energies_artifact(
    root: Path,
    name: str = "tmp_mock_isolated_atom_energies",
    energies_text: str = MOCK_LINBO3_ISOLATED_ATOM_ENERGIES_EXTXYZ,
    metadata_text: str = MOCK_ISOLATED_ATOM_ENERGIES_METADATA_YAML,
) -> Path:
    artifact_dir = root / name
    artifact_dir.mkdir()
    (artifact_dir / "isolated_atom_energies.extxyz").write_text(
        energies_text, encoding="utf-8"
    )
    (artifact_dir / "metadata.yaml").write_text(metadata_text, encoding="utf-8")
    return artifact_dir


def _write_mock_extxyz(path: Path, extxyz_text: str) -> Path:
    path.write_text(extxyz_text, encoding="utf-8")
    return path


def test_isolated_atom_energies_ref_reads_energies_and_metadata(tmp_path: Path) -> None:
    artifact_dir = _write_mock_isolated_atom_energies_artifact(tmp_path)

    ref = IsolatedAtomEnergiesRef(path=artifact_dir)

    assert ref.energies == MOCK_ISOLATED_ATOM_ENERGIES_DICT
    assert ref.metadata["name"] == "MOCK"


def test_isolated_atom_energies_ref_reads_registry_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir = _write_mock_isolated_atom_energies_artifact(
        tmp_path, name=IsolatedAtomEnergiesRegistry.gpaw.value
    )
    monkeypatch.setattr(
        "crystal.CRYSTAL_CONFIG.programs.isolated_atom_energies_dir", tmp_path
    )

    ref = IsolatedAtomEnergiesRef(registry_energies=IsolatedAtomEnergiesRegistry.gpaw)

    assert artifact_dir == tmp_path / IsolatedAtomEnergiesRegistry.gpaw.value
    assert ref.energies == MOCK_ISOLATED_ATOM_ENERGIES_DICT


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "registry_energies": IsolatedAtomEnergiesRegistry.gpaw,
                "path": Path("refs"),
            },
            "Provide only one of path or registry_energies, not both.",
        )
    ],
)
def test_isolated_atom_energies_ref_ensure_path_or_registry(
    kwargs: dict[str, IsolatedAtomEnergiesRegistry | Path], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = IsolatedAtomEnergiesRef(**kwargs)


def test_isolated_atom_energies_ref_defaults_to_placeholder_registry() -> None:
    ref = IsolatedAtomEnergiesRef()

    assert ref.registry_energies == IsolatedAtomEnergiesRegistry.gpaw


def test_isolated_atom_energies_ref_with_path_uses_path_only(tmp_path: Path) -> None:
    ref = IsolatedAtomEnergiesRef(path=tmp_path)

    assert ref.registry_energies == IsolatedAtomEnergiesRegistry.gpaw
    assert ref.artifact_path == tmp_path


def test_cohesive_energies_calculator_raises_for_all_missing_references_across_structure() -> (
    None
):
    calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies={LI_ELEMENT_STR: LI_MOCK_ENERGY}
    )

    with pytest.raises(
        ValueError,
        match=(
            "Missing isolated atom energy references for elements: "
            f"{NB_ELEMENT_STR}, {O_ELEMENT_STR}"
        ),
    ):
        _ = calculator.from_pmg_structures(
            structures=[TEST_LINBO3_STRUCTURE], total_energies=[-20.0]
        )


def test_cohesive_energies_calculator_reports_all_missing_references_across_batch() -> (
    None
):
    calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies={NB_ELEMENT_STR: NB_MOCK_ENERGY}
    )

    with pytest.raises(
        ValueError,
        match=(
            "Missing isolated atom energy references for elements: "
            f"{LI_ELEMENT_STR}, {O_ELEMENT_STR}, {SI_ELEMENT_STR}"
        ),
    ):
        _ = calculator.from_pmg_structures(
            structures=[TEST_LINBO3_STRUCTURE, TEST_SI_STRUCTURE],
            total_energies=[-20.0, -5.0],
        )


def test_load_isolated_atom_energies_from_extxyz_raises_for_multi_atom_structure(
    tmp_path: Path,
) -> None:
    extxyz_path = _write_mock_extxyz(
        tmp_path / "isolated_atom_energies.extxyz",
        (
            "2\n"
            "Properties=species:S:1:pos:R:3 energy=-1.0\n"
            f"{LI_ELEMENT_STR} 0 0 0\n"
            f"{O_ELEMENT_STR} 0 0 0\n"
        ),
    )

    with pytest.raises(
        ValueError,
        match=(r"Isolated atom energy file must contain only single atom structures\."),
    ):
        _ = IsolatedAtomEnergiesRef.load_energies_from_extxyz(extxyz_path)


def test_load_isolated_atom_energies_from_extxyz_raises_for_duplicate_element(
    tmp_path: Path,
) -> None:
    extxyz_path = _write_mock_extxyz(
        tmp_path / "isolated_atom_energies.extxyz",
        MOCK_LI_EXTXYZ_STRUCTURE
        + (f"1\nProperties=species:S:1:pos:R:3 energy=-2.0\n{LI_ELEMENT_STR} 0 0 0\n"),
    )

    with pytest.raises(
        ValueError,
        match=f"Duplicate isolated atom energy reference for {LI_ELEMENT_STR}.",
    ):
        _ = IsolatedAtomEnergiesRef.load_energies_from_extxyz(extxyz_path)


def test_cohesive_energies_calculator_raises_for_mismatched_inputs() -> None:
    calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies=MOCK_ISOLATED_ATOM_ENERGIES_DICT
    )

    with pytest.raises(
        ValueError,
        match=(
            "Number of element symbol lists must match number of total energies: "
            r"1 element symbol lists, 2 energies\."
        ),
    ):
        _ = calculator.from_pmg_structures(
            structures=[TEST_LINBO3_STRUCTURE], total_energies=[-20.0, -30.0]
        )


def test_cohesive_energies_calculator_from_pmg_structures() -> None:
    calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies=MOCK_ISOLATED_ATOM_ENERGIES_DICT
    )

    cohesive_energies = calculator.from_pmg_structures(
        structures=[TEST_LINBO3_STRUCTURE, TEST_LINBO3_STRUCTURE],
        total_energies=[-20.0, -30.0],
    )

    assert len(cohesive_energies) == 2
    assert math.isclose(
        cohesive_energies[0], (-20.0 - MOCK_LINBO3_REFERENCE_TOTAL) / 20
    )
    assert math.isclose(
        cohesive_energies[1], (-30.0 - MOCK_LINBO3_REFERENCE_TOTAL) / 20
    )


def test_cohesive_energies_calculator_from_structure_summaries() -> None:
    calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies=MOCK_ISOLATED_ATOM_ENERGIES_DICT
    )
    atomic_numbers = np.asarray(
        [site.specie.Z for site in TEST_LINBO3_STRUCTURE], dtype=np.int_
    )

    cohesive_energies = calculator.from_structure_summaries(
        structure_summaries=[
            StructureSummary(
                atomic_numbers=atomic_numbers,
                forces=np.empty((len(TEST_LINBO3_STRUCTURE), 3), dtype=np.float64),
                charges=np.empty(len(TEST_LINBO3_STRUCTURE), dtype=np.float64),
                stress=np.empty(6, dtype=np.float64),
                total_energy=-20.0,
            ),
            StructureSummary(
                atomic_numbers=atomic_numbers,
                forces=np.empty((len(TEST_LINBO3_STRUCTURE), 3), dtype=np.float64),
                charges=np.empty(len(TEST_LINBO3_STRUCTURE), dtype=np.float64),
                stress=np.empty(6, dtype=np.float64),
                total_energy=-30.0,
            ),
        ]
    )

    assert len(cohesive_energies) == 2
    assert math.isclose(
        cohesive_energies[0], (-20.0 - MOCK_LINBO3_REFERENCE_TOTAL) / 20
    )
    assert math.isclose(
        cohesive_energies[1], (-30.0 - MOCK_LINBO3_REFERENCE_TOTAL) / 20
    )


def test_scale_total_energies_with_num_atoms() -> None:
    scaled_energy = scale_total_energy_with_num_atoms(
        total_energy=1.0, initial_num_atoms=2, final_num_atoms=4
    )
    assert scaled_energy == pytest.approx(2.0, abs=1e-3)


def test_scale_total_energies_fails_with_invalid_num_atoms() -> None:
    with pytest.raises(ValueError, match="Number of atoms must be"):
        _ = scale_total_energy_with_num_atoms(
            total_energy=1.0, initial_num_atoms=0, final_num_atoms=4
        )
