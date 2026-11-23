"""Integration-level test that QE config/loading feeds Espresso kwargs correctly."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io.espresso import write_espresso_in

from crystal.calculators.qe.espresso_arguments_builder import EspressoArgumentsBuilder
from crystal.schemas.qe import QeConfig
from tests.helpers.constants.qe import TEST_ESPRESSO_PROFILE
from tests.helpers.constants.structure import (
    SAMPLE_QE_SCF_CONFIG_PATH,
    TEST_SI_PSEUDOPOTENTIAL_PATH,
)
from tests.helpers.test_paths import (
    TEST_PW_INPUT_FILENAME,
    TEST_SI_PSEUDOPOTENTIAL_FILENAME,
)


def test_espresso_calculator_arguments_include_expected_keywords() -> None:
    config = QeConfig.from_file(config_path=SAMPLE_QE_SCF_CONFIG_PATH)
    builder = EspressoArgumentsBuilder(
        config=config,
        pseudopotentials={"Si": str(TEST_SI_PSEUDOPOTENTIAL_PATH)},
        profile=TEST_ESPRESSO_PROFILE,
    )
    kwargs = builder.calculator_kwargs

    assert kwargs["input_data"]["control"]["calculation"] == "scf"
    assert kwargs["pseudopotentials"]["Si"].endswith(TEST_SI_PSEUDOPOTENTIAL_FILENAME)
    assert ("kspacing" in kwargs) ^ ("kpts" in kwargs)


def test_write_espresso_in_renders_expected_keywords(tmp_path: Path) -> None:
    config = QeConfig.from_file(config_path=SAMPLE_QE_SCF_CONFIG_PATH)
    builder = EspressoArgumentsBuilder(
        config=config,
        pseudopotentials={"Si": str(TEST_SI_PSEUDOPOTENTIAL_PATH)},
        profile=TEST_ESPRESSO_PROFILE,
    )
    kwargs = builder.calculator_kwargs

    atoms = Atoms("Si2", positions=[[0, 0, 0], [0, 0, 2]])
    atoms.set_cell([5.43, 5.43, 5.43])  # type: ignore[no-untyped-call]
    atoms.center()  # type: ignore[no-untyped-call]

    input_path = tmp_path / TEST_PW_INPUT_FILENAME
    with input_path.open("w", encoding="utf-8") as fd:
        write_espresso_in(
            fd,
            atoms,
            input_data=kwargs["input_data"],
            pseudopotentials=kwargs["pseudopotentials"],
            kspacing=kwargs.get("kspacing"),
            kpts=kwargs.get("kpts"),
            koffset=kwargs.get("koffset") or (0, 0, 0),
        )

    text = input_path.read_text()
    assert "calculation      = 'scf'" in text
    assert TEST_SI_PSEUDOPOTENTIAL_FILENAME in text
