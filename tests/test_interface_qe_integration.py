"""Integration test for interface structures passed into QE input generation."""

from pathlib import Path

import numpy as np
import pytest
from ase.io.espresso import write_espresso_in
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.calculators.qe.espresso_arguments_builder import EspressoArgumentsBuilder
from crystal.schemas.qe import QeConfig
from crystal.structure.interfaces.assembler import InterfaceAssembler
from crystal.structure.interfaces.utils import build_in_plane_offset_grid
from crystal.structure.slabs.builder import SlabFactory
from tests.helpers.constants.qe import TEST_ESPRESSO_PROFILE
from tests.helpers.constants.structure import (
    SAMPLE_QE_SCF_CONFIG_PATH,
    TEST_SI_PSEUDOPOTENTIAL_PATH,
    TEST_SI_STRUCTURE,
)
from tests.helpers.test_paths import (
    TEST_INTERFACE_GRID_INPUT_TEMPLATE,
    TEST_INTERFACE_QE_INTEGRATION_INPUT_FILENAME,
    TEST_SI_PSEUDOPOTENTIAL_FILENAME,
)


def _atomic_positions_block(qe_input_text: str) -> list[str]:
    lines = qe_input_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().startswith("ATOMIC_POSITIONS"):
            start = index + 1
            break
    if start is None:
        msg = "ATOMIC_POSITIONS section not found"
        raise ValueError(msg)
    block_lines: list[str] = []
    for line in lines[start:]:
        if line.strip() == "":
            break
        block_lines.append(line)
    return block_lines


def _count_atomic_position_lines(qe_input_text: str, element_symbol: str) -> int:
    block_lines = _atomic_positions_block(qe_input_text)
    return sum(1 for line in block_lines if line.strip().startswith(element_symbol))


def test_interface_structure_feeds_qe_input(tmp_path: Path) -> None:
    silicon_slab_factory = SlabFactory(
        structure=TEST_SI_STRUCTURE,
        supercell=(1, 1, 1),
        vacuum_thickness_in_layers=2,
        surface_miller_indices=[0, 0, 1],
    )
    silicon_slab = silicon_slab_factory.generate_slabs()[0]
    interface_assembler = InterfaceAssembler(
        top_slab=silicon_slab, bottom_slab=silicon_slab
    )
    [interface_structure] = list(
        interface_assembler.generate_interfaces(
            offsets=build_in_plane_offset_grid((1, 1)), gaps=[1.0], vacuum=0.0
        )
    )

    ase_atoms = AseAtomsAdaptor().get_atoms(interface_structure)
    assert len(ase_atoms) == len(interface_structure)
    assert np.allclose(ase_atoms.get_cell().array, interface_structure.lattice.matrix)  # type: ignore[no-untyped-call]

    qe_config = QeConfig.from_file(config_path=SAMPLE_QE_SCF_CONFIG_PATH)
    qe_arguments_builder = EspressoArgumentsBuilder(
        config=qe_config,
        pseudopotentials={"Si": str(TEST_SI_PSEUDOPOTENTIAL_PATH)},
        profile=TEST_ESPRESSO_PROFILE,
    )
    qe_kwargs = qe_arguments_builder.calculator_kwargs

    qe_input_path = tmp_path / TEST_INTERFACE_QE_INTEGRATION_INPUT_FILENAME
    with qe_input_path.open("w", encoding="utf-8") as handle:
        write_espresso_in(
            handle,
            ase_atoms,
            input_data=qe_kwargs["input_data"],
            pseudopotentials=qe_kwargs["pseudopotentials"],
            kspacing=qe_kwargs.get("kspacing"),
            kpts=qe_kwargs.get("kpts"),
            koffset=qe_kwargs.get("koffset") or (0, 0, 0),
        )

    qe_input_text = qe_input_path.read_text()
    assert TEST_SI_PSEUDOPOTENTIAL_FILENAME in qe_input_text
    assert "ATOMIC_POSITIONS" in qe_input_text
    assert _count_atomic_position_lines(qe_input_text, "Si") == len(ase_atoms)


@pytest.mark.parametrize(
    ("grid_dimensions", "expected_count", "expect_variation"),
    [((2, 2), 4, True), ((3, 1), 3, True)],
)
def test_interface_grid_feeds_qe_inputs(
    tmp_path: Path,
    grid_dimensions: tuple[int, int],
    expected_count: int,
    *,
    expect_variation: bool,
) -> None:
    silicon_slab_factory = SlabFactory(
        structure=TEST_SI_STRUCTURE,
        supercell=(1, 1, 1),
        vacuum_thickness_in_layers=2,
        surface_miller_indices=[0, 0, 1],
    )
    silicon_slab = silicon_slab_factory.generate_slabs()[0]
    interface_assembler = InterfaceAssembler(
        top_slab=silicon_slab, bottom_slab=silicon_slab
    )
    in_plane_offsets = build_in_plane_offset_grid(grid_dimensions)
    interface_structures = list(
        interface_assembler.generate_interfaces(
            offsets=in_plane_offsets, gaps=[1.0], vacuum=0.0
        )
    )
    assert len(interface_structures) == expected_count

    qe_config = QeConfig.from_file(config_path=SAMPLE_QE_SCF_CONFIG_PATH)
    qe_arguments_builder = EspressoArgumentsBuilder(
        config=qe_config,
        pseudopotentials={"Si": str(TEST_SI_PSEUDOPOTENTIAL_PATH)},
        profile=TEST_ESPRESSO_PROFILE,
    )
    qe_kwargs = qe_arguments_builder.calculator_kwargs

    atomic_position_blocks: list[str] = []
    for index, interface_structure in enumerate(interface_structures):
        ase_atoms = AseAtomsAdaptor().get_atoms(interface_structure)
        qe_input_path = tmp_path / TEST_INTERFACE_GRID_INPUT_TEMPLATE.format(
            index=index
        )
        with qe_input_path.open("w", encoding="utf-8") as handle:
            write_espresso_in(
                handle,
                ase_atoms,
                input_data=qe_kwargs["input_data"],
                pseudopotentials=qe_kwargs["pseudopotentials"],
                kspacing=qe_kwargs.get("kspacing"),
                kpts=qe_kwargs.get("kpts"),
                koffset=qe_kwargs.get("koffset") or (0, 0, 0),
            )
        qe_input_text = qe_input_path.read_text()
        assert _count_atomic_position_lines(qe_input_text, "Si") == len(ase_atoms)
        atomic_position_blocks.append("\n".join(_atomic_positions_block(qe_input_text)))

    if expect_variation:
        assert len(set(atomic_position_blocks)) > 1
