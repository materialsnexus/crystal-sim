"""Unit tests for MLIP helper functions."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from matgl.ext.ase import Atoms2Graph

from crystal.helpers.mlip_helpers import prepare_atoms_for_matgl


def test_prepare_atoms_for_matgl_restores_periodic_graph_edges_without_mutating_input() -> (
    None
):
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 5.0], [3.5, 0.0, 5.0]],
        cell=[4.0, 10.0, 12.0],
        pbc=(True, True, False),
    )
    matgl_graph_converter = Atoms2Graph(element_types=("Si",), cutoff=1.0)
    original_cell = atoms.cell.array.copy()
    original_positions = atoms.positions.copy()

    matgl_ready_atoms = prepare_atoms_for_matgl(atoms)
    graph_from_partial_pbc_atoms, _, _ = matgl_graph_converter.get_graph(atoms)
    graph_from_matgl_ready_atoms, _, _ = matgl_graph_converter.get_graph(
        matgl_ready_atoms
    )

    assert graph_from_partial_pbc_atoms.edge_index.shape[1] == 0
    assert graph_from_matgl_ready_atoms.edge_index.shape[1] == 2
    assert tuple(matgl_ready_atoms.pbc) == (True, True, True)
    assert tuple(atoms.pbc) == (True, True, False)
    np.testing.assert_allclose(atoms.cell.array, original_cell)
    np.testing.assert_allclose(atoms.positions, original_positions)


@pytest.mark.parametrize(
    "pbc", [(False, True, True), (True, False, True), (False, False, False)]
)
def test_prepare_atoms_for_matgl_rejects_inputs_nonperiodic_in_x_or_y(
    pbc: tuple[bool, bool, bool],
) -> None:
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=pbc)

    with pytest.raises(ValueError, match="periodicity in X and Y"):
        prepare_atoms_for_matgl(atoms)
