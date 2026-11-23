"""Behavior tests for cohesive-energy structure filtering."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from crystal.helpers.cohesive_energy import filter_structures_by_cohesive_energy
from crystal.schemas.filtering import CohesiveEnergyFilteringSettings
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry

LI_ISOLATED_ATOM_ENERGY = -1.0
O_ISOLATED_ATOM_ENERGY = -2.0


def _structure(species: list[str]) -> Structure:
    return Structure(
        lattice=Lattice.cubic(10.0),
        species=species,
        coords=[[0.0, 0.0, 0.0]] * len(species),
    )


def test_cohesive_energy_filter_accepts_energy_equal_to_threshold() -> None:
    settings = CohesiveEnergyFilteringSettings(
        isolated_atom_energies={"Li": LI_ISOLATED_ATOM_ENERGY},
        maximum_cohesive_energy_per_atom=0.0,
    )

    result = filter_structures_by_cohesive_energy(
        structures=[_structure(["Li"])],
        total_energies=[LI_ISOLATED_ATOM_ENERGY],
        settings=settings,
    )

    assert result == [_structure(["Li"])]


def test_cohesive_energy_filter_calculates_mixed_compositions_and_atom_counts() -> None:
    structures = [_structure(["Li"]), _structure(["Li", "Li"]), _structure(["Li", "O"])]
    total_energies = [-1.25, -1.5, -3.5]
    settings = CohesiveEnergyFilteringSettings(
        isolated_atom_energies={
            "Li": LI_ISOLATED_ATOM_ENERGY,
            "O": O_ISOLATED_ATOM_ENERGY,
        },
        maximum_cohesive_energy_per_atom=0.0,
    )

    result = filter_structures_by_cohesive_energy(
        structures=structures, total_energies=total_energies, settings=settings
    )

    assert result == [structures[0], structures[2]]


def test_cohesive_energy_filter_uses_element_symbol_for_oxidized_sites() -> None:
    structure = _structure(["Si"])
    structure.add_oxidation_state_by_element({"Si": 4})
    settings = CohesiveEnergyFilteringSettings(
        isolated_atom_energies={"Si": -1.0}, maximum_cohesive_energy_per_atom=0.0
    )

    result = filter_structures_by_cohesive_energy(
        structures=[structure], total_energies=[-1.0], settings=settings
    )

    assert result == [structure]


def test_cohesive_energy_filter_uses_evaluator_model_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "crystal.helpers.cohesive_energy.get_mlip_isolated_atom_energies",
        lambda _: {"Li": LI_ISOLATED_ATOM_ENERGY},
    )
    settings = CohesiveEnergyFilteringSettings(
        evaluator_model=MlipModelRef(
            registry_model=QetModelRegistry.foundational_matpes_pbe
        ),
        maximum_cohesive_energy_per_atom=0.0,
    )

    result = filter_structures_by_cohesive_energy(
        structures=[_structure(["Li"])],
        total_energies=[LI_ISOLATED_ATOM_ENERGY],
        settings=settings,
    )

    assert result == [_structure(["Li"])]


def test_cohesive_energy_filter_can_be_disabled() -> None:
    settings = CohesiveEnergyFilteringSettings(maximum_cohesive_energy_per_atom=None)

    assert (
        filter_structures_by_cohesive_energy(
            structures=[_structure(["Li"])],
            total_energies=[LI_ISOLATED_ATOM_ENERGY],
            settings=settings,
        )
        is None
    )


def test_cohesive_energy_filtering_requires_reference_source_when_enabled() -> None:
    with pytest.raises(
        ValidationError,
        match="requires either explicit isolated atom energies or an evaluator MLIP model",
    ):
        CohesiveEnergyFilteringSettings(maximum_cohesive_energy_per_atom=0.0)
