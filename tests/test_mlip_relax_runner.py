"""Tests for MLIP runner utilities."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointDFTCalculator
from matgl.apps.pes import Potential
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure as PMG_Structure

from crystal import CRYSTAL_CONFIG
from crystal.calculators.base_atomistic_runner import DEFAULT_RELAX_TRAJECTORY_FILE
from crystal.calculators.mlip.mlip_runner import (
    MlipModelError,
    MlipRunner,
    get_mlip_isolated_atom_energies,
    run_mlip_relax,
)
from crystal.constants.qe import DEFAULT_RELAX_LOGFILE
from crystal.helpers.mlip_helpers import configure_default_torch_device
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.mlip import MlipRelaxSimulationSettings
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from crystal.schemas.relax_settings import OptimizerType, RelaxOptimizerSettings
from crystal.schemas.relaxation_constraints import RelaxationConstraints
from tests.helpers.constants.properties import MOCK_MLIP_RELAX_PROPERTIES
from tests.helpers.constants.structure import TEST_SI_STRUCTURE


def test_get_mlip_isolated_atom_energies_reads_model_element_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_tensor = Mock()
    reference_tensor.detach.return_value.cpu.return_value.tolist.return_value = [
        -5.0,
        -3.0,
    ]
    potential = MagicMock(spec=Potential)
    potential.model = SimpleNamespace(element_types=("Si", "O"))
    potential.element_refs = SimpleNamespace(property_offset=reference_tensor)
    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_potential",
        lambda _, **__: potential,
    )

    isolated_atom_energies = get_mlip_isolated_atom_energies(
        MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    )

    assert isolated_atom_energies == {"Si": -5.0, "O": -3.0}


def test_get_mlip_isolated_atom_energies_rejects_models_without_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    potential = MagicMock(spec=Potential)
    potential.model = SimpleNamespace(element_types=("Si",))
    potential.element_refs = None
    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_potential",
        lambda _, **__: potential,
    )

    with pytest.raises(MlipModelError, match="does not contain element references"):
        get_mlip_isolated_atom_energies(
            MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
        )


def test_get_mlip_isolated_atom_energies_rejects_models_without_element_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    potential = MagicMock(spec=Potential)
    potential.model = SimpleNamespace()
    potential.element_refs = MagicMock()
    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_potential",
        lambda _, **__: potential,
    )

    with pytest.raises(MlipModelError, match="does not contain element types"):
        get_mlip_isolated_atom_energies(
            MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
        )


def test_get_mlip_isolated_atom_energies_rejects_mismatched_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_tensor = Mock()
    reference_tensor.detach.return_value.cpu.return_value.tolist.return_value = [
        -5.0,
        -3.0,
    ]
    potential = MagicMock(spec=Potential)
    potential.model = SimpleNamespace(element_types=("Si",))
    potential.element_refs = SimpleNamespace(property_offset=reference_tensor)
    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_potential",
        lambda _, **__: potential,
    )

    with pytest.raises(MlipModelError, match="does not match"):
        get_mlip_isolated_atom_energies(
            MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
        )


def test_mlip_runner_normalizes_partial_pbc_before_model_evaluation() -> None:
    partial_structure = PMG_Structure(
        lattice=Lattice(matrix=np.diag([4.0, 10.0, 12.0]), pbc=(True, True, False)),
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 5.0], [3.5, 0.0, 5.0]],
        coords_are_cartesian=True,
    )
    full_structure = deepcopy(partial_structure)
    full_structure.lattice.pbc = (True, True, True)
    model = MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)

    partial_properties = MlipRunner(
        structure=partial_structure, model=model
    ).run_single_point()
    full_properties = MlipRunner(
        structure=full_structure, model=model
    ).run_single_point()

    assert partial_properties.energy == pytest.approx(full_properties.energy)
    assert partial_properties.structure == full_properties.structure
    assert tuple(partial_structure.lattice.pbc) == (True, True, False)


@run_in_temp_dir
def test_mlip_run_relax_without_constraints() -> None:
    relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=0.0,
        max_steps=1,
    )
    runner = MlipRunner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe),
    )
    properties = runner.run_relax(optimizer_settings=relax_optimizer_settings)
    assert isinstance(properties, RelaxProperties)
    [relax_step_0, relax_step_n] = properties.relax_steps
    assert math.isclose(relax_step_0.energy, -5.899, abs_tol=1e-1)
    assert math.isclose(relax_step_n.energy, -5.922, abs_tol=1e-1)
    assert relax_step_n.structure != relax_step_0.structure
    assert relax_step_n.charges is not relax_step_0.charges


@run_in_temp_dir
def test_mlip_run_relax_can_disable_trajectory_file() -> None:
    relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=0.0,
        max_steps=1,
    )
    runner = MlipRunner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe),
    )

    runner.run_relax(
        optimizer_settings=relax_optimizer_settings, store_trajectory=False
    )

    assert not Path("relax.traj").exists()


@run_in_temp_dir
def test_run_mlip_relax_keeps_all_step_properties_by_default() -> None:
    simulation_settings = MlipRelaxSimulationSettings(
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    )
    properties = RelaxProperties(
        relax_steps=[
            MOCK_MLIP_RELAX_PROPERTIES.relax_steps[0],
            MOCK_MLIP_RELAX_PROPERTIES.relax_steps[0].model_copy(update={"step": 1}),
        ]
    )

    with patch.object(MlipRunner, "run_relax", return_value=properties):
        result = run_mlip_relax(
            structure=deepcopy(TEST_SI_STRUCTURE),
            simulation_settings=simulation_settings,
        )

    assert result is properties
    assert len(result.relax_steps) == 2


@run_in_temp_dir
def test_run_mlip_relax_keeps_only_final_step_properties() -> None:
    simulation_settings = MlipRelaxSimulationSettings(
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe),
        store_final_step_properties_only=True,
    )
    properties = RelaxProperties(
        relax_steps=[
            MOCK_MLIP_RELAX_PROPERTIES.relax_steps[0],
            MOCK_MLIP_RELAX_PROPERTIES.relax_steps[0].model_copy(
                update={"step": 1, "energy": 1.2}
            ),
        ]
    )

    with patch.object(MlipRunner, "run_relax", return_value=properties):
        result = run_mlip_relax(
            structure=deepcopy(TEST_SI_STRUCTURE),
            simulation_settings=simulation_settings,
        )

    assert result is not properties
    assert result.relax_steps == [properties.relax_steps[-1]]
    assert result.final_energy == properties.relax_steps[-1].energy
    assert result.relaxed_structure == properties.relax_steps[-1].structure


@run_in_temp_dir
def test_mlip_run_relax_stores_artifacts_in_output_directory() -> None:
    relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=0.0,
        max_steps=1,
    )
    output_directory = Path("relax-output")
    runner = MlipRunner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe),
    )

    runner.run_relax(
        optimizer_settings=relax_optimizer_settings, output_directory=output_directory
    )

    assert (output_directory / DEFAULT_RELAX_LOGFILE).is_file()
    assert (output_directory / DEFAULT_RELAX_TRAJECTORY_FILE).is_file()
    assert not Path(DEFAULT_RELAX_LOGFILE).exists()
    assert not Path(DEFAULT_RELAX_TRAJECTORY_FILE).exists()


@run_in_temp_dir
def test_mlip_run_relax_with_custom_constraints() -> None:
    relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=0.7,
        max_steps=20,
    )
    constraints = RelaxationConstraints(fix_cell=False)
    runner = MlipRunner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe),
    )
    properties = runner.run_relax(
        optimizer_settings=relax_optimizer_settings, constraints=constraints
    )
    assert len(properties.relax_steps) == 2
    [relax_step_0, *_, relax_step_n] = properties.relax_steps
    assert relax_step_0.structure.lattice != relax_step_n.structure.lattice


@pytest.mark.parametrize(
    ("kwargs", "init_context", "path_context", "expected_model_path_stem"),
    [
        (
            {"registry_model": QetModelRegistry.foundational_matq},
            nullcontext(),
            nullcontext(),
            "QET-PES-MatQ",
        ),
        (
            {"path": Path("/this/model/does_not/exist")},
            nullcontext(),
            pytest.raises(ValueError, match="does not exist"),
            None,
        ),
        (
            {"path": Path("/this/model/does_not/exist"), "registry_model": None},
            nullcontext(),
            pytest.raises(ValueError, match="does not exist"),
            None,
        ),
        (
            {
                "path": Path("/this/model/does_not/exist"),
                "registry_model": QetModelRegistry.foundational_matq,
            },
            pytest.raises(ValueError, match="Provide only one of path or registry"),
            nullcontext(),
            None,
        ),
    ],
)
def test_mlip_model_ensure_path_or_registry(
    kwargs: dict[str, Path | QetModelRegistry | None],
    init_context: AbstractContextManager[object],
    path_context: AbstractContextManager[object],
    expected_model_path_stem: str | None,
) -> None:
    with init_context:
        model = MlipModelRef(**kwargs)

    if "model" not in locals():
        return

    with path_context:
        assert model.model_path.stem == expected_model_path_stem


def test_mlip_model_non_built_in_registry_resolves_from_model_dir() -> None:
    model = MlipModelRef(registry_model=QetModelRegistry.finetuned_linbosi)
    assert isinstance(model.model_path, Path)
    assert CRYSTAL_CONFIG.programs.mlip_models_dir is not None
    assert (
        model.model_path
        == CRYSTAL_CONFIG.programs.mlip_models_dir
        / QetModelRegistry.finetuned_linbosi.value
    )


def test_configure_default_torch_device_sets_cuda_when_gpus_requested() -> None:
    mock_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=Mock(return_value=True),
            current_device=Mock(return_value=0),
            get_device_name=Mock(return_value="Mock GPU"),
        ),
        set_default_device=Mock(),
    )

    with patch.dict("sys.modules", {"torch": mock_torch}):
        configure_default_torch_device(requested_gpu_count=1)

    mock_torch.set_default_device.assert_called_once_with("cuda")


def test_configure_default_torch_device_raises_without_cuda() -> None:
    mock_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=Mock(return_value=False)),
        set_default_device=Mock(),
    )

    with (
        patch.dict("sys.modules", {"torch": mock_torch}),
        pytest.raises(RuntimeError, match="CUDA is not available"),
    ):
        configure_default_torch_device(requested_gpu_count=1)


@run_in_temp_dir
def test_base_calculator_replaces_stale_ase_calc_from_structure() -> None:
    model = MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=0.0,
        max_steps=1,
    )
    relax_properties = MlipRunner(
        structure=deepcopy(TEST_SI_STRUCTURE), model=model
    ).run_relax(optimizer_settings=relax_optimizer_settings)
    relaxed_structure = relax_properties.relaxed_structure
    assert "final_charge" in relaxed_structure.site_properties
    roundtrip_relaxed_structure = PMG_Structure.from_dict(relaxed_structure.as_dict())

    single_point_runner = MlipRunner(structure=roundtrip_relaxed_structure, model=model)
    relaxed_structure_atoms = single_point_runner._ase_atoms

    with pytest.raises(
        PropertyNotImplementedError, match='The property "energy" is not available'
    ):
        _ = relaxed_structure_atoms.get_potential_energy()  # type:ignore[no-untyped-call]

    with pytest.raises(
        PropertyNotImplementedError, match='The property "forces" is not available'
    ):
        _ = relaxed_structure_atoms.get_forces()  # type:ignore[no-untyped-call]

    assert isinstance(single_point_runner._ase_atoms.calc, SinglePointDFTCalculator)

    single_point_properties = single_point_runner.run_single_point()
    assert isinstance(single_point_properties, SinglePointProperties)
    assert single_point_properties.energy == pytest.approx(-5.922, abs=1e-3)


@run_in_temp_dir
def test_mlip_relax_replaces_calculator_from_existing_structure_labels() -> None:
    labelled_structure = deepcopy(TEST_SI_STRUCTURE)
    labelled_structure.add_site_property(
        property_name="final_charge", values=[0.0, 0.0]
    )
    runner = MlipRunner(
        structure=labelled_structure,
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe),
    )
    assert isinstance(runner._ase_atoms.calc, SinglePointDFTCalculator)
    with pytest.raises(
        PropertyNotImplementedError, match='The property "energy" is not available'
    ):
        _ = runner._ase_atoms.get_potential_energy()  # type:ignore[no-untyped-call]

    properties = runner.run_relax(
        optimizer_settings=RelaxOptimizerSettings(
            optimizer_type=OptimizerType.BFGS,
            force_threshold_in_ev_per_angstrom=0.0,
            max_steps=1,
        ),
        store_trajectory=False,
    )
    assert properties.relax_steps
