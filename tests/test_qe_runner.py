"""Tests for QE SCF runner utilities."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock, patch

import ase
import numpy as np
import pytest
from ase.constraints import FixAtoms
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.calculators.qe.qe_runner import QeRunner, adapt_atoms_to_policy
from crystal.constants.qe import RYDBERG_TO_EV
from crystal.helpers.structure import (
    apply_constraints_to_atoms,
    get_ase_atoms_midpoint_along_c_axis,
)
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from crystal.schemas.qe import (
    EsmBoundaryCondition,
    IsolatedSystemPolicy,
    QeConfig,
    QeScfInputData,
    ScfElectronsParameters,
    ScfSystemParameters,
    SerializableEspressoProfile,
    Smearing,
)
from crystal.schemas.relax_settings import OptimizerType, RelaxOptimizerSettings
from crystal.schemas.relaxation_constraints import RelaxationConstraints
from tests.helpers.constants.properties import MOCK_SCF_PROPERTIES
from tests.helpers.constants.qe import TEST_ESPRESSO_PROFILE
from tests.helpers.constants.structure import (
    FLOATING_POINT_TOLERANCE,
    TEST_LINBO3_SLAB,
    TEST_SI_STRUCTURE,
)
from tests.helpers.test_data_paths import TESTS_DATA_DIR
from tests.helpers.test_paths import (
    TEST_PSEUDOPOTENTIALS_DIRNAME,
    TEST_SI_PSEUDOPOTENTIAL_FILENAME,
)

TEST_PSEUDOPOTENTIALS_DIR = TESTS_DATA_DIR / TEST_PSEUDOPOTENTIALS_DIRNAME
QE_COMMAND_WITH_FOUR_PROCESSES: str = SerializableEspressoProfile._command(
    number_of_processes=4
)
TEST_PROFILE_FOR_RUNNING_QE = SerializableEspressoProfile(  # type: ignore[no-untyped-call]
    command=QE_COMMAND_WITH_FOUR_PROCESSES,
    pseudo_dir=TEST_PSEUDOPOTENTIALS_DIR.parent.as_posix(),
)
TEST_PROFILE_FOR_RUNNING_QE_WITH_PBE = SerializableEspressoProfile(  # type: ignore[no-untyped-call]
    command=QE_COMMAND_WITH_FOUR_PROCESSES,
    pseudo_dir=TEST_PSEUDOPOTENTIALS_DIR.as_posix(),
)

TEST_ESM_BC_CONFIG = QeConfig(
    input_data=QeScfInputData(
        electrons=ScfElectronsParameters(conv_thr_energy=1e-3 * RYDBERG_TO_EV),
        system=ScfSystemParameters(
            assume_isolated=IsolatedSystemPolicy.esm,
            esm_bc=EsmBoundaryCondition.bc1,
            smearing=Smearing.cold,
        ),
    )
)
TEST_RUNNER_FOR_LINBO3 = QeRunner(
    structure=deepcopy(TEST_LINBO3_SLAB),
    config=TEST_ESM_BC_CONFIG,
    pseudopotentials={"Li": "Li.pbe", "Nb": "Nb.pbe", "O": "O.pbe"},
    espresso_profile=TEST_PROFILE_FOR_RUNNING_QE_WITH_PBE,
)


def make_default_qe_runner(
    structure: PMG_Structure,
    config: QeConfig | None = None,
    espresso_profile: SerializableEspressoProfile = TEST_ESPRESSO_PROFILE,
) -> QeRunner:
    return QeRunner(
        structure=structure,
        config=config or QeConfig(),
        pseudopotentials={"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME},
        espresso_profile=espresso_profile,
    )


@patch("crystal.calculators.qe.qe_runner.RecenteredEspresso")
def test_runner_builds_calculator_with_builder(mock_espresso: MagicMock) -> None:
    runner = make_default_qe_runner(
        structure=deepcopy(TEST_SI_STRUCTURE), espresso_profile=TEST_ESPRESSO_PROFILE
    )
    _ = runner._calculator

    assert mock_espresso.called
    kwargs = mock_espresso.call_args.kwargs
    assert kwargs["input_data"]["control"]["calculation"] == "scf"
    assert kwargs["pseudopotentials"] == {"Si": TEST_SI_PSEUDOPOTENTIAL_FILENAME}
    assert kwargs["kspacing"] == 0.25


@patch.object(QeRunner, "_get_output")
def test_run_single_point_calls_energy_force_stress(mock_runner: MagicMock) -> None:
    mock_runner.return_value = MOCK_SCF_PROPERTIES
    runner = make_default_qe_runner(structure=deepcopy(TEST_SI_STRUCTURE))
    qe_scf_properties = runner.run_single_point()

    assert qe_scf_properties.energy == 0.2
    assert qe_scf_properties.forces is not None
    assert qe_scf_properties.forces[0][2] == 0.0
    assert qe_scf_properties.stress is not None
    assert qe_scf_properties.stress[0] == 0


@pytest.mark.needs_third_party_executables
@run_in_temp_dir
def test_run_relax_without_constraints() -> None:
    config = QeConfig()
    config_override = config.get_modified_instance(
        overrides={
            "input_data": {
                "electrons": {
                    "conv_thr_energy": 1e-2 * RYDBERG_TO_EV,
                    "electron_maxstep": 100,
                }
            }
        }
    )
    relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=0.0,
        max_steps=1,
    )
    runner = make_default_qe_runner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        espresso_profile=TEST_PROFILE_FOR_RUNNING_QE,
        config=config_override,
    )
    properties = runner.run_relax(optimizer_settings=relax_optimizer_settings)
    assert isinstance(properties, RelaxProperties)
    [relax_step_1, relax_step_2] = properties.relax_steps
    assert math.isclose(relax_step_1.energy, -303.908, abs_tol=1e-1)
    assert math.isclose(relax_step_2.energy, -304.008, abs_tol=1e-1)
    assert relax_step_2.structure != runner.structure


@pytest.mark.needs_third_party_executables
@run_in_temp_dir
def test_qe_run_relax_with_constraints() -> None:
    qe_relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=0.07,
        max_steps=20,
    )
    constraints = RelaxationConstraints(fix_cell=False)
    runner = make_default_qe_runner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        espresso_profile=TEST_PROFILE_FOR_RUNNING_QE,
    )
    qe_relax_properties = runner.run_relax(
        optimizer_settings=qe_relax_optimizer_settings, constraints=constraints
    )
    assert len(qe_relax_properties.relax_steps) == 5


@run_in_temp_dir
def test_optimizer_uses_unit_cell_filter_for_fix_cell() -> None:
    qe_relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=2.0,
        max_steps=2,
    )
    constraints = RelaxationConstraints(fix_cell=False)
    runner = make_default_qe_runner(structure=deepcopy(TEST_SI_STRUCTURE))
    relaxer = runner._initialize_optimizer(
        atoms=runner._ase_atoms,
        optimizer_settings=qe_relax_optimizer_settings,
        fix_cell=constraints.fix_cell,
    )
    atoms = runner._get_atoms(relaxer.atoms)
    assert isinstance(atoms, ase.Atoms)


@run_in_temp_dir
def test_fix_indices_constraints() -> None:
    runner = make_default_qe_runner(structure=deepcopy(TEST_SI_STRUCTURE))
    constraints = RelaxationConstraints(fixed_atom_indices=[0])
    constrained_atoms = apply_constraints_to_atoms(
        atoms=runner._ase_atoms, constraints=constraints
    )
    assert constraints.fixed_atom_indices is not None
    assert len(constrained_atoms.constraints) == 1
    assert all(
        isinstance(constraint, FixAtoms) for constraint in constrained_atoms.constraints
    )

    qe_relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=2.0,
        max_steps=2,
    )
    runner = make_default_qe_runner(structure=deepcopy(TEST_SI_STRUCTURE))
    relaxer = runner._initialize_optimizer(
        atoms=runner._ase_atoms,
        optimizer_settings=qe_relax_optimizer_settings,
        fix_cell=True,
    )
    assert relaxer.atoms is not None
    assert isinstance(relaxer.atoms, ase.Atoms)
    atoms = runner._get_atoms(relaxer.atoms)
    assert isinstance(atoms, ase.Atoms)


def test_centered_structure_for_esm_bc() -> None:
    runner = make_default_qe_runner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        espresso_profile=TEST_PROFILE_FOR_RUNNING_QE,
        config=TEST_ESM_BC_CONFIG,
    )
    assert isinstance(runner.structure, PMG_Structure)
    initial_ase_atoms_center = get_ase_atoms_midpoint_along_c_axis(
        atoms=runner._ase_atoms
    )
    calculation_ase_atoms = adapt_atoms_to_policy(
        atoms=runner._ase_atoms,
        isolated_system_policy=runner.config.input_data.system.assume_isolated,
    )
    calculation_ase_atoms_center = get_ase_atoms_midpoint_along_c_axis(
        atoms=calculation_ase_atoms
    )

    assert not np.isclose(initial_ase_atoms_center, 0.0, atol=FLOATING_POINT_TOLERANCE)
    assert np.isclose(calculation_ase_atoms_center, 0.0, atol=FLOATING_POINT_TOLERANCE)


def test_esm_requires_orthogonal_c_axis() -> None:
    operation = SymmOp.from_rotation_and_translation(
        rotation_matrix=([1, 0, 0], [0, 1, 0], [0, 0.1, 1]), translation_vec=(0, 0, 0)
    )
    non_orthogonal_structure = TEST_SI_STRUCTURE.copy().apply_operation(operation)

    with pytest.raises(ValueError, match="c-axis orthogonal"):
        _ = make_default_qe_runner(
            structure=non_orthogonal_structure, config=TEST_ESM_BC_CONFIG
        )


@pytest.mark.needs_third_party_executables
@run_in_temp_dir
@patch("crystal.calculators.qe.qe_runner.adapt_atoms_to_policy")
def test_qe_scf_fails_with_esm_bc_without_centered_structure(
    mock_uncentered_atoms: MagicMock,
) -> None:
    runner = deepcopy(TEST_RUNNER_FOR_LINBO3)
    mock_uncentered_atoms.return_value = runner._ase_atoms.copy()  # type: ignore[no-untyped-call]

    with pytest.raises(RuntimeError, match="non-zero exit status"):
        _ = runner.run_single_point()


@pytest.mark.needs_third_party_executables
@run_in_temp_dir
def test_qe_scf_calculation_for_esm_bc() -> None:
    runner = deepcopy(TEST_RUNNER_FOR_LINBO3)
    properties = runner.run_single_point()
    assert isinstance(properties, SinglePointProperties)
    assert math.isclose(properties.energy, -25792.658, abs_tol=FLOATING_POINT_TOLERANCE)


@pytest.mark.needs_third_party_executables
@run_in_temp_dir
def test_qe_relax_calculation_for_esm_bc() -> None:
    runner = make_default_qe_runner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        espresso_profile=TEST_PROFILE_FOR_RUNNING_QE,
        config=TEST_ESM_BC_CONFIG,
    )
    qe_relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=1.0,
        max_steps=2,
    )
    properties = runner.run_relax(optimizer_settings=qe_relax_optimizer_settings)
    assert isinstance(properties, RelaxProperties)
    ase_atoms_in_relax_steps = [
        AseAtomsAdaptor.get_atoms(step.structure) for step in properties.relax_steps
    ]
    assert len(ase_atoms_in_relax_steps) <= qe_relax_optimizer_settings.max_steps + 1
    assert np.allclose(
        [
            get_ase_atoms_midpoint_along_c_axis(atoms=ase_atoms)
            for ase_atoms in ase_atoms_in_relax_steps
        ],
        [0.0] * len(ase_atoms_in_relax_steps),
        atol=FLOATING_POINT_TOLERANCE,
    )


@pytest.mark.needs_third_party_executables
@run_in_temp_dir
def test_recentered_espresso_called_once_per_step() -> None:
    """Ensure energy, forces, and stress are computed exactly once per step during relax."""
    runner = make_default_qe_runner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        espresso_profile=TEST_PROFILE_FOR_RUNNING_QE,
    )
    # Access the calculator and count the number of times force, energy are called
    calculator = runner._calculator
    original_calculate_method: Callable[..., None] = calculator.calculate
    call_counter = {"count": 0}

    def counting_calculate(*args: Any, **kwargs: Any) -> None:
        call_counter["count"] += 1
        return original_calculate_method(*args, **kwargs)

    calculator.calculate = counting_calculate  # type: ignore[method-assign]
    qe_relax_optimizer_settings = RelaxOptimizerSettings(
        optimizer_type=OptimizerType.BFGS,
        force_threshold_in_ev_per_angstrom=1.0,
        max_steps=2,
    )
    _ = runner.run_relax(optimizer_settings=qe_relax_optimizer_settings)
    # Each step should call calculate exactly once
    assert call_counter["count"] <= qe_relax_optimizer_settings.max_steps + 1
