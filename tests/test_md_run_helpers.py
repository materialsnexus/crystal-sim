"""Unit tests for MD run helper functions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import ase
import pytest
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.helpers.md_run_helpers import (
    compute_slab_volumetric_density_g_cm3,
    get_md_step_count,
    load_atoms,
    run_equilibration_check,
)
from crystal.schemas.md.checks import AverageEnergyDriftCheck
from crystal.schemas.md.properties import MdStageResult
from crystal.schemas.md.settings import MdEnsemble
from tests.helpers.constants.md import (
    MOCK_DEFAULT_LOG_CONTENT,
    MOCK_USER_DEFINED_LOG_CONTENT,
    TEST_TRAJECTORY_PATH,
)
from tests.helpers.constants.structure import (
    FLOATING_POINT_TOLERANCE,
    LINBO3_STRUCTURE_FILENAME,
    TEST_BCC_FE_STRUCTURE,
)

TEST_SI_ATOMS = ase.Atoms(
    "Si2", positions=[[0, 0, 0], [0, 0, 2]], cell=[(1, 0, 0), (0, 1, 0), (0, 0, 2)]
)


@pytest.mark.parametrize(
    "atoms_arg", [TEST_SI_ATOMS, TEST_BCC_FE_STRUCTURE, LINBO3_STRUCTURE_FILENAME]
)
def test_load_atoms(atoms_arg: ase.Atoms | PMG_Structure | Path) -> None:
    atoms = load_atoms(structure_or_path=atoms_arg)
    assert isinstance(atoms, ase.Atoms)


def test_md_step_count_converts_ps_to_fs() -> None:
    assert get_md_step_count(simulation_time_ps=2.5, time_step_fs=0.5) == 5000


def test_compute_slab_volumetric_density_g_cm3_returns_expected_density_for_known_structure() -> (
    None
):
    atoms = deepcopy(TEST_SI_ATOMS)
    density = compute_slab_volumetric_density_g_cm3(atoms)
    expected_density = sum(atoms.get_masses()) * 1.66053906660 / atoms.get_volume()  # type:ignore[no-untyped-call]
    assert density == pytest.approx(expected_density, abs=FLOATING_POINT_TOLERANCE)


def test_run_equilibration_check(tmp_path: Path) -> None:
    default_log_path = tmp_path / "nve_default.log"
    default_log_path.write_text(MOCK_DEFAULT_LOG_CONTENT, encoding="utf-8")
    user_defined_log_path = tmp_path / "nve_user_defined.log"
    user_defined_log_path.write_text(MOCK_USER_DEFINED_LOG_CONTENT, encoding="utf-8")

    result = run_equilibration_check(
        md_stage_result=MdStageResult(
            ensemble=MdEnsemble.nve,
            trajectory_path=TEST_TRAJECTORY_PATH,
            default_log_path=default_log_path,
            user_defined_log_path=user_defined_log_path,
        ),
        time_averaging_window=0.1,
        check=AverageEnergyDriftCheck(max_absolute_drift_slope=0.25),
    )

    assert result.check_type == "AverageEnergyDriftCheck"
    assert not result.passed
    assert result.observed_value == pytest.approx(1.25)

    looser_check_result = run_equilibration_check(
        md_stage_result=MdStageResult(
            ensemble=MdEnsemble.nve,
            trajectory_path=TEST_TRAJECTORY_PATH,
            default_log_path=default_log_path,
            user_defined_log_path=user_defined_log_path,
        ),
        time_averaging_window=0.1,
        check=AverageEnergyDriftCheck(max_absolute_drift_slope=2.0),
    )

    assert looser_check_result.check_type == "AverageEnergyDriftCheck"
    assert looser_check_result.passed
    assert looser_check_result.observed_value == pytest.approx(1.25)
