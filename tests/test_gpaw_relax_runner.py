"""Behavior tests for GPAW external relax execution."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crystal.calculators.gpaw.relax_runner import run_gpaw_relax
from crystal.constants.gpaw import (
    GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
    GPAW_ENV_PYTHON_RELATIVE_PATH,
    GPAW_EXTERNAL_RELAX_SCRIPT_FILENAME,
    GPAW_SCF_INPUT_STRUCTURE_FILENAME,
    GPAW_SCF_KWARGS_FILENAME,
    GPAW_SCF_OUTPUT_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
    GpawExternalRelaxScriptOutputKey,
    GpawExternalRelaxScriptStepKey,
)
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.schemas.gpaw import (
    GpawExternalInstallPathsConfig,
    GpawRelaxSimulationSettings,
)
from crystal.schemas.properties import RelaxProperties
from crystal.schemas.relax_settings import OptimizerType
from crystal.schemas.scratch_execution import LocalScratchPolicy
from tests.helpers.constants.process import TEST_PROCESS_SUCCESS_EXIT_CODE
from tests.helpers.constants.structure import (
    TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
    TEST_RELAX_MAX_STEPS,
    TEST_SI_STRUCTURE,
)
from tests.helpers.test_paths import (
    TEST_GPAW_ENV_ROOT_DIRNAME,
    TEST_GPAW_SETUPS_DIRNAME,
)

TEST_GPAW_MOCK_COMMAND_ARGUMENTS = ["gpaw"]
TEST_GPAW_RELAX_STEP_INDEX_ZERO = 0
TEST_GPAW_RELAX_STEP_INDEX_ONE = 1
TEST_GPAW_RELAX_STEP_STRESS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TEST_GPAW_RELAX_STEP_FORCES = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
TEST_GPAW_RELAX_FIRST_STEP_ENERGY = -1.0
TEST_GPAW_RELAX_FINAL_STEP_ENERGY = -1.5
TEST_GPAW_RELAX_STEP_COUNT = 2
TEST_GPAW_INVALID_PBC_VALUES = ["true", "true", "false"]
TEST_GPAW_RELAX_LOG_TEXT = "GPAW relax text log"


def _build_relax_simulation_settings_for_test() -> GpawRelaxSimulationSettings:
    return GpawRelaxSimulationSettings(
        optimizer_type=OptimizerType.BFGS,
        max_steps=TEST_RELAX_MAX_STEPS,
        force_threshold_in_ev_per_angstrom=TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
    )


def _build_relax_step_payload(*, step: int, energy: float) -> dict[str, object]:
    return {
        GpawExternalRelaxScriptStepKey.step: step,
        GpawExternalRelaxScriptStepKey.symbols: [
            str(site.specie.symbol) for site in TEST_SI_STRUCTURE
        ],
        GpawExternalRelaxScriptStepKey.positions: TEST_SI_STRUCTURE.cart_coords.tolist(),
        GpawExternalRelaxScriptStepKey.cell: TEST_SI_STRUCTURE.lattice.matrix.tolist(),
        GpawExternalRelaxScriptStepKey.pbc: [True, True, True],
        GpawExternalRelaxScriptStepKey.forces: TEST_GPAW_RELAX_STEP_FORCES,
        GpawExternalRelaxScriptStepKey.stress: TEST_GPAW_RELAX_STEP_STRESS,
        GpawExternalRelaxScriptStepKey.energy: energy,
    }


def _build_external_runtime_config_for_test(
    *, root_directory: Path
) -> GpawExternalInstallPathsConfig:
    gpaw_env_root = root_directory / TEST_GPAW_ENV_ROOT_DIRNAME
    return GpawExternalInstallPathsConfig(
        gpaw_env_root=gpaw_env_root,
        python_executable_path=gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH,
        gpaw_executable_path=gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
        gpaw_setups_dir=root_directory / TEST_GPAW_SETUPS_DIRNAME,
    )


@patch(
    "crystal.calculators.gpaw.relax_runner.GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime"
)
def test_run_gpaw_relax_uses_single_external_process_and_returns_relax_steps(
    mock_load_and_validate_external_runtime: MagicMock, tmp_path: Path
) -> None:
    mock_load_and_validate_external_runtime.return_value = (
        _build_external_runtime_config_for_test(root_directory=tmp_path)
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        TEST_GPAW_MOCK_COMMAND_ARGUMENTS, TEST_PROCESS_SUCCESS_EXIT_CODE
    )

    def _mock_subprocess_run(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output_path = tmp_path / GPAW_SCF_OUTPUT_FILENAME
        output_path.write_text(
            json.dumps(
                {
                    "relax_steps": [
                        _build_relax_step_payload(
                            step=TEST_GPAW_RELAX_STEP_INDEX_ZERO,
                            energy=TEST_GPAW_RELAX_FIRST_STEP_ENERGY,
                        ),
                        _build_relax_step_payload(
                            step=TEST_GPAW_RELAX_STEP_INDEX_ONE,
                            energy=TEST_GPAW_RELAX_FINAL_STEP_ENERGY,
                        ),
                    ]
                }
            ),
            encoding=DEFAULT_TEXT_ENCODING,
        )
        return completed_process

    with patch(
        "crystal.calculators.gpaw.external_runner.subprocess.run",
        side_effect=_mock_subprocess_run,
    ):
        relax_properties = run_gpaw_relax(
            structure=deepcopy(TEST_SI_STRUCTURE),
            simulation_settings=_build_relax_simulation_settings_for_test(),
            run_directory=tmp_path,
        )

    assert isinstance(relax_properties, RelaxProperties)
    assert len(relax_properties.relax_steps) == TEST_GPAW_RELAX_STEP_COUNT
    assert relax_properties.relax_steps[0].step == TEST_GPAW_RELAX_STEP_INDEX_ZERO
    assert relax_properties.final_energy == TEST_GPAW_RELAX_FINAL_STEP_ENERGY
    assert (
        relax_properties.relaxed_structure.composition == TEST_SI_STRUCTURE.composition
    )
    assert (tmp_path / GPAW_EXTERNAL_RELAX_SCRIPT_FILENAME).is_file()
    assert (tmp_path / GPAW_SCF_INPUT_STRUCTURE_FILENAME).is_file()
    assert (tmp_path / GPAW_SCF_KWARGS_FILENAME).is_file()
    assert (tmp_path / GPAW_SCF_OUTPUT_FILENAME).is_file()


@patch(
    "crystal.calculators.gpaw.relax_runner.GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime"
)
def test_run_gpaw_relax_executes_in_scratch_and_moves_gpaw_artifacts_to_final_directory(
    mock_load_and_validate_external_runtime: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "slurm_tmp"
    final_directory = tmp_path / "persistent_relax"
    monkeypatch.setenv("SLURM_TMPDIR", scratch_parent.as_posix())
    mock_load_and_validate_external_runtime.return_value = (
        _build_external_runtime_config_for_test(root_directory=tmp_path)
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        TEST_GPAW_MOCK_COMMAND_ARGUMENTS, TEST_PROCESS_SUCCESS_EXIT_CODE
    )

    def _mock_subprocess_run(
        *_args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        cwd = kwargs["cwd"]
        assert isinstance(cwd, Path)
        execution_directory = cwd
        assert execution_directory.parent == scratch_parent
        assert execution_directory != final_directory
        (execution_directory / GPAW_TEXT_LOG_FILENAME).write_text(
            TEST_GPAW_RELAX_LOG_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )
        (execution_directory / GPAW_SCF_OUTPUT_FILENAME).write_text(
            json.dumps(
                {
                    GpawExternalRelaxScriptOutputKey.relax_steps: [
                        _build_relax_step_payload(
                            step=TEST_GPAW_RELAX_STEP_INDEX_ZERO,
                            energy=TEST_GPAW_RELAX_FIRST_STEP_ENERGY,
                        ),
                        _build_relax_step_payload(
                            step=TEST_GPAW_RELAX_STEP_INDEX_ONE,
                            energy=TEST_GPAW_RELAX_FINAL_STEP_ENERGY,
                        ),
                    ]
                }
            ),
            encoding=DEFAULT_TEXT_ENCODING,
        )
        return completed_process

    with patch(
        "crystal.calculators.gpaw.external_runner.subprocess.run",
        side_effect=_mock_subprocess_run,
    ):
        relax_properties = run_gpaw_relax(
            structure=deepcopy(TEST_SI_STRUCTURE),
            simulation_settings=_build_relax_simulation_settings_for_test(),
            run_directory=final_directory,
            local_scratch_policy=LocalScratchPolicy(enabled=True),
        )

    assert relax_properties.final_energy == TEST_GPAW_RELAX_FINAL_STEP_ENERGY
    assert (final_directory / GPAW_SCF_OUTPUT_FILENAME).is_file()
    assert (final_directory / GPAW_TEXT_LOG_FILENAME).read_text(
        encoding=DEFAULT_TEXT_ENCODING
    ) == TEST_GPAW_RELAX_LOG_TEXT
    assert not any(scratch_parent.rglob(GPAW_SCF_OUTPUT_FILENAME))
    assert not any(scratch_parent.rglob(GPAW_TEXT_LOG_FILENAME))


@patch(
    "crystal.calculators.gpaw.relax_runner.GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime"
)
def test_run_gpaw_relax_raises_runtime_error_when_external_relax_fails(
    mock_load_and_validate_external_runtime: MagicMock, tmp_path: Path
) -> None:
    mock_load_and_validate_external_runtime.return_value = (
        _build_external_runtime_config_for_test(root_directory=tmp_path)
    )
    with (
        patch(
            "crystal.calculators.gpaw.external_runner.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                returncode=1, cmd=TEST_GPAW_MOCK_COMMAND_ARGUMENTS
            ),
        ),
        pytest.raises(RuntimeError),
    ):
        _ = run_gpaw_relax(
            structure=deepcopy(TEST_SI_STRUCTURE),
            simulation_settings=_build_relax_simulation_settings_for_test(),
            run_directory=tmp_path,
        )


@pytest.mark.parametrize(
    "output_payload",
    [
        pytest.param({}, id="missing-relax-steps"),
        pytest.param(
            {
                GpawExternalRelaxScriptOutputKey.relax_steps: [
                    {
                        **_build_relax_step_payload(
                            step=TEST_GPAW_RELAX_STEP_INDEX_ZERO,
                            energy=TEST_GPAW_RELAX_FIRST_STEP_ENERGY,
                        ),
                        GpawExternalRelaxScriptStepKey.pbc: TEST_GPAW_INVALID_PBC_VALUES,
                    }
                ]
            },
            id="non-boolean-pbc-values",
        ),
    ],
)
@patch(
    "crystal.calculators.gpaw.relax_runner.GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime"
)
def test_run_gpaw_relax_raises_runtime_error_for_malformed_external_output(
    mock_load_and_validate_external_runtime: MagicMock,
    output_payload: dict[str, object],
    tmp_path: Path,
) -> None:
    mock_load_and_validate_external_runtime.return_value = (
        _build_external_runtime_config_for_test(root_directory=tmp_path)
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        TEST_GPAW_MOCK_COMMAND_ARGUMENTS, TEST_PROCESS_SUCCESS_EXIT_CODE
    )

    def _mock_subprocess_run(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output_path = tmp_path / GPAW_SCF_OUTPUT_FILENAME
        output_path.write_text(
            json.dumps(output_payload), encoding=DEFAULT_TEXT_ENCODING
        )
        return completed_process

    with (
        patch(
            "crystal.calculators.gpaw.external_runner.subprocess.run",
            side_effect=_mock_subprocess_run,
        ),
        pytest.raises(RuntimeError),
    ):
        _ = run_gpaw_relax(
            structure=deepcopy(TEST_SI_STRUCTURE),
            simulation_settings=_build_relax_simulation_settings_for_test(),
            run_directory=tmp_path,
        )
