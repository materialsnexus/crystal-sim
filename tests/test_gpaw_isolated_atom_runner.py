"""Tests for the GPAW isolated-atom external runner."""

from __future__ import annotations

import json
import subprocess
from math import nan
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import read

from crystal.calculators.gpaw.isolated_atom_runner import (
    GpawExternalIsolatedAtomRunFiles,
    GpawExternalIsolatedAtomRunner,
    run_gpaw_isolated_atom,
)
from crystal.constants.gpaw import (
    GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
    GPAW_ENV_PYTHON_RELATIVE_PATH,
    GPAW_ISOLATED_ATOM_OUTPUT_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
)
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.schemas.gpaw import (
    GpawExternalInstallPathsConfig,
    GpawIsolatedAtomSimulationSettings,
)
from crystal.schemas.scratch_execution import LocalScratchPolicy
from tests.helpers.test_paths import (
    TEST_GPAW_ENV_ROOT_DIRNAME,
    TEST_GPAW_SETUPS_DIRNAME,
)

TEST_GPAW_MOCK_COMMAND_ARGUMENTS = ["gpaw"]
TEST_GPAW_MOCK_ISOLATED_ATOM_ENERGY = -5.4321
TEST_GPAW_MOCK_MAGNETIC_MOMENT = 2.0
TEST_GPAW_RUNTIME_VERSION = "26.7.0"
TEST_GPAW_ISOLATED_ATOM_EIGENSOLVER = "etdm-fdpw"
TEST_GPAW_SILICON_SETUP_DIGEST = "a" * 64
TEST_GPAW_SCRATCH_JOB_ID = "98765"
TEST_GPAW_SILICON_ATOMIC_DATA: dict[str, str | int] = {
    "element_symbol": "Si",
    "paw_setup_sha256_digest": TEST_GPAW_SILICON_SETUP_DIGEST,
    "paw_setup_filename": "Si.PBE.gz",
    "paw_setup_kind": "paw",
    "valence_electron_count": 4,
}
TEST_GPAW_RUNTIME_PROVENANCE = {
    "gpaw_version": TEST_GPAW_RUNTIME_VERSION,
    "atomic_data": [TEST_GPAW_SILICON_ATOMIC_DATA],
}


def _external_runtime_config(tmp_path: Path) -> GpawExternalInstallPathsConfig:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    return GpawExternalInstallPathsConfig(
        gpaw_env_root=gpaw_env_root,
        python_executable_path=gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH,
        gpaw_executable_path=gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
        gpaw_setups_dir=tmp_path / TEST_GPAW_SETUPS_DIRNAME,
    )


def _isolated_atom_output_payload() -> dict[str, object]:
    return {
        "energy": TEST_GPAW_MOCK_ISOLATED_ATOM_ENERGY,
        "magnetic_moment": TEST_GPAW_MOCK_MAGNETIC_MOMENT,
        "eigensolver": TEST_GPAW_ISOLATED_ATOM_EIGENSOLVER,
        "runtime_provenance": TEST_GPAW_RUNTIME_PROVENANCE,
    }


def test_runner_writes_energy_only_atom_inputs_and_returns_output(
    tmp_path: Path,
) -> None:
    runner = GpawExternalIsolatedAtomRunner(
        element="Si",
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
        external_runtime_config=_external_runtime_config(tmp_path),
    )
    run_files = GpawExternalIsolatedAtomRunFiles(run_directory=tmp_path)
    run_files.output_path.write_text(
        json.dumps(_isolated_atom_output_payload()), encoding=DEFAULT_TEXT_ENCODING
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        TEST_GPAW_MOCK_COMMAND_ARGUMENTS, 0
    )

    with patch(
        "crystal.calculators.gpaw.external_runner.subprocess.run",
        return_value=completed_process,
    ):
        result = runner.run(run_directory=tmp_path)

    assert result.energy == TEST_GPAW_MOCK_ISOLATED_ATOM_ENERGY
    assert result.element == "Si"
    assert result.magnetic_moment == TEST_GPAW_MOCK_MAGNETIC_MOMENT
    assert result.eigensolver == TEST_GPAW_ISOLATED_ATOM_EIGENSOLVER
    assert result.runtime_provenance.gpaw_version == TEST_GPAW_RUNTIME_VERSION
    assert result.runtime_provenance.atomic_data_by_element[
        "Si"
    ].paw_setup_sha256_digest == (TEST_GPAW_SILICON_SETUP_DIGEST)
    script_input_payload = json.loads(
        run_files.kwargs_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
    )
    atoms = read(run_files.input_structure_path)
    assert isinstance(atoms, Atoms)
    assert atoms.get_chemical_symbols() == ["Si"]  # type: ignore[no-untyped-call]
    assert atoms.get_positions().tolist() == [  # type: ignore[no-untyped-call]
        [10.0, 10.0, 10.0]
    ]
    assert atoms.cell.lengths().tolist() == [20.0, 20.0, 20.0]
    assert atoms.pbc.tolist() == [False, False, False]
    assert set(script_input_payload) == {"calculator_kwargs"}
    assert script_input_payload["calculator_kwargs"]["txt"] == GPAW_TEXT_LOG_FILENAME
    assert not any("chargemol" in path.name for path in tmp_path.iterdir())


@patch(
    "crystal.calculators.gpaw.isolated_atom_runner.GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime"
)
def test_run_gpaw_isolated_atom_uses_scratch_and_moves_energy_artifacts(
    mock_load_and_validate_external_runtime: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "slurm_tmp"
    final_directory = tmp_path / "persistent_atom"
    monkeypatch.setenv("SLURM_TMPDIR", scratch_parent.as_posix())
    monkeypatch.setenv("SLURM_JOB_ID", TEST_GPAW_SCRATCH_JOB_ID)
    mock_load_and_validate_external_runtime.return_value = _external_runtime_config(
        tmp_path
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        TEST_GPAW_MOCK_COMMAND_ARGUMENTS, 0
    )

    def _mock_subprocess_run(
        *_args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        cwd = kwargs["cwd"]
        assert isinstance(cwd, Path)
        assert cwd.parent == scratch_parent
        (cwd / GPAW_TEXT_LOG_FILENAME).write_text("GPAW output", encoding="utf-8")
        (cwd / GPAW_ISOLATED_ATOM_OUTPUT_FILENAME).write_text(
            json.dumps(_isolated_atom_output_payload()), encoding=DEFAULT_TEXT_ENCODING
        )
        return completed_process

    with patch(
        "crystal.calculators.gpaw.external_runner.subprocess.run",
        side_effect=_mock_subprocess_run,
    ):
        result = run_gpaw_isolated_atom(
            element="Si",
            simulation_settings=GpawIsolatedAtomSimulationSettings(),
            run_directory=final_directory,
            local_scratch_policy=LocalScratchPolicy(enabled=True),
        )

    assert result.energy == TEST_GPAW_MOCK_ISOLATED_ATOM_ENERGY
    assert (final_directory / GPAW_ISOLATED_ATOM_OUTPUT_FILENAME).is_file()
    assert (final_directory / GPAW_TEXT_LOG_FILENAME).is_file()
    assert not any(scratch_parent.rglob(GPAW_ISOLATED_ATOM_OUTPUT_FILENAME))


def test_runner_raises_runtime_error_for_malformed_atom_output(tmp_path: Path) -> None:
    runner = GpawExternalIsolatedAtomRunner(
        element="Si",
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
        external_runtime_config=_external_runtime_config(tmp_path),
    )
    run_files = GpawExternalIsolatedAtomRunFiles(run_directory=tmp_path)
    malformed_output_payload = _isolated_atom_output_payload()
    malformed_output_payload.pop("eigensolver")
    run_files.output_path.write_text(
        json.dumps(malformed_output_payload), encoding=DEFAULT_TEXT_ENCODING
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        TEST_GPAW_MOCK_COMMAND_ARGUMENTS, 0
    )

    with (
        patch(
            "crystal.calculators.gpaw.external_runner.subprocess.run",
            return_value=completed_process,
        ),
        pytest.raises(RuntimeError, match="isolated atom"),
    ):
        runner.run(run_directory=tmp_path)


@pytest.mark.parametrize(
    "output_payload",
    [
        pytest.param(
            {**_isolated_atom_output_payload(), "energy": nan}, id="non-finite-energy"
        ),
        pytest.param(
            {**_isolated_atom_output_payload(), "magnetic_moment": nan},
            id="non-finite-magnetic-moment",
        ),
        pytest.param(
            {
                **_isolated_atom_output_payload(),
                "runtime_provenance": {
                    **TEST_GPAW_RUNTIME_PROVENANCE,
                    "atomic_data": [
                        {**TEST_GPAW_SILICON_ATOMIC_DATA, "element_symbol": "O"}
                    ],
                },
            },
            id="provenance-for-other-element",
        ),
        pytest.param(
            {
                **_isolated_atom_output_payload(),
                "runtime_provenance": {
                    **TEST_GPAW_RUNTIME_PROVENANCE,
                    "atomic_data": [
                        *TEST_GPAW_RUNTIME_PROVENANCE["atomic_data"],
                        {**TEST_GPAW_SILICON_ATOMIC_DATA, "element_symbol": "O"},
                    ],
                },
            },
            id="provenance-for-extra-element",
        ),
    ],
)
def test_runner_raises_runtime_error_for_invalid_isolated_atom_result(
    tmp_path: Path, output_payload: dict[str, object]
) -> None:
    runner = GpawExternalIsolatedAtomRunner(
        element="Si",
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
        external_runtime_config=_external_runtime_config(tmp_path),
    )
    run_files = GpawExternalIsolatedAtomRunFiles(run_directory=tmp_path)
    run_files.output_path.write_text(
        json.dumps(output_payload), encoding=DEFAULT_TEXT_ENCODING
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        TEST_GPAW_MOCK_COMMAND_ARGUMENTS, 0
    )

    with (
        patch(
            "crystal.calculators.gpaw.external_runner.subprocess.run",
            return_value=completed_process,
        ),
        pytest.raises(RuntimeError, match="isolated atom"),
    ):
        runner.run(run_directory=tmp_path)
