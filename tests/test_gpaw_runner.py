"""Tests for GPAW SCF runner utilities."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.calculators.gpaw.gpaw_runner import (
    GpawExternalScfRunFiles,
    GpawExternalScfRunner,
    adapt_atoms_to_gpaw_bulk_or_slab,
    run_gpaw_scf,
)
from crystal.constants.chargemol import (
    CHARGEMOL_ATOMIC_DENSITIES_DIRECTORY_PLACEHOLDER,
    CHARGEMOL_JOB_CONTROL_FILENAME,
)
from crystal.constants.gpaw import (
    GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
    GPAW_ENV_PYTHON_RELATIVE_PATH,
    GPAW_SCF_OUTPUT_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
    GPAW_TOTAL_DENSITY_CUBE_FILENAME,
)
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.helpers.structure import get_ase_atoms_midpoint_along_c_axis
from crystal.schemas.gpaw import (
    GpawBulkOrSlab,
    GpawExternalInstallPathsConfig,
    GpawScfSimulationSettings,
)
from crystal.schemas.scratch_execution import LocalScratchPolicy
from tests.helpers.constants.gpaw_chargemol import (
    TEST_CHARGEMOL_CORE_ELECTRONS_SI_ENTRY,
    TEST_CHARGEMOL_INPUT_FILENAME_OPENING_TAG,
    TEST_CHARGEMOL_NUMBER_OF_CORE_ELECTRONS_OPENING_TAG,
    TEST_CHARGEMOL_PERIODICITY_TAG_BLOCK_TRUE_TRUE_TRUE,
)
from tests.helpers.constants.process import TEST_PROCESS_SUCCESS_EXIT_CODE
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.test_paths import (
    TEST_GPAW_ENV_ROOT_DIRNAME,
    TEST_GPAW_SETUPS_DIRNAME,
)

TEST_GPAW_MOCK_COMMAND_ARGUMENTS = ["gpaw"]
TEST_GPAW_MOCK_SINGLE_POINT_ENERGY = -1.0
TEST_GPAW_ZERO_VECTOR = [0.0, 0.0, 0.0]
TEST_GPAW_MOCK_SINGLE_POINT_STRESS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TEST_GPAW_PERIODICITY_FLAGS = [True, True, True]
TEST_GPAW_MOCK_TOTAL_DENSITY_CUBE_TEXT = "mock cube artifact"
TEST_GPAW_SLAB_TRANSLATION_FRACTION = -1.0
TEST_GPAW_OVERSIZED_SLAB_TRANSLATION_FRACTION = 1.1
TEST_GPAW_INVALID_PBC_VALUES = ["true", "true", "false"]
TEST_GPAW_OUTPUT_FIELD_ENERGY = "energy"
TEST_GPAW_OUTPUT_FIELD_FORCES = "forces"
TEST_GPAW_OUTPUT_FIELD_STRESS = "stress"
TEST_GPAW_OUTPUT_FIELD_PBC = "pbc"
TEST_GPAW_OUTPUT_FIELD_RUNTIME_PROVENANCE = "runtime_provenance"
TEST_GPAW_SCRATCH_JOB_ID = "98765"
TEST_GPAW_RUNTIME_VERSION = "26.7.0"
TEST_GPAW_SILICON_SETUP_DIGEST = "a" * 64
TEST_GPAW_RUNTIME_PROVENANCE = {
    "gpaw_version": TEST_GPAW_RUNTIME_VERSION,
    "atomic_data": [
        {
            "element_symbol": "Si",
            "paw_setup_sha256_digest": TEST_GPAW_SILICON_SETUP_DIGEST,
            "paw_setup_filename": "Si.PBE.gz",
            "paw_setup_kind": "paw",
            "valence_electron_count": 4,
        }
    ],
}


def test_adapt_atoms_to_gpaw_bulk_or_slab_sets_slab_interface_pbc() -> None:
    atoms = AseAtomsAdaptor.get_atoms(deepcopy(TEST_SI_STRUCTURE))
    atoms.set_pbc((True, True, True))  # type: ignore[no-untyped-call]

    adapted_atoms = adapt_atoms_to_gpaw_bulk_or_slab(
        atoms=atoms, bulk_or_slab=GpawBulkOrSlab.slab_interface
    )
    assert tuple(adapted_atoms.pbc.tolist()) == (True, True, False)
    assert get_ase_atoms_midpoint_along_c_axis(atoms=adapted_atoms) == pytest.approx(
        0.0
    )


def test_adapt_atoms_to_gpaw_bulk_or_slab_keeps_bulk_pbc() -> None:
    atoms = AseAtomsAdaptor.get_atoms(deepcopy(TEST_SI_STRUCTURE))
    atoms.set_pbc((True, True, True))  # type: ignore[no-untyped-call]

    adapted_atoms = adapt_atoms_to_gpaw_bulk_or_slab(
        atoms=atoms, bulk_or_slab=GpawBulkOrSlab.bulk
    )
    assert tuple(adapted_atoms.pbc.tolist()) == (True, True, True)
    assert get_ase_atoms_midpoint_along_c_axis(atoms=adapted_atoms) == pytest.approx(
        0.0
    )


def test_runner_run_single_point_writes_chargemol_artifacts(tmp_path: Path) -> None:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    runner = GpawExternalScfRunner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        simulation_settings=GpawScfSimulationSettings(),
        external_runtime_config=GpawExternalInstallPathsConfig(
            gpaw_env_root=gpaw_env_root,
            python_executable_path=gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH,
            gpaw_executable_path=gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
            gpaw_setups_dir=tmp_path / TEST_GPAW_SETUPS_DIRNAME,
        ),
    )
    run_files = GpawExternalScfRunFiles(run_directory=tmp_path)
    total_density_cube_path = tmp_path / GPAW_TOTAL_DENSITY_CUBE_FILENAME
    total_density_cube_path.write_text(
        TEST_GPAW_MOCK_TOTAL_DENSITY_CUBE_TEXT, encoding=DEFAULT_TEXT_ENCODING
    )
    run_files.output_path.write_text(
        json.dumps(
            {
                "energy": TEST_GPAW_MOCK_SINGLE_POINT_ENERGY,
                "forces": [
                    TEST_GPAW_ZERO_VECTOR for _ in range(len(TEST_SI_STRUCTURE))
                ],
                "stress": TEST_GPAW_MOCK_SINGLE_POINT_STRESS,
                "pbc": TEST_GPAW_PERIODICITY_FLAGS,
                TEST_GPAW_OUTPUT_FIELD_RUNTIME_PROVENANCE: (
                    TEST_GPAW_RUNTIME_PROVENANCE
                ),
            }
        ),
        encoding=DEFAULT_TEXT_ENCODING,
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        TEST_GPAW_MOCK_COMMAND_ARGUMENTS, TEST_PROCESS_SUCCESS_EXIT_CODE
    )
    with patch(
        "crystal.calculators.gpaw.external_runner.subprocess.run",
        return_value=completed_process,
    ):
        result = runner.run_single_point(run_directory=tmp_path)

    assert result.properties.energy == TEST_GPAW_MOCK_SINGLE_POINT_ENERGY
    result_structure_atoms = AseAtomsAdaptor.get_atoms(result.properties.structure)
    assert get_ase_atoms_midpoint_along_c_axis(
        atoms=result_structure_atoms
    ) == pytest.approx(0.0)
    assert result.runtime_provenance.gpaw_version == TEST_GPAW_RUNTIME_VERSION
    assert result.runtime_provenance.atomic_data_by_element[
        "Si"
    ].paw_setup_sha256_digest == (TEST_GPAW_SILICON_SETUP_DIGEST)
    assert total_density_cube_path.is_file()
    job_control_path = tmp_path / CHARGEMOL_JOB_CONTROL_FILENAME
    assert job_control_path.is_file()
    job_control_text = job_control_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
    assert TEST_CHARGEMOL_INPUT_FILENAME_OPENING_TAG in job_control_text
    assert GPAW_TOTAL_DENSITY_CUBE_FILENAME in job_control_text
    assert TEST_CHARGEMOL_NUMBER_OF_CORE_ELECTRONS_OPENING_TAG in job_control_text
    assert "<net charge>\n0.0\n</net charge>" in job_control_text
    assert TEST_CHARGEMOL_PERIODICITY_TAG_BLOCK_TRUE_TRUE_TRUE in job_control_text
    assert TEST_CHARGEMOL_CORE_ELECTRONS_SI_ENTRY not in job_control_text
    assert CHARGEMOL_ATOMIC_DENSITIES_DIRECTORY_PLACEHOLDER in job_control_text


@patch(
    "crystal.calculators.gpaw.gpaw_runner.GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime"
)
def test_run_gpaw_scf_executes_in_scratch_and_moves_gpaw_artifacts_to_final_directory(
    mock_load_and_validate_external_runtime: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "slurm_tmp"
    final_directory = tmp_path / "persistent_scf"
    monkeypatch.setenv("SLURM_TMPDIR", scratch_parent.as_posix())
    monkeypatch.setenv("SLURM_JOB_ID", TEST_GPAW_SCRATCH_JOB_ID)
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    mock_load_and_validate_external_runtime.return_value = (
        GpawExternalInstallPathsConfig(
            gpaw_env_root=gpaw_env_root,
            python_executable_path=gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH,
            gpaw_executable_path=gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
            gpaw_setups_dir=tmp_path / TEST_GPAW_SETUPS_DIRNAME,
        )
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
        (execution_directory / GPAW_TOTAL_DENSITY_CUBE_FILENAME).write_text(
            TEST_GPAW_MOCK_TOTAL_DENSITY_CUBE_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )
        (execution_directory / GPAW_TEXT_LOG_FILENAME).write_text(
            TEST_GPAW_MOCK_TOTAL_DENSITY_CUBE_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )
        (execution_directory / GPAW_SCF_OUTPUT_FILENAME).write_text(
            json.dumps(
                {
                    TEST_GPAW_OUTPUT_FIELD_ENERGY: TEST_GPAW_MOCK_SINGLE_POINT_ENERGY,
                    TEST_GPAW_OUTPUT_FIELD_FORCES: [
                        TEST_GPAW_ZERO_VECTOR for _ in range(len(TEST_SI_STRUCTURE))
                    ],
                    TEST_GPAW_OUTPUT_FIELD_STRESS: TEST_GPAW_MOCK_SINGLE_POINT_STRESS,
                    TEST_GPAW_OUTPUT_FIELD_PBC: TEST_GPAW_PERIODICITY_FLAGS,
                    TEST_GPAW_OUTPUT_FIELD_RUNTIME_PROVENANCE: (
                        TEST_GPAW_RUNTIME_PROVENANCE
                    ),
                }
            ),
            encoding=DEFAULT_TEXT_ENCODING,
        )
        return completed_process

    with patch(
        "crystal.calculators.gpaw.external_runner.subprocess.run",
        side_effect=_mock_subprocess_run,
    ):
        result = run_gpaw_scf(
            structure=deepcopy(TEST_SI_STRUCTURE),
            simulation_settings=GpawScfSimulationSettings(),
            run_directory=final_directory,
            local_scratch_policy=LocalScratchPolicy(enabled=True),
        )

    assert result.properties.energy == TEST_GPAW_MOCK_SINGLE_POINT_ENERGY
    assert (final_directory / GPAW_SCF_OUTPUT_FILENAME).is_file()
    assert (final_directory / GPAW_TOTAL_DENSITY_CUBE_FILENAME).is_file()
    assert (final_directory / CHARGEMOL_JOB_CONTROL_FILENAME).is_file()
    assert (final_directory / GPAW_TEXT_LOG_FILENAME).is_file()
    assert not any(scratch_parent.rglob(GPAW_SCF_OUTPUT_FILENAME))
    assert not any(scratch_parent.rglob(GPAW_TOTAL_DENSITY_CUBE_FILENAME))
    assert not any(scratch_parent.rglob(CHARGEMOL_JOB_CONTROL_FILENAME))
    assert not any(scratch_parent.rglob(GPAW_TEXT_LOG_FILENAME))


@pytest.mark.parametrize(
    "output_payload",
    [
        pytest.param(
            {
                TEST_GPAW_OUTPUT_FIELD_ENERGY: TEST_GPAW_MOCK_SINGLE_POINT_ENERGY,
                TEST_GPAW_OUTPUT_FIELD_FORCES: [
                    TEST_GPAW_ZERO_VECTOR for _ in range(len(TEST_SI_STRUCTURE))
                ],
                TEST_GPAW_OUTPUT_FIELD_STRESS: TEST_GPAW_MOCK_SINGLE_POINT_STRESS,
                TEST_GPAW_OUTPUT_FIELD_PBC: TEST_GPAW_INVALID_PBC_VALUES,
            },
            id="non-boolean-pbc-values",
        ),
        pytest.param(
            {
                TEST_GPAW_OUTPUT_FIELD_FORCES: [
                    TEST_GPAW_ZERO_VECTOR for _ in range(len(TEST_SI_STRUCTURE))
                ],
                TEST_GPAW_OUTPUT_FIELD_STRESS: TEST_GPAW_MOCK_SINGLE_POINT_STRESS,
                TEST_GPAW_OUTPUT_FIELD_PBC: TEST_GPAW_PERIODICITY_FLAGS,
            },
            id="missing-energy",
        ),
    ],
)
def test_runner_run_single_point_raises_runtime_error_for_malformed_output_payload(
    tmp_path: Path, output_payload: dict[str, object]
) -> None:
    gpaw_env_root = tmp_path / TEST_GPAW_ENV_ROOT_DIRNAME
    runner = GpawExternalScfRunner(
        structure=deepcopy(TEST_SI_STRUCTURE),
        simulation_settings=GpawScfSimulationSettings(),
        external_runtime_config=GpawExternalInstallPathsConfig(
            gpaw_env_root=gpaw_env_root,
            python_executable_path=gpaw_env_root / GPAW_ENV_PYTHON_RELATIVE_PATH,
            gpaw_executable_path=gpaw_env_root / GPAW_ENV_GPAW_EXECUTABLE_RELATIVE_PATH,
            gpaw_setups_dir=tmp_path / TEST_GPAW_SETUPS_DIRNAME,
        ),
    )
    run_files = GpawExternalScfRunFiles(run_directory=tmp_path)
    total_density_cube_path = tmp_path / GPAW_TOTAL_DENSITY_CUBE_FILENAME
    total_density_cube_path.write_text(
        TEST_GPAW_MOCK_TOTAL_DENSITY_CUBE_TEXT, encoding=DEFAULT_TEXT_ENCODING
    )
    run_files.output_path.write_text(
        json.dumps(output_payload), encoding=DEFAULT_TEXT_ENCODING
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        TEST_GPAW_MOCK_COMMAND_ARGUMENTS, TEST_PROCESS_SUCCESS_EXIT_CODE
    )
    with (
        patch(
            "crystal.calculators.gpaw.external_runner.subprocess.run",
            return_value=completed_process,
        ),
        pytest.raises(RuntimeError),
    ):
        _ = runner.run_single_point(run_directory=tmp_path)
