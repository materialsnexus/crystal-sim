"""Tests for scratch execution behavior."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from crystal.constants.gpaw import GPAW_TEXT_LOG_FILENAME
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.helpers.scratch_execution import (
    ScratchDirectoryUnavailableError,
    local_scratch_execution_directory,
)
from crystal.schemas.scratch_execution import (
    SCRATCH_EXECUTION_MANIFEST_FILENAME,
    SLURM_JOB_ID_ENVIRONMENT_VARIABLE,
    SLURM_NODELIST_ENVIRONMENT_VARIABLE,
    SLURM_TMPDIR_ENVIRONMENT_VARIABLE,
    TMPDIR_ENVIRONMENT_VARIABLE,
    LocalScratchPolicy,
    ScratchExecutionStatus,
)
from tests.helpers.constants.scratch_execution import (
    TEST_FAILURE_MESSAGE,
    TEST_LOG_TEXT,
    TEST_MANIFEST_FINAL_DIRECTORY_KEY,
    TEST_MANIFEST_HOSTNAME_KEY,
    TEST_MANIFEST_SCRATCH_DIRECTORY_KEY,
    TEST_MANIFEST_SLURM_JOB_ID_KEY,
    TEST_MANIFEST_SLURM_NODE_NAME_KEY,
    TEST_MANIFEST_STATUS_KEY,
    TEST_PARTIAL_RESULT_FILENAME,
    TEST_PARTIAL_RESULT_TEXT,
    TEST_RESULT_FILENAME,
    TEST_RESULT_TEXT,
    TEST_SLURM_JOB_ID,
    TEST_SLURM_NODE_NAME,
    TEST_TMPDIR_NODE_NAME,
)


def test_local_scratch_policy_rejects_removed_scratch_env_keys_option() -> None:
    with pytest.raises(ValidationError):
        LocalScratchPolicy.model_validate({"scratch_env_keys": ("TMPDIR",)})


def test_persistent_execution_returns_no_manifest_and_writes_no_manifest_file(
    tmp_path: Path,
) -> None:
    final_directory = tmp_path / "final"

    with local_scratch_execution_directory(
        policy=LocalScratchPolicy(),
        destination_directory=final_directory,
        result_filenames=(TEST_RESULT_FILENAME,),
    ) as execution_directory:
        (execution_directory / TEST_RESULT_FILENAME).write_text(
            TEST_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )

    assert (final_directory / TEST_RESULT_FILENAME).is_file()
    assert not (final_directory / SCRATCH_EXECUTION_MANIFEST_FILENAME).exists()


def test_persistent_execution_accepts_destination_directory_keyword(
    tmp_path: Path,
) -> None:
    destination_directory = tmp_path / "destination"

    with local_scratch_execution_directory(
        policy=LocalScratchPolicy(),
        destination_directory=destination_directory,
        result_filenames=(TEST_RESULT_FILENAME,),
    ) as execution_directory:
        (execution_directory / TEST_RESULT_FILENAME).write_text(
            TEST_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )

    assert (destination_directory / TEST_RESULT_FILENAME).is_file()
    assert not (destination_directory / SCRATCH_EXECUTION_MANIFEST_FILENAME).exists()


def test_persistent_execution_rejects_legacy_final_directory_keyword(
    tmp_path: Path,
) -> None:
    legacy_final_directory = tmp_path / "final"

    with pytest.raises(TypeError, match="final_directory"):
        local_scratch_execution_directory(
            policy=LocalScratchPolicy(),
            final_directory=legacy_final_directory,  # type: ignore[call-arg]
            result_filenames=(TEST_RESULT_FILENAME,),
        )


def test_enabled_local_scratch_policy_requires_scratch_env_or_explicit_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(SLURM_TMPDIR_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(TMPDIR_ENVIRONMENT_VARIABLE, raising=False)
    final_directory = tmp_path / "final"
    final_directory.mkdir()

    with (
        pytest.raises(ScratchDirectoryUnavailableError),
        local_scratch_execution_directory(
            policy=LocalScratchPolicy(enabled=True),
            destination_directory=final_directory,
            result_filenames=(),
        ),
    ):
        pass


def test_scratch_run_moves_requested_results_and_gpaw_log_to_final_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch_parent = tmp_path / "slurm_tmp"
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    monkeypatch.setenv(SLURM_TMPDIR_ENVIRONMENT_VARIABLE, scratch_parent.as_posix())
    monkeypatch.setenv(SLURM_JOB_ID_ENVIRONMENT_VARIABLE, TEST_SLURM_JOB_ID)
    monkeypatch.setenv(SLURM_NODELIST_ENVIRONMENT_VARIABLE, TEST_SLURM_NODE_NAME)
    policy = LocalScratchPolicy(enabled=True)

    with local_scratch_execution_directory(
        policy=policy,
        destination_directory=final_directory,
        result_filenames=(TEST_RESULT_FILENAME, GPAW_TEXT_LOG_FILENAME),
    ) as execution_directory:
        assert execution_directory != final_directory
        assert execution_directory.parent == scratch_parent
        (execution_directory / TEST_RESULT_FILENAME).write_text(
            TEST_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )
        (execution_directory / GPAW_TEXT_LOG_FILENAME).write_text(
            TEST_LOG_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )
        scratch_directory = execution_directory

    manifest_data = json.loads(
        (final_directory / SCRATCH_EXECUTION_MANIFEST_FILENAME).read_text(
            encoding=DEFAULT_TEXT_ENCODING
        )
    )
    assert (final_directory / TEST_RESULT_FILENAME).read_text(
        encoding=DEFAULT_TEXT_ENCODING
    ) == TEST_RESULT_TEXT
    assert (final_directory / GPAW_TEXT_LOG_FILENAME).read_text(
        encoding=DEFAULT_TEXT_ENCODING
    ) == TEST_LOG_TEXT
    assert manifest_data[TEST_MANIFEST_STATUS_KEY] == ScratchExecutionStatus.success
    assert (
        manifest_data[TEST_MANIFEST_FINAL_DIRECTORY_KEY] == final_directory.as_posix()
    )
    assert Path(manifest_data[TEST_MANIFEST_SCRATCH_DIRECTORY_KEY]) == scratch_directory
    assert not scratch_directory.exists()


def test_scratch_run_uses_tmpdir_when_slurm_tmpdir_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch_parent = tmp_path / "tmpdir"
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    monkeypatch.delenv(SLURM_TMPDIR_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setenv(TMPDIR_ENVIRONMENT_VARIABLE, scratch_parent.as_posix())
    monkeypatch.setenv(SLURM_NODELIST_ENVIRONMENT_VARIABLE, TEST_TMPDIR_NODE_NAME)

    with local_scratch_execution_directory(
        policy=LocalScratchPolicy(enabled=True),
        destination_directory=final_directory,
        result_filenames=(TEST_RESULT_FILENAME,),
    ) as execution_directory:
        assert execution_directory.parent == scratch_parent
        (execution_directory / TEST_RESULT_FILENAME).write_text(
            TEST_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )
        scratch_directory = execution_directory

    assert (final_directory / TEST_RESULT_FILENAME).is_file()
    assert not scratch_directory.exists()


def test_scratch_run_uses_explicit_fallback_parent_when_env_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch_parent = tmp_path / "configured_scratch"
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    monkeypatch.delenv(SLURM_TMPDIR_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(TMPDIR_ENVIRONMENT_VARIABLE, raising=False)
    policy = LocalScratchPolicy(enabled=True, fallback_scratch_parent=scratch_parent)

    with local_scratch_execution_directory(
        policy=policy,
        destination_directory=final_directory,
        result_filenames=(TEST_RESULT_FILENAME,),
    ) as execution_directory:
        assert execution_directory.parent == scratch_parent
        (execution_directory / TEST_RESULT_FILENAME).write_text(
            TEST_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )

    assert (final_directory / TEST_RESULT_FILENAME).is_file()


def test_scratch_run_prefers_slurm_tmpdir_over_tmpdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    slurm_scratch_parent = tmp_path / "slurm_tmp"
    tmpdir_scratch_parent = tmp_path / "tmpdir"
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    monkeypatch.setenv(
        SLURM_TMPDIR_ENVIRONMENT_VARIABLE, slurm_scratch_parent.as_posix()
    )
    monkeypatch.setenv(TMPDIR_ENVIRONMENT_VARIABLE, tmpdir_scratch_parent.as_posix())

    with local_scratch_execution_directory(
        policy=LocalScratchPolicy(enabled=True),
        destination_directory=final_directory,
        result_filenames=(TEST_RESULT_FILENAME,),
    ) as execution_directory:
        assert execution_directory.parent == slurm_scratch_parent
        (execution_directory / TEST_RESULT_FILENAME).write_text(
            TEST_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )

    assert (final_directory / TEST_RESULT_FILENAME).is_file()


def test_failed_scratch_run_moves_partial_results_and_records_machine_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch_parent = tmp_path / "slurm_tmp"
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    monkeypatch.setenv(SLURM_TMPDIR_ENVIRONMENT_VARIABLE, scratch_parent.as_posix())
    monkeypatch.setenv(SLURM_JOB_ID_ENVIRONMENT_VARIABLE, TEST_SLURM_JOB_ID)
    monkeypatch.setenv(SLURM_NODELIST_ENVIRONMENT_VARIABLE, TEST_SLURM_NODE_NAME)
    policy = LocalScratchPolicy(enabled=True)
    scratch_directory: Path | None = None

    def _run_failing_scratch_calculation() -> None:
        nonlocal scratch_directory
        with local_scratch_execution_directory(
            policy=policy,
            destination_directory=final_directory,
            result_filenames=(TEST_PARTIAL_RESULT_FILENAME, GPAW_TEXT_LOG_FILENAME),
        ) as execution_directory:
            scratch_directory = execution_directory
            (execution_directory / TEST_PARTIAL_RESULT_FILENAME).write_text(
                TEST_PARTIAL_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING
            )
            (execution_directory / GPAW_TEXT_LOG_FILENAME).write_text(
                TEST_LOG_TEXT, encoding=DEFAULT_TEXT_ENCODING
            )
            raise RuntimeError(TEST_FAILURE_MESSAGE)

    with pytest.raises(RuntimeError, match=TEST_FAILURE_MESSAGE):
        _run_failing_scratch_calculation()

    assert scratch_directory is not None
    manifest_data = json.loads(
        (final_directory / SCRATCH_EXECUTION_MANIFEST_FILENAME).read_text(
            encoding=DEFAULT_TEXT_ENCODING
        )
    )
    assert Path(manifest_data[TEST_MANIFEST_SCRATCH_DIRECTORY_KEY]) == scratch_directory
    assert (
        manifest_data[TEST_MANIFEST_FINAL_DIRECTORY_KEY] == final_directory.as_posix()
    )
    assert manifest_data[TEST_MANIFEST_HOSTNAME_KEY] == socket.gethostname()
    assert manifest_data[TEST_MANIFEST_SLURM_JOB_ID_KEY] == TEST_SLURM_JOB_ID
    assert manifest_data[TEST_MANIFEST_SLURM_NODE_NAME_KEY] == TEST_SLURM_NODE_NAME
    assert manifest_data[TEST_MANIFEST_STATUS_KEY] == ScratchExecutionStatus.failed
    assert (final_directory / TEST_PARTIAL_RESULT_FILENAME).read_text(
        encoding=DEFAULT_TEXT_ENCODING
    ) == TEST_PARTIAL_RESULT_TEXT
    assert (final_directory / GPAW_TEXT_LOG_FILENAME).read_text(
        encoding=DEFAULT_TEXT_ENCODING
    ) == TEST_LOG_TEXT
    assert scratch_directory.is_dir()
    assert not (scratch_directory / TEST_PARTIAL_RESULT_FILENAME).exists()
    assert not (scratch_directory / GPAW_TEXT_LOG_FILENAME).exists()


def test_scratch_run_rewrites_requested_text_paths_to_final_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch_parent = tmp_path / "slurm_tmp"
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    monkeypatch.setenv(SLURM_TMPDIR_ENVIRONMENT_VARIABLE, scratch_parent.as_posix())
    policy = LocalScratchPolicy(enabled=True)

    with local_scratch_execution_directory(
        policy=policy,
        destination_directory=final_directory,
        result_filenames=(TEST_RESULT_FILENAME,),
        text_result_filenames_replacing_execution_directory=(TEST_RESULT_FILENAME,),
    ) as execution_directory:
        (execution_directory / TEST_PARTIAL_RESULT_FILENAME).write_text(
            TEST_PARTIAL_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING
        )
        (execution_directory / TEST_RESULT_FILENAME).write_text(
            f"path={execution_directory.as_posix()}", encoding=DEFAULT_TEXT_ENCODING
        )

    assert (final_directory / TEST_RESULT_FILENAME).read_text(
        encoding=DEFAULT_TEXT_ENCODING
    ) == f"path={final_directory.as_posix()}"


def test_scratch_move_failure_preserves_existing_final_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch_parent = tmp_path / "slurm_tmp"
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    existing_result_path = final_directory / TEST_RESULT_FILENAME
    existing_result_path.write_text(TEST_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING)
    monkeypatch.setenv(SLURM_TMPDIR_ENVIRONMENT_VARIABLE, scratch_parent.as_posix())
    policy = LocalScratchPolicy(enabled=True)

    def _run_calculation_with_directory_result() -> None:
        with local_scratch_execution_directory(
            policy=policy,
            destination_directory=final_directory,
            result_filenames=(TEST_RESULT_FILENAME,),
        ) as execution_directory:
            (execution_directory / TEST_RESULT_FILENAME).mkdir()

    with pytest.raises(IsADirectoryError):
        _run_calculation_with_directory_result()

    assert (
        existing_result_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
        == TEST_RESULT_TEXT
    )
    manifest_path = final_directory / SCRATCH_EXECUTION_MANIFEST_FILENAME
    assert manifest_path.is_file()
    manifest_data = json.loads(manifest_path.read_text(encoding=DEFAULT_TEXT_ENCODING))
    assert manifest_data[TEST_MANIFEST_STATUS_KEY] == ScratchExecutionStatus.failed


def test_scratch_move_failure_preserves_scratch_directory_for_debugging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch_parent = tmp_path / "slurm_tmp"
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    existing_result_path = final_directory / TEST_RESULT_FILENAME
    existing_result_path.write_text(TEST_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING)
    monkeypatch.setenv(SLURM_TMPDIR_ENVIRONMENT_VARIABLE, scratch_parent.as_posix())
    policy = LocalScratchPolicy(enabled=True)
    scratch_directory: Path | None = None

    def _run_calculation_with_directory_result() -> None:
        nonlocal scratch_directory
        with local_scratch_execution_directory(
            policy=policy,
            destination_directory=final_directory,
            result_filenames=(TEST_RESULT_FILENAME,),
        ) as execution_directory:
            scratch_directory = execution_directory
            (execution_directory / TEST_RESULT_FILENAME).mkdir()

    with pytest.raises(IsADirectoryError):
        _run_calculation_with_directory_result()

    assert scratch_directory is not None
    assert (
        existing_result_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
        == TEST_RESULT_TEXT
    )
    assert scratch_directory.is_dir()


def test_failed_calculation_and_move_failure_still_writes_failure_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch_parent = tmp_path / "slurm_tmp"
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    existing_result_path = final_directory / TEST_RESULT_FILENAME
    existing_result_path.write_text(TEST_RESULT_TEXT, encoding=DEFAULT_TEXT_ENCODING)
    monkeypatch.setenv(SLURM_TMPDIR_ENVIRONMENT_VARIABLE, scratch_parent.as_posix())
    policy = LocalScratchPolicy(enabled=True)

    def _run_failing_calculation_with_directory_result() -> None:
        with local_scratch_execution_directory(
            policy=policy,
            destination_directory=final_directory,
            result_filenames=(TEST_RESULT_FILENAME,),
        ) as execution_directory:
            (execution_directory / TEST_RESULT_FILENAME).mkdir()
            raise RuntimeError(TEST_FAILURE_MESSAGE)

    with pytest.raises(RuntimeError, match=TEST_FAILURE_MESSAGE):
        _run_failing_calculation_with_directory_result()

    manifest_path = final_directory / SCRATCH_EXECUTION_MANIFEST_FILENAME
    assert manifest_path.is_file()
    manifest_data = json.loads(manifest_path.read_text(encoding=DEFAULT_TEXT_ENCODING))
    assert manifest_data[TEST_MANIFEST_STATUS_KEY] == ScratchExecutionStatus.failed
    assert TEST_FAILURE_MESSAGE in (manifest_data.get("error_message") or "")
