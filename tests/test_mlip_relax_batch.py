"""Tests for MLIP batch relaxation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from ase.calculators.calculator import BaseCalculator, Calculator
from jobflow.core.flow import Flow
from jobflow.core.job import job
from jobflow.managers.local import run_locally
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.mlip.mlip_runner import MlipRunner, run_mlip_relax_batch
from crystal.calculators.mlip.relax_maker import MlipRelaxBatchMaker
from crystal.constants.mlip import MLIP_RELAX_TASK_DOCUMENT_BLOB_STORE_NAME
from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.mlip import MlipRelaxSimulationSettings, MlipRelaxTaskDocument
from crystal.schemas.properties import RelaxProperties
from tests.helpers.constants.mlip import TEST_MLIP_RELAX_SIMULATION_SETTINGS
from tests.helpers.constants.properties import MOCK_MLIP_RELAX_PROPERTIES
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.jobflow import make_blob_jobstore


class SharedCalculator(Calculator):
    """Typed calculator stand-in used to verify calculator reuse."""

    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]


def test_run_mlip_relax_batch_reuses_one_calculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structures = [deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)]
    shared_calculator = SharedCalculator()
    load_calculator = Mock(return_value=shared_calculator)
    seen_calculators: list[BaseCalculator | None] = []
    run_directories = [Path("relax-0"), Path("relax-1")]
    seen_output_directories: list[Path] = []

    def _mock_run_relax(
        runner: MlipRunner,
        *,
        optimizer_settings: object,
        constraints: object,
        store_trajectory: bool,
        output_directory: Path | None,
    ) -> RelaxProperties:
        assert store_trajectory is False
        assert output_directory is not None
        _ = optimizer_settings, constraints
        seen_calculators.append(runner.calculator)
        seen_output_directories.append(output_directory)
        return MOCK_MLIP_RELAX_PROPERTIES

    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_pes_calculator",
        load_calculator,
    )
    monkeypatch.setattr(MlipRunner, "run_relax", _mock_run_relax)

    outcomes = run_mlip_relax_batch(
        structures=structures,
        simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS,
        run_directories=run_directories,
    )

    assert load_calculator.call_count == 1
    assert seen_calculators == [shared_calculator, shared_calculator]
    assert seen_output_directories == run_directories
    assert len(outcomes) == 2
    assert all(isinstance(document, MlipRelaxTaskDocument) for document in outcomes)
    assert all(
        document.properties is MOCK_MLIP_RELAX_PROPERTIES for document in outcomes
    )
    assert [document.dir_name for document in outcomes] == [
        directory.as_posix() for directory in run_directories
    ]


def test_run_mlip_relax_batch_keeps_partial_failures_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes: list[RelaxProperties | Exception] = [
        MOCK_MLIP_RELAX_PROPERTIES,
        RuntimeError("second relaxation failed"),
    ]
    run_directories = [Path("relax-0"), Path("relax-1")]

    def _mock_run_relax(
        _runner: MlipRunner,
        *,
        optimizer_settings: object,
        constraints: object,
        store_trajectory: bool,
        output_directory: Path | None,
    ) -> RelaxProperties:
        assert store_trajectory is False
        assert output_directory in run_directories
        _ = optimizer_settings, constraints
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_pes_calculator",
        Mock(return_value=SharedCalculator()),
    )
    monkeypatch.setattr(MlipRunner, "run_relax", _mock_run_relax)

    results = run_mlip_relax_batch(
        structures=[deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)],
        simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS,
        run_directories=run_directories,
    )

    assert results[0].properties is MOCK_MLIP_RELAX_PROPERTIES
    assert results[0].errors == ""
    assert results[1].properties is None
    assert results[1].errors == "second relaxation failed"
    assert [document.dir_name for document in results] == [
        directory.as_posix() for directory in run_directories
    ]


def test_run_mlip_relax_batch_does_not_load_model_for_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calculator = Mock()
    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_pes_calculator",
        load_calculator,
    )

    assert (
        run_mlip_relax_batch(
            structures=[],
            simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS,
            run_directories=[],
        )
        == []
    )
    load_calculator.assert_not_called()


def test_run_mlip_relax_batch_requires_one_run_directory_per_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calculator = Mock()
    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_pes_calculator",
        load_calculator,
    )

    with pytest.raises(ValueError, match="requires one run directory per structure"):
        run_mlip_relax_batch(
            structures=[deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)],
            simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS,
            run_directories=[Path("relax-0")],
        )

    load_calculator.assert_not_called()


def test_run_mlip_relax_batch_reports_model_load_failure_per_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_pes_calculator",
        Mock(side_effect=RuntimeError("model load failed")),
    )
    run_directories = [Path("relax-0"), Path("relax-1")]

    documents = run_mlip_relax_batch(
        structures=[deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)],
        simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS,
        run_directories=run_directories,
    )

    assert [document.errors for document in documents] == [
        "model load failed",
        "model load failed",
    ]
    assert all(document.properties is None for document in documents)
    assert [document.dir_name for document in documents] == [
        directory.as_posix() for directory in run_directories
    ]


def test_batch_maker_returns_standard_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    properties = MOCK_MLIP_RELAX_PROPERTIES
    monkeypatch.setattr(
        "crystal.calculators.mlip.relax_maker.run_mlip_relax_batch",
        Mock(
            return_value=[
                MlipRelaxTaskDocument(properties=properties),
                MlipRelaxTaskDocument(properties=None, errors="relaxation failed"),
            ]
        ),
    )
    maker = MlipRelaxBatchMaker(simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS)

    job = maker.make(
        structures=[deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)]
    )

    jobstore, task_documents_store = make_blob_jobstore(
        additional_store_name=MLIP_RELAX_TASK_DOCUMENT_BLOB_STORE_NAME
    )
    run_locally(job, root_dir=tmp_path, create_folders=True, store=jobstore)
    documents = MontyDecoder().process_decoded(jobstore.get_output(job.uuid, load=True))

    assert len(documents) == 2
    assert all(isinstance(document, MlipRelaxTaskDocument) for document in documents)
    assert documents[0].errors == ""
    assert isinstance(documents[0].properties, RelaxProperties)
    assert documents[0].properties.final_energy == pytest.approx(
        properties.final_energy
    )
    assert (
        documents[0].properties.relaxed_structure.as_dict()
        == properties.relaxed_structure.as_dict()
    )
    assert documents[1].errors == "relaxation failed"
    stored_job = jobstore.docs_store.query_one({"uuid": job.uuid})
    assert stored_job is not None
    stored_output = stored_job["output"]
    assert all("blob_uuid" in document for document in stored_output)
    blob_uuids = [document["blob_uuid"] for document in stored_output]
    assert len(set(blob_uuids)) == 2
    stored_documents = list(task_documents_store.query({"job_uuid": job.uuid}))
    assert len(stored_documents) == 2
    assert {document["blob_uuid"] for document in stored_documents} == set(blob_uuids)


def test_batch_maker_creates_indexed_run_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _mock_run_mlip_relax_batch(
        *,
        structures: list[PMG_Structure],
        simulation_settings: MlipRelaxSimulationSettings,
        run_directories: list[Path],
    ) -> list[MlipRelaxTaskDocument]:
        _ = simulation_settings
        return [
            MlipRelaxTaskDocument(
                dir_name=run_directory.as_posix(), properties=MOCK_MLIP_RELAX_PROPERTIES
            )
            for _structure, run_directory in zip(
                structures, run_directories, strict=True
            )
        ]

    monkeypatch.setattr(
        "crystal.calculators.mlip.relax_maker.run_mlip_relax_batch",
        _mock_run_mlip_relax_batch,
    )
    maker = MlipRelaxBatchMaker(simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS)
    job = maker.make(
        structures=[deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)]
    )

    responses = run_locally(job, root_dir=tmp_path, create_folders=True)
    documents = responses[job.uuid][1].output
    run_directories = [Path(document.dir_name) for document in documents]

    assert [run_directory.name for run_directory in run_directories] == [
        "relax-0",
        "relax-1",
    ]
    assert all(run_directory.is_dir() for run_directory in run_directories)


def test_batch_maker_reports_output_cardinality_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "crystal.calculators.mlip.relax_maker.run_mlip_relax_batch",
        Mock(return_value=[]),
    )
    maker = MlipRelaxBatchMaker(simulation_settings=TEST_MLIP_RELAX_SIMULATION_SETTINGS)

    job = maker.make(
        structures=[deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)]
    )
    responses = run_locally(job, root_dir=tmp_path, create_folders=True)
    documents = responses[job.uuid][1].output

    assert len(documents) == 2
    assert all(document.properties is None for document in documents)
    assert all(
        document.errors
        == "MLIP relaxation batch returned 0 documents for 2 structures."
        for document in documents
    )
    run_directories = [Path(document.dir_name) for document in documents]
    assert [run_directory.name for run_directory in run_directories] == [
        "relax-0",
        "relax-1",
    ]
    assert all(run_directory.is_dir() for run_directory in run_directories)


def _successful_task_document() -> MlipRelaxTaskDocument:
    return MlipRelaxTaskDocument(properties=deepcopy(MOCK_MLIP_RELAX_PROPERTIES))


def _failed_task_document() -> MlipRelaxTaskDocument:
    return MlipRelaxTaskDocument(properties=None, errors="relaxation failed")


@job(**{MLIP_RELAX_TASK_DOCUMENT_BLOB_STORE_NAME: MlipRelaxTaskDocument})
def _produce_mlip_relax_documents() -> list[MlipRelaxTaskDocument]:
    return [_successful_task_document(), _failed_task_document()]


@job
def _consume_mlip_relax_documents(
    documents: list[MlipRelaxTaskDocument],
) -> list[str | float]:
    successful_document = documents[0]
    assert isinstance(successful_document.properties, RelaxProperties)
    return [successful_document.properties.final_energy, documents[1].errors]


def test_mlip_relax_task_document_monty_round_trip_preserves_nested_properties() -> (
    None
):
    document = _successful_task_document()

    encoded = json.dumps(document, cls=MontyEncoder)
    restored = MontyDecoder().process_decoded(json.loads(encoded))

    assert isinstance(restored, MlipRelaxTaskDocument)
    assert isinstance(restored.properties, RelaxProperties)
    assert isinstance(document.properties, RelaxProperties)
    assert restored.properties.final_energy == pytest.approx(
        document.properties.final_energy
    )
    assert (
        restored.properties.relaxed_structure.as_dict()
        == document.properties.relaxed_structure.as_dict()
    )


@run_in_temp_dir
def test_downstream_jobs_receive_restored_mlip_relax_documents() -> None:
    producer_job = _produce_mlip_relax_documents()  # type: ignore[operator]
    consumer_job = _consume_mlip_relax_documents(
        documents=[producer_job.output[0], producer_job.output[1]]
    )
    jobstore, _ = make_blob_jobstore(
        additional_store_name=MLIP_RELAX_TASK_DOCUMENT_BLOB_STORE_NAME
    )

    run_locally(
        Flow([producer_job, consumer_job]),
        root_dir=Path.cwd(),
        create_folders=True,
        store=jobstore,
    )

    assert jobstore.get_output(consumer_job.uuid, load=True) == [
        MOCK_MLIP_RELAX_PROPERTIES.final_energy,
        "relaxation failed",
    ]
