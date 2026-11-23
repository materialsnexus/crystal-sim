"""Tests for batched MLIP single-point calculations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from ase.calculators.calculator import BaseCalculator, Calculator
from jobflow.core.flow import Flow
from jobflow.core.job import job
from jobflow.managers.local import run_locally

from crystal.calculators.mlip.mlip_runner import MlipRunner, run_mlip_single_point_batch
from crystal.calculators.mlip.single_point_maker import MlipSinglePointBatchMaker
from crystal.constants.mlip import MLIP_SINGLE_POINT_TASK_DOCUMENT_BLOB_STORE_NAME
from crystal.schemas.mlip import (
    MlipSinglePointSimulationSettings,
    MlipSinglePointTaskDocument,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.properties import SinglePointProperties
from tests.helpers.constants.properties import MOCK_MLIP_SINGLE_POINT_PROPERTIES
from tests.helpers.constants.structure import TEST_SI_STRUCTURE
from tests.helpers.jobflow import make_blob_jobstore


def _simulation_settings() -> MlipSinglePointSimulationSettings:
    return MlipSinglePointSimulationSettings(
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
    )


def test_run_mlip_single_point_batch_reuses_calculator_and_keeps_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_calculator = Calculator()  # type: ignore[no-untyped-call]
    outcomes = [MOCK_MLIP_SINGLE_POINT_PROPERTIES, RuntimeError("evaluation failed")]
    seen_calculators: list[BaseCalculator | None] = []

    def _mock_run_single_point(runner: MlipRunner) -> object:
        seen_calculators.append(runner.calculator)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_pes_calculator",
        Mock(return_value=shared_calculator),
    )
    monkeypatch.setattr(MlipRunner, "run_single_point", _mock_run_single_point)

    documents = run_mlip_single_point_batch(
        structures=[deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)],
        simulation_settings=_simulation_settings(),
    )

    assert seen_calculators == [shared_calculator, shared_calculator]
    assert documents[0].properties is MOCK_MLIP_SINGLE_POINT_PROPERTIES
    assert documents[0].errors == ""
    assert documents[1].properties is None
    assert documents[1].errors == "evaluation failed"


@job
def _consume_single_point_documents(
    documents: list[MlipSinglePointTaskDocument],
) -> list[str | float]:
    successful_document = documents[0]
    assert isinstance(successful_document.properties, SinglePointProperties)
    return [successful_document.properties.energy, documents[1].errors]


def test_downstream_jobs_receive_blob_restored_single_point_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    documents = [
        MlipSinglePointTaskDocument(
            properties=deepcopy(MOCK_MLIP_SINGLE_POINT_PROPERTIES)
        ),
        MlipSinglePointTaskDocument(properties=None, errors="evaluation failed"),
    ]
    monkeypatch.setattr(
        "crystal.calculators.mlip.single_point_maker.run_mlip_single_point_batch",
        Mock(return_value=documents),
    )
    maker = MlipSinglePointBatchMaker(simulation_settings=_simulation_settings())
    batch_job = maker.make(
        structures=[deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)]
    )
    consumer_job = _consume_single_point_documents(
        documents=[batch_job.output[0], batch_job.output[1]]
    )
    jobstore, _ = make_blob_jobstore(
        additional_store_name=MLIP_SINGLE_POINT_TASK_DOCUMENT_BLOB_STORE_NAME
    )

    run_locally(
        Flow([batch_job, consumer_job]),
        root_dir=tmp_path,
        create_folders=True,
        store=jobstore,
    )

    assert jobstore.get_output(consumer_job.uuid, load=True) == [
        MOCK_MLIP_SINGLE_POINT_PROPERTIES.energy,
        "evaluation failed",
    ]


def test_run_mlip_single_point_batch_does_not_load_model_for_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calculator = Mock()
    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_pes_calculator",
        load_calculator,
    )

    assert (
        run_mlip_single_point_batch(
            structures=[], simulation_settings=_simulation_settings()
        )
        == []
    )
    load_calculator.assert_not_called()


def test_run_mlip_single_point_batch_reports_model_load_failure_per_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "crystal.calculators.mlip.mlip_runner._load_mlip_pes_calculator",
        Mock(side_effect=RuntimeError("model load failed")),
    )

    documents = run_mlip_single_point_batch(
        structures=[deepcopy(TEST_SI_STRUCTURE), deepcopy(TEST_SI_STRUCTURE)],
        simulation_settings=_simulation_settings(),
    )

    assert [document.errors for document in documents] == [
        "model load failed",
        "model load failed",
    ]
    assert all(document.properties is None for document in documents)
