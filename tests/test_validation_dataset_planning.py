"""Behavioral tests for validation-dataset planning models."""

from __future__ import annotations

from crystal.schemas.validation_dataset import (
    ValidationDatasetMdRunKind,
    ValidationDatasetMdSamplingSettings,
)
from crystal.structure.validation_dataset.md_sampling import (
    plan_validation_dataset_md_runs,
)


def test_md_planning_is_deterministic_and_caps_submitted_attempts() -> None:
    settings = ValidationDatasetMdSamplingSettings(
        temperatures_k=[300.0, 600.0], maximum_md_runs=3
    )

    runs = plan_validation_dataset_md_runs(
        slab_ids=["slab-a"],
        interface_ids=["interface-a", "interface-b"],
        md_sampling_settings=settings,
    )

    assert [(run.source_kind, run.source_id, run.temperature_k) for run in runs] == [
        (ValidationDatasetMdRunKind.slab, "slab-a", 300.0),
        (ValidationDatasetMdRunKind.slab, "slab-a", 600.0),
        (ValidationDatasetMdRunKind.interface, "interface-a", 300.0),
    ]
    assert [run.run_id for run in runs] == [
        "slab:slab-a:300K",
        "slab:slab-a:600K",
        "interface:interface-a:300K",
    ]


def test_md_planning_uses_default_attempt_cap() -> None:
    settings = ValidationDatasetMdSamplingSettings(temperatures_k=[300.0, 600.0])

    runs = plan_validation_dataset_md_runs(
        slab_ids=[f"slab-{index}" for index in range(51)],
        interface_ids=[],
        md_sampling_settings=settings,
    )

    assert len(runs) == 100
