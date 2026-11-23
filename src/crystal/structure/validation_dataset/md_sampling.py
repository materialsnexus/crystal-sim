"""Choose which slabs and interfaces receive molecular-dynamics runs."""

from __future__ import annotations

from collections.abc import Sequence

from crystal.schemas.validation_dataset import (
    ValidationDatasetMdRunKind,
    ValidationDatasetMdRunSpec,
    ValidationDatasetMdSamplingSettings,
)


def plan_validation_dataset_md_runs(
    *,
    slab_ids: Sequence[str],
    interface_ids: Sequence[str],
    md_sampling_settings: ValidationDatasetMdSamplingSettings,
) -> list[ValidationDatasetMdRunSpec]:
    """Build a deterministic, bounded list of submitted MD attempts."""
    all_sources = [
        (ValidationDatasetMdRunKind.slab, source_id) for source_id in slab_ids
    ] + [
        (ValidationDatasetMdRunKind.interface, source_id) for source_id in interface_ids
    ]
    planned_runs: list[ValidationDatasetMdRunSpec] = []
    for source_kind, source_id in all_sources:
        for temperature_k in md_sampling_settings.temperatures_k:
            planned_runs.append(
                ValidationDatasetMdRunSpec(
                    run_id=f"{source_kind.value}:{source_id}:{temperature_k:g}K",
                    source_kind=source_kind,
                    source_id=source_id,
                    temperature_k=temperature_k,
                )
            )
            if (
                md_sampling_settings.maximum_md_runs is not None
                and len(planned_runs) >= md_sampling_settings.maximum_md_runs
            ):
                return planned_runs
    return planned_runs
