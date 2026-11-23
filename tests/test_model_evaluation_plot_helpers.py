"""Tests for model-evaluation plotting helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy.typing import NDArray
from pydantic import ValidationError

from crystal.helpers.cohesive_energy import CohesiveEnergiesCalculator
from crystal.helpers.model_evaluation_plot_helpers import (
    CHARGE_LABEL,
    COHESIVE_ENERGY_LABEL,
    FORCE_COMPONENT_LABEL,
    FORCE_NORM_LABEL,
    STRESS_COMPONENT_LABEL,
    STRESS_NORM_LABEL,
    ModelEvaluationEntry,
    ReferencePredictionEntry,
    plot_model_comparison_parity_plots,
    plot_model_comparison_residual_plots,
    plot_reference_prediction_sequence_as_parity,
    plot_reference_prediction_sequence_as_residuals,
)
from crystal.helpers.model_evaluator import (
    ModelEvaluationResult,
    ReferencePredictionPair,
)
from crystal.helpers.structure_summary import StructureSummary
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.properties import VectorMode

ISOLATED_ATOM_ENERGIES = {"H": -1.0}
REFERENCE_STRUCTURE = StructureSummary(
    atomic_numbers=np.array([1], dtype=np.int_),
    forces=np.array([(1.0, 2.0, 3.0)], dtype=np.float64),
    stress=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64),
    charges=np.array([-0.2], dtype=np.float64),
    total_energy=-3.0,
)
PREDICTED_STRUCTURE = StructureSummary(
    atomic_numbers=np.array([1], dtype=np.int_),
    forces=np.array([(1.5, 1.0, 4.0)], dtype=np.float64),
    stress=np.array([0.2, 0.1, 0.5, 0.4, 0.7, 0.3], dtype=np.float64),
    charges=np.array([-0.1], dtype=np.float64),
    total_energy=-2.0,
)


def _build_model_evaluation_result(*, model_path: Path) -> ModelEvaluationResult:
    return ModelEvaluationResult(
        model=MlipModelRef(path=model_path),
        reference_structure_summaries=(REFERENCE_STRUCTURE,),
        predicted_structure_summaries=(PREDICTED_STRUCTURE,),
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES,
    )


@pytest.fixture
def model_evaluation_entries(tmp_path: Path) -> list[ModelEvaluationEntry]:
    results = [
        _build_model_evaluation_result(model_path=tmp_path / "first-model"),
        _build_model_evaluation_result(model_path=tmp_path / "second-model"),
    ]
    return [
        ModelEvaluationEntry(result=result, label=f"display-{index}")
        for index, result in enumerate(results, start=1)
    ]


@pytest.fixture
def reference_prediction_entry() -> ReferencePredictionEntry:
    return ReferencePredictionEntry(
        values=ReferencePredictionPair(
            reference=np.array([1.0, 2.0]), predicted=np.array([1.5, 1.0])
        ),
        label="model",
    )


def test_reference_prediction_entry_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="must contain at least one value"):
        _ = ReferencePredictionEntry(
            values=ReferencePredictionPair(
                reference=np.array([], dtype=np.float64),
                predicted=np.array([], dtype=np.float64),
            ),
            label="model",
        )


@pytest.mark.parametrize("base_alpha", [None, 0.4], ids=["default", "custom"])
def test_model_evaluation_entry_adapts_alpha_to_marker_count(
    model_evaluation_entries: list[ModelEvaluationEntry], base_alpha: float | None
) -> None:
    scatter_kwargs: dict[str, Any] = {"color": "red"}
    if base_alpha is not None:
        scatter_kwargs["alpha"] = base_alpha
    entry = ModelEvaluationEntry(
        result=model_evaluation_entries[0].result,
        label="model",
        scatter_kwargs=scatter_kwargs,
    )
    expected_base_alpha = 1.0 if base_alpha is None else base_alpha

    cohesive_energy_entry = entry.to_cohesive_energy_entry()
    force_component_entry = entry.to_force_entry(
        mode=VectorMode.component, adaptive_alpha=True
    )
    stress_component_entry = entry.to_stress_entry(
        mode=VectorMode.component, adaptive_alpha=True
    )

    assert cohesive_energy_entry.scatter_kwargs == scatter_kwargs
    assert force_component_entry.scatter_kwargs["alpha"] == pytest.approx(
        expected_base_alpha / np.sqrt(3)
    )
    assert stress_component_entry.scatter_kwargs["alpha"] == pytest.approx(
        expected_base_alpha / np.sqrt(6)
    )
    assert entry.scatter_kwargs == scatter_kwargs


def test_plot_reference_prediction_sequence_as_parity_plots_values(
    reference_prediction_entry: ReferencePredictionEntry,
) -> None:
    figure, ax = plt.subplots()

    plot_reference_prediction_sequence_as_parity(
        ax, [reference_prediction_entry], property_label=FORCE_NORM_LABEL
    )

    actual_points = cast("NDArray[np.float64]", ax.collections[0].get_offsets())
    np.testing.assert_allclose(actual_points, np.array([(1.0, 1.5), (2.0, 1.0)]))
    plt.close(figure)


def test_plot_reference_prediction_sequence_as_parity_applies_formatting(
    reference_prediction_entry: ReferencePredictionEntry,
) -> None:
    figure, ax = plt.subplots()

    returned_ax = plot_reference_prediction_sequence_as_parity(
        ax, [reference_prediction_entry], property_label=FORCE_NORM_LABEL
    )

    assert returned_ax is ax
    line = ax.lines[0]
    line_x_values = cast("Sequence[float]", line.get_xdata())
    line_y_values = cast("Sequence[float]", line.get_ydata())
    np.testing.assert_allclose(line_x_values, line_y_values)
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_xlabel() == f"True {FORCE_NORM_LABEL}"
    assert ax.get_ylabel() == f"Pred {FORCE_NORM_LABEL}"
    assert ax.get_box_aspect() == 1
    assert _get_legend_labels(ax) == ["model"]
    plt.close(figure)


def test_plot_reference_prediction_sequence_as_residuals_plots_values(
    reference_prediction_entry: ReferencePredictionEntry,
) -> None:
    figure, ax = plt.subplots()

    plot_reference_prediction_sequence_as_residuals(
        ax, [reference_prediction_entry], property_label=FORCE_NORM_LABEL
    )

    actual_points = cast("NDArray[np.float64]", ax.collections[0].get_offsets())
    np.testing.assert_allclose(actual_points, np.array([(1.0, 0.5), (2.0, -1.0)]))
    plt.close(figure)


def test_plot_reference_prediction_sequence_as_residuals_applies_formatting(
    reference_prediction_entry: ReferencePredictionEntry,
) -> None:
    figure, ax = plt.subplots()

    returned_ax = plot_reference_prediction_sequence_as_residuals(
        ax, [reference_prediction_entry], property_label=FORCE_NORM_LABEL
    )

    assert returned_ax is ax
    line_y_values = cast("Sequence[float]", ax.lines[0].get_ydata())
    np.testing.assert_allclose(line_y_values, np.array([0.0, 0.0]))
    assert ax.get_xlabel() == f"True {FORCE_NORM_LABEL}"
    assert ax.get_ylabel() == f"Pred-True {FORCE_NORM_LABEL}"
    assert ax.get_box_aspect() == 1
    assert _get_legend_labels(ax) == ["model"]
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [
        plot_reference_prediction_sequence_as_parity,
        plot_reference_prediction_sequence_as_residuals,
    ],
    ids=["parity", "residual"],
)
def test_reference_prediction_sequence_plotters_apply_scatter_kwargs(
    plotter: Callable[..., Axes],
) -> None:
    figure, ax = plt.subplots()
    entry = ReferencePredictionEntry(
        values=ReferencePredictionPair(
            reference=np.array([1.0, 2.0]), predicted=np.array([1.5, 1.0])
        ),
        label="model",
        scatter_kwargs={
            "alpha": 0.25,
            "color": "red",
            "label": "ignored",
            "marker": "s",
            "s": 40,
        },
    )

    plotter(ax, [entry], property_label=FORCE_NORM_LABEL)

    collection = ax.collections[0]
    assert isinstance(collection, PathCollection)
    assert collection.get_alpha() == 0.25
    np.testing.assert_allclose(
        np.asarray(collection.get_facecolor(), dtype=float)[0, :3], [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(collection.get_sizes(), [40])
    expected_marker = (
        MarkerStyle("s").get_path().transformed(MarkerStyle("s").get_transform())
    )
    np.testing.assert_allclose(
        np.asarray(collection.get_paths()[0].vertices, dtype=float),
        np.asarray(expected_marker.vertices, dtype=float),
    )
    assert _get_legend_labels(ax) == ["model"]
    plt.close(figure)


@pytest.mark.parametrize(
    ("plotter", "expected_points"),
    [
        (
            plot_reference_prediction_sequence_as_parity,
            [(1.0, 1.5), (2.0, 1.0), (3.0, 2.0), (4.0, 5.0)],
        ),
        (
            plot_reference_prediction_sequence_as_residuals,
            [(1.0, 0.5), (2.0, -1.0), (3.0, -1.0), (4.0, 1.0)],
        ),
    ],
    ids=["parity", "residual"],
)
def test_reference_prediction_sequence_plotters_accept_multidimensional_values(
    plotter: Callable[..., Axes], expected_points: list[tuple[float, float]]
) -> None:
    figure, ax = plt.subplots()
    entry = ReferencePredictionEntry(
        values=ReferencePredictionPair(
            reference=np.array([[1.0, 2.0], [3.0, 4.0]]),
            predicted=np.array([[1.5, 1.0], [2.0, 5.0]]),
        ),
        label="model",
    )

    plotter(ax, [entry], property_label=FORCE_NORM_LABEL)

    actual_points = cast("NDArray[np.float64]", ax.collections[0].get_offsets())
    np.testing.assert_allclose(actual_points, expected_points)
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [
        plot_reference_prediction_sequence_as_parity,
        plot_reference_prediction_sequence_as_residuals,
    ],
    ids=["parity", "residual"],
)
def test_entry_plotters_reject_empty_entries(plotter: Callable[..., Axes]) -> None:
    figure, ax = plt.subplots()

    with pytest.raises(ValidationError, match="at least 1 item"):
        plotter(ax, [], property_label=CHARGE_LABEL)
    plt.close(figure)


def test_model_comparison_parity_plots_plot_expected_values(
    model_evaluation_entries: list[ModelEvaluationEntry],
) -> None:
    figure, axes = plot_model_comparison_parity_plots(model_evaluation_entries)
    cohesive_energy_calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES
    )
    expected_reference_values = [
        cohesive_energy_calculator.from_structure_summaries(
            structure_summaries=[REFERENCE_STRUCTURE]
        ),
        np.linalg.norm(REFERENCE_STRUCTURE.forces, axis=1),
        np.linalg.norm(np.stack([REFERENCE_STRUCTURE.stress]), axis=1),
        REFERENCE_STRUCTURE.charges,
    ]
    expected_prediction_values = [
        cohesive_energy_calculator.from_structure_summaries(
            structure_summaries=[PREDICTED_STRUCTURE]
        ),
        np.linalg.norm(PREDICTED_STRUCTURE.forces, axis=1),
        np.linalg.norm(np.stack([PREDICTED_STRUCTURE.stress]), axis=1),
        PREDICTED_STRUCTURE.charges,
    ]

    for ax, expected_reference, expected_prediction in zip(
        axes.ravel(), expected_reference_values, expected_prediction_values, strict=True
    ):
        actual_points = ax.collections[0].get_offsets()
        expected_points = np.column_stack((expected_reference, expected_prediction))
        np.testing.assert_allclose(actual_points, expected_points)
    plt.close(figure)


def test_model_comparison_residual_plots_plot_expected_values(
    model_evaluation_entries: list[ModelEvaluationEntry],
) -> None:
    figure, axes = plot_model_comparison_residual_plots(model_evaluation_entries)
    cohesive_energy_calculator = CohesiveEnergiesCalculator(
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES
    )
    expected_reference_values = [
        cohesive_energy_calculator.from_structure_summaries(
            structure_summaries=[REFERENCE_STRUCTURE]
        ),
        np.linalg.norm(REFERENCE_STRUCTURE.forces, axis=1),
        np.linalg.norm(np.stack([REFERENCE_STRUCTURE.stress]), axis=1),
        REFERENCE_STRUCTURE.charges,
    ]
    expected_prediction_values = [
        cohesive_energy_calculator.from_structure_summaries(
            structure_summaries=[PREDICTED_STRUCTURE]
        ),
        np.linalg.norm(PREDICTED_STRUCTURE.forces, axis=1),
        np.linalg.norm(np.stack([PREDICTED_STRUCTURE.stress]), axis=1),
        PREDICTED_STRUCTURE.charges,
    ]

    for ax, expected_reference, expected_prediction in zip(
        axes.ravel(), expected_reference_values, expected_prediction_values, strict=True
    ):
        actual_points = ax.collections[0].get_offsets()
        expected_residual = np.subtract(expected_prediction, expected_reference)
        expected_points = np.column_stack((expected_reference, expected_residual))
        np.testing.assert_allclose(actual_points, expected_points)
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_model_comparison_parity_plots, plot_model_comparison_residual_plots],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_skip_charge_panel_when_reference_charges_are_missing(
    tmp_path: Path, plotter: Callable[..., tuple[Figure, NDArray[np.object_]]]
) -> None:
    reference_without_charges = replace(REFERENCE_STRUCTURE, charges=None)
    result = ModelEvaluationResult(
        model=MlipModelRef(path=tmp_path / "model"),
        reference_structure_summaries=(reference_without_charges,),
        predicted_structure_summaries=(PREDICTED_STRUCTURE,),
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES,
    )
    entry = ModelEvaluationEntry(result=result, label="model")

    assert result.reference_structure_summaries[0].charges is None
    assert result.predicted_structure_summaries[0].charges is not None

    with pytest.warns(UserWarning, match="reference charges are missing"):
        figure, axes = plotter([entry])

    assert all(ax.collections for ax in axes.ravel()[:3])
    charge_ax = axes[1, 1]
    assert not charge_ax.collections
    assert not charge_ax.lines
    assert not charge_ax.texts
    assert charge_ax.get_xlabel() == f"True {CHARGE_LABEL}"
    assert charge_ax.get_box_aspect() == 1
    plt.close(figure)


@pytest.mark.parametrize(
    ("plotter", "y_label_prefix"),
    [
        (plot_model_comparison_parity_plots, "Pred "),
        (plot_model_comparison_residual_plots, "Pred-True "),
    ],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_apply_expected_formatting(
    model_evaluation_entries: list[ModelEvaluationEntry],
    plotter: Callable[[list[ModelEvaluationEntry]], tuple[Figure, NDArray[np.object_]]],
    y_label_prefix: str,
) -> None:
    figure, axes = plotter(model_evaluation_entries)

    assert axes.shape == (2, 2)
    assert figure is axes[0, 0].figure
    assert [ax.get_xlabel() for ax in axes.ravel()] == [
        f"True {COHESIVE_ENERGY_LABEL}",
        f"True {FORCE_NORM_LABEL}",
        f"True {STRESS_NORM_LABEL}",
        f"True {CHARGE_LABEL}",
    ]
    assert all(ax.get_ylabel().startswith(y_label_prefix) for ax in axes.ravel())
    assert all(ax.collections for ax in axes.ravel())
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_model_comparison_parity_plots, plot_model_comparison_residual_plots],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_only_show_legend_on_force_panel(
    model_evaluation_entries: list[ModelEvaluationEntry],
    plotter: Callable[..., tuple[Figure, NDArray[np.object_]]],
) -> None:
    figure, axes = plotter(model_evaluation_entries)

    assert all(len(ax.collections) == 2 for ax in axes.ravel())
    cohesive_energy_ax = axes[0, 0]
    force_ax = axes[0, 1]
    stress_ax = axes[1, 0]
    charge_ax = axes[1, 1]

    assert _get_legend_labels(cohesive_energy_ax) == []
    assert _get_legend_labels(force_ax) == ["display-1", "display-2"]
    assert _get_legend_labels(stress_ax) == []
    assert _get_legend_labels(charge_ax) == []
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_model_comparison_parity_plots, plot_model_comparison_residual_plots],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_show_rmse_panel_on_every_subplot(
    model_evaluation_entries: list[ModelEvaluationEntry],
    plotter: Callable[..., tuple[Figure, NDArray[np.object_]]],
) -> None:
    figure, axes = plotter(model_evaluation_entries)
    expected_entries_by_panel = [
        [entry.to_cohesive_energy_entry() for entry in model_evaluation_entries],
        [
            entry.to_force_entry(mode=VectorMode.norm)
            for entry in model_evaluation_entries
        ],
        [
            entry.to_stress_entry(mode=VectorMode.norm)
            for entry in model_evaluation_entries
        ],
        [entry.to_charge_entry() for entry in model_evaluation_entries],
    ]

    for ax, expected_entries in zip(
        axes.ravel(), expected_entries_by_panel, strict=True
    ):
        expected_summary = "\n".join(
            f"{entry.label} RMSE: {entry.values.rmse:.3g}" for entry in expected_entries
        )
        assert [text.get_text() for text in ax.texts] == [expected_summary]
        assert ax.texts[0].get_position() == (0.02, 0.02)
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_model_comparison_parity_plots, plot_model_comparison_residual_plots],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_can_hide_rmse_panels(
    model_evaluation_entries: list[ModelEvaluationEntry],
    plotter: Callable[..., tuple[Figure, NDArray[np.object_]]],
) -> None:
    figure, axes = plotter(model_evaluation_entries, show_rmse=False)

    assert all(not ax.texts for ax in axes.ravel())
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_model_comparison_parity_plots, plot_model_comparison_residual_plots],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_apply_entry_scatter_kwargs_to_every_panel(
    model_evaluation_entries: list[ModelEvaluationEntry],
    plotter: Callable[..., tuple[Figure, NDArray[np.object_]]],
) -> None:
    source_entry = model_evaluation_entries[0]
    styled_entry = ModelEvaluationEntry(
        result=source_entry.result,
        label="styled model",
        scatter_kwargs={"alpha": 0.3, "color": "red", "marker": "s", "s": 40},
    )

    figure, axes = plotter([styled_entry], adaptive_alpha=False)

    for ax in axes.ravel():
        collection = ax.collections[0]
        assert collection.get_alpha() == 0.3
        np.testing.assert_allclose(collection.get_facecolors()[0, :3], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(collection.get_sizes(), [40])
    assert _get_legend_labels(axes[0, 1]) == ["styled model"]
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_model_comparison_parity_plots, plot_model_comparison_residual_plots],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_can_adapt_alpha_to_each_panel_marker_count(
    model_evaluation_entries: list[ModelEvaluationEntry],
    plotter: Callable[..., tuple[Figure, NDArray[np.object_]]],
) -> None:
    source_entry = model_evaluation_entries[0]
    entry = ModelEvaluationEntry(
        result=source_entry.result, label="model", scatter_kwargs={"alpha": 0.4}
    )

    figure, axes = plotter(
        [entry], force_mode=VectorMode.component, stress_mode=VectorMode.component
    )

    expected_alphas = [0.4, 0.4 / np.sqrt(3), 0.4 / np.sqrt(6), 0.4]
    assert [ax.collections[0].get_alpha() for ax in axes.ravel()] == pytest.approx(
        expected_alphas
    )
    legend = axes[0, 1].get_legend()
    assert legend is not None
    assert [handle.get_alpha() for handle in legend.legend_handles] == [0.4]
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_model_comparison_parity_plots, plot_model_comparison_residual_plots],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_can_use_force_and_stress_components(
    model_evaluation_entries: list[ModelEvaluationEntry],
    plotter: Callable[..., tuple[Figure, NDArray[np.object_]]],
) -> None:
    figure, axes = plotter(
        model_evaluation_entries,
        force_mode=VectorMode.component,
        stress_mode=VectorMode.component,
    )

    assert axes[0, 1].get_xlabel() == f"True {FORCE_COMPONENT_LABEL}"
    assert axes[1, 0].get_xlabel() == f"True {STRESS_COMPONENT_LABEL}"
    assert len(axes[0, 1].collections[0].get_offsets()) == 3
    assert len(axes[1, 0].collections[0].get_offsets()) == 6
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_model_comparison_parity_plots, plot_model_comparison_residual_plots],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_can_use_supplied_axes(
    model_evaluation_entries: list[ModelEvaluationEntry],
    plotter: Callable[..., tuple[Figure, NDArray[np.object_]]],
) -> None:
    figure, axs = plt.subplots(nrows=2, ncols=2)

    returned_figure, returned_axs = plotter(model_evaluation_entries, axs=axs)

    assert returned_figure is figure
    assert returned_axs is axs
    assert all(ax.collections for ax in axs.ravel())
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_model_comparison_parity_plots, plot_model_comparison_residual_plots],
    ids=["parity", "residual"],
)
def test_model_comparison_plots_reject_wrong_axes_shape(
    model_evaluation_entries: list[ModelEvaluationEntry],
    plotter: Callable[..., tuple[Figure, NDArray[np.object_]]],
) -> None:
    figure, axs = plt.subplots(nrows=1, ncols=2)

    with pytest.raises(ValueError, match=r"axs must have shape \(2, 2\), got \(2,\)"):
        plotter(model_evaluation_entries, axs=axs)

    plt.close(figure)


def _get_legend_labels(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    if legend is None:
        return []
    return [text.get_text() for text in legend.get_texts()]
