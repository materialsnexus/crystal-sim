"""Tests for dataset comparison plotting helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from crystal.helpers.dataset import DatasetSummary
from crystal.helpers.dataset_plot_helpers import (
    ATOM_COUNT_AXIS_LABEL,
    CHARGE_AXIS_LABEL,
    COHESIVE_ENERGY_AXIS_LABEL,
    COUNT_DISTRIBUTION_X_AXIS_PADDING,
    DEFAULT_FILL_ALPHA,
    DENSITY_AXIS_LABEL,
    ELEMENT_AXIS_LABEL,
    FORCE_COMPONENT_AXIS_LABEL,
    FORCE_NORM_AXIS_LABEL,
    FRACTION_AXIS_LABEL,
    STRESS_COMPONENT_AXIS_LABEL,
    STRESS_NORM_AXIS_LABEL,
    CountDistributionEntry,
    HistogramEntry,
    HistogramPlotOptions,
    plot_dataset_comparison_charge_histogram,
    plot_dataset_comparison_cohesive_energy_histogram,
    plot_dataset_comparison_force_histogram,
    plot_dataset_comparison_normalised_atom_count_distribution,
    plot_dataset_comparison_normalised_element_counts,
    plot_dataset_comparison_property_predictions,
    plot_dataset_comparison_stress_histogram,
    plot_histogram_entries,
    plot_integer_count_distribution_entries,
)
from crystal.helpers.structure_summary import StructureSummary
from crystal.schemas.properties import ForceComponent, StressComponent, VectorMode

ISOLATED_ATOM_ENERGIES = {"H": -1.0, "He": -2.0}
FIRST_DATASET_LABEL = "first"
SECOND_DATASET_LABEL = "second"
FOUR_BIN_HISTOGRAM_OPTIONS = HistogramPlotOptions(bins=4)

FIRST_STRUCTURE = StructureSummary(
    atomic_numbers=np.asarray([1, 1], dtype=np.int_),
    forces=np.asarray([(3.0, 4.0, 0.0), (0.0, 0.0, 2.0)], dtype=np.float64),
    stress=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
    charges=np.asarray([0.1, -0.1], dtype=np.float64),
    total_energy=-5.0,
)
SECOND_STRUCTURE = StructureSummary(
    atomic_numbers=np.asarray([1, 2], dtype=np.int_),
    forces=np.asarray([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float64),
    stress=np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=np.float64),
    charges=np.asarray([0.2, 0.3], dtype=np.float64),
    total_energy=-8.0,
)
STRUCTURE_WITHOUT_CHARGES = StructureSummary(
    atomic_numbers=np.asarray([1, 2], dtype=np.int_),
    forces=np.asarray([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float64),
    stress=np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=np.float64),
    charges=None,
    total_energy=-8.0,
)


@pytest.fixture
def dataset_summary() -> DatasetSummary:
    return DatasetSummary(
        label=FIRST_DATASET_LABEL,
        structure_summaries=[FIRST_STRUCTURE, SECOND_STRUCTURE],
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES,
    )


@pytest.fixture
def second_dataset_summary() -> DatasetSummary:
    return DatasetSummary(
        label=SECOND_DATASET_LABEL,
        structure_summaries=[SECOND_STRUCTURE],
        isolated_atom_energies=ISOLATED_ATOM_ENERGIES,
    )


@pytest.fixture
def dataset_summaries(
    dataset_summary: DatasetSummary, second_dataset_summary: DatasetSummary
) -> list[DatasetSummary]:
    return [dataset_summary, second_dataset_summary]


@pytest.mark.parametrize(
    ("plotter", "entries", "kwargs"),
    [
        (
            plot_histogram_entries,
            [HistogramEntry(values=np.asarray([1.0], dtype=np.float64), label="first")],
            {
                "histogram_plot_options": HistogramPlotOptions(bins=1),
                "x_axis_label": FORCE_NORM_AXIS_LABEL,
            },
        ),
        (
            plot_integer_count_distribution_entries,
            [
                CountDistributionEntry(
                    values=np.asarray([1], dtype=np.int_), label="first"
                )
            ],
            {"x_axis_label": ATOM_COUNT_AXIS_LABEL},
        ),
    ],
    ids=["histogram", "count-distribution"],
)
def test_entry_plotters_return_input_axis(
    plotter: Callable[..., Axes], entries: Sequence[object], kwargs: dict[str, object]
) -> None:
    figure, ax = plt.subplots()

    returned_ax = plotter(ax, entries, **kwargs)

    assert returned_ax is ax
    plt.close(figure)


@pytest.mark.parametrize(
    ("plotter", "entries", "kwargs", "warning_match"),
    [
        (
            plot_histogram_entries,
            [
                HistogramEntry(
                    values=np.asarray([1.0], dtype=np.float64), label="zero"
                ),
                HistogramEntry(values=np.asarray([], dtype=np.float64), label="first"),
                HistogramEntry(
                    values=np.asarray([2.0], dtype=np.float64), label="second"
                ),
            ],
            {
                "histogram_plot_options": HistogramPlotOptions(
                    bins=np.asarray([0.5, 1.5, 2.5], dtype=np.float64)
                ),
                "x_axis_label": FORCE_NORM_AXIS_LABEL,
            },
            "Skipping empty histogram data for first.",
        ),
        (
            plot_integer_count_distribution_entries,
            [
                CountDistributionEntry(
                    values=np.asarray([1], dtype=np.int_), label="zero"
                ),
                CountDistributionEntry(
                    values=np.asarray([], dtype=np.int_), label="first"
                ),
                CountDistributionEntry(
                    values=np.asarray([2], dtype=np.int_), label="second"
                ),
            ],
            {"x_axis_label": ATOM_COUNT_AXIS_LABEL},
            "Skipping empty count distribution data for first.",
        ),
    ],
    ids=["histogram", "count-distribution"],
)
def test_entry_plotters_use_input_index_default_colors_after_skipping_empty_entries(
    plotter: Callable[..., Axes],
    entries: Sequence[object],
    kwargs: dict[str, object],
    warning_match: str,
) -> None:
    figure, ax = plt.subplots()

    with pytest.warns(UserWarning, match=warning_match):
        plotter(ax, entries, **kwargs)

    bars = ax.patches
    assert bars
    filled_bars = [bar for bar in bars if bar.get_fill()]
    assert [bar.get_facecolor() for bar in filled_bars] == [
        to_rgba("C0", alpha=DEFAULT_FILL_ALPHA),
        to_rgba("C0", alpha=DEFAULT_FILL_ALPHA),
        to_rgba("C2", alpha=DEFAULT_FILL_ALPHA),
        to_rgba("C2", alpha=DEFAULT_FILL_ALPHA),
    ]
    if plotter is plot_histogram_entries:
        assert all(bar.get_edgecolor() == to_rgba("none") for bar in filled_bars)
    else:
        assert [bar.get_edgecolor() for bar in filled_bars] == [
            to_rgba("C0"),
            to_rgba("C0"),
            to_rgba("C2"),
            to_rgba("C2"),
        ]
    assert _get_legend_labels(ax) == ["zero", "second"]
    plt.close(figure)


def test_plot_integer_count_distribution_entries_uses_full_count_span_by_default() -> (
    None
):
    figure, ax = plt.subplots()

    plot_integer_count_distribution_entries(
        ax,
        [
            CountDistributionEntry(
                values=np.asarray([1, 1, 3], dtype=np.int_), label="first"
            ),
            CountDistributionEntry(
                values=np.asarray([4], dtype=np.int_), label="second"
            ),
        ],
        x_axis_label=ATOM_COUNT_AXIS_LABEL,
    )

    tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert tick_labels == ["1", "2", "3", "4"]
    assert ax.get_xlim() == (
        1 - COUNT_DISTRIBUTION_X_AXIS_PADDING,
        4 + COUNT_DISTRIBUTION_X_AXIS_PADDING,
    )

    bar_heights = _get_bar_heights(ax)
    assert bar_heights[:4] == [2 / 3, 0, 1 / 3, 0]
    assert bar_heights[4:] == [0, 0, 0, 1]
    plt.close(figure)


def test_plot_integer_count_distribution_entries_can_exclude_unobserved_counts() -> (
    None
):
    figure, ax = plt.subplots()

    plot_integer_count_distribution_entries(
        ax,
        [
            CountDistributionEntry(
                values=np.asarray([1, 1, 3], dtype=np.int_), label="first"
            ),
            CountDistributionEntry(
                values=np.asarray([4], dtype=np.int_), label="second"
            ),
        ],
        x_axis_label=ATOM_COUNT_AXIS_LABEL,
        include_unobserved_counts=False,
    )

    tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert tick_labels == ["1", "3", "4"]
    assert ax.get_xlim() == (
        0 - COUNT_DISTRIBUTION_X_AXIS_PADDING,
        2 + COUNT_DISTRIBUTION_X_AXIS_PADDING,
    )

    bar_heights = _get_bar_heights(ax)
    assert bar_heights[:3] == [2 / 3, 1 / 3, 0]
    assert bar_heights[3:] == [0, 0, 1]
    plt.close(figure)


@pytest.mark.parametrize(
    ("plotter", "entries", "x_axis_label", "warning_match", "expected_y_axis_label"),
    [
        (
            plot_histogram_entries,
            [HistogramEntry(values=np.asarray([], dtype=np.float64), label="empty")],
            FORCE_NORM_AXIS_LABEL,
            "Skipping empty histogram data",
            DENSITY_AXIS_LABEL,
        ),
        (
            plot_integer_count_distribution_entries,
            [
                CountDistributionEntry(
                    values=np.asarray([], dtype=np.int_), label="empty"
                )
            ],
            ATOM_COUNT_AXIS_LABEL,
            "Skipping empty count distribution data",
            FRACTION_AXIS_LABEL,
        ),
    ],
    ids=["histogram", "count-distribution"],
)
def test_entry_plotters_label_empty_plot(
    plotter: Callable[..., Axes],
    entries: Sequence[object],
    x_axis_label: str,
    warning_match: str,
    expected_y_axis_label: str,
) -> None:
    figure, ax = plt.subplots()

    with pytest.warns(UserWarning, match=warning_match):
        plotter(ax, entries, x_axis_label=x_axis_label)

    assert not ax.patches
    assert ax.get_xlabel() == x_axis_label
    assert ax.get_ylabel() == expected_y_axis_label
    plt.close(figure)


def test_normalised_atom_count_distribution_plots_atom_counts(
    dataset_summaries: list[DatasetSummary],
) -> None:
    figure, ax = plt.subplots()

    returned_ax = plot_dataset_comparison_normalised_atom_count_distribution(
        ax, dataset_summaries
    )

    assert returned_ax is ax
    assert ax.get_xlabel() == ATOM_COUNT_AXIS_LABEL
    assert ax.get_ylabel() == FRACTION_AXIS_LABEL
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["2"]
    assert _get_legend_labels(ax) == [FIRST_DATASET_LABEL, SECOND_DATASET_LABEL]
    assert _get_bar_heights(ax) == [1.0, 1.0]
    plt.close(figure)


def test_normalised_atom_count_distribution_can_exclude_unobserved_counts(
    dataset_summaries: list[DatasetSummary],
) -> None:
    figure, ax = plt.subplots()

    plot_dataset_comparison_normalised_atom_count_distribution(
        ax, dataset_summaries, include_unobserved_atom_counts=False
    )

    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["2"]
    assert ax.get_xlim() == (
        0 - COUNT_DISTRIBUTION_X_AXIS_PADDING,
        0 + COUNT_DISTRIBUTION_X_AXIS_PADDING,
    )
    plt.close(figure)


@pytest.mark.parametrize(
    ("plotter", "kwargs", "expected_label"),
    [
        (
            plot_dataset_comparison_charge_histogram,
            {"atomic_number": 1, "histogram_plot_options": FOUR_BIN_HISTOGRAM_OPTIONS},
            f"{FIRST_DATASET_LABEL} H",
        ),
        (
            plot_dataset_comparison_force_histogram,
            {"atomic_number": 1, "histogram_plot_options": FOUR_BIN_HISTOGRAM_OPTIONS},
            f"{FIRST_DATASET_LABEL} H",
        ),
        (
            plot_dataset_comparison_force_histogram,
            {
                "mode": VectorMode.component,
                "components": [ForceComponent.x, ForceComponent.z],
                "histogram_plot_options": FOUR_BIN_HISTOGRAM_OPTIONS,
            },
            f"{FIRST_DATASET_LABEL} x,z",
        ),
        (
            plot_dataset_comparison_stress_histogram,
            {
                "mode": VectorMode.component,
                "components": [StressComponent.xx, StressComponent.xz],
                "histogram_plot_options": FOUR_BIN_HISTOGRAM_OPTIONS,
            },
            f"{FIRST_DATASET_LABEL} xx,xz",
        ),
    ],
    ids=["charge-element", "force-element", "force-components", "stress-components"],
)
def test_histograms_generate_descriptive_legend_labels(
    dataset_summary: DatasetSummary,
    plotter: Callable[..., Axes],
    kwargs: dict[str, object],
    expected_label: str,
) -> None:
    figure, ax = plt.subplots()

    returned_ax = plotter(ax, [dataset_summary], **kwargs)

    assert returned_ax is ax
    assert ax.patches
    assert _get_legend_labels(ax) == [expected_label]
    plt.close(figure)


@pytest.mark.filterwarnings("ignore:Skipping empty histogram data:UserWarning")
def test_cohesive_energy_histogram_warns_and_skips_missing_isolated_atom_energies() -> (
    None
):
    figure, ax = plt.subplots()
    summary_without_isolated_atom_energies = DatasetSummary(
        label="missing isolated atom energies",
        structure_summaries=[FIRST_STRUCTURE, SECOND_STRUCTURE],
    )

    with pytest.warns(UserWarning, match="isolated atom energies are missing"):
        returned_ax = plot_dataset_comparison_cohesive_energy_histogram(
            ax, [summary_without_isolated_atom_energies]
        )

    assert returned_ax is ax
    assert not ax.patches
    assert ax.get_xlabel()
    plt.close(figure)


@pytest.mark.filterwarnings("ignore:Skipping empty histogram data:UserWarning")
def test_charge_histogram_warns_and_skips_missing_charges() -> None:
    figure, ax = plt.subplots()
    summary_without_charges = DatasetSummary(
        label="missing charges",
        structure_summaries=[STRUCTURE_WITHOUT_CHARGES, STRUCTURE_WITHOUT_CHARGES],
    )

    with pytest.warns(UserWarning, match="charges are missing"):
        returned_ax = plot_dataset_comparison_charge_histogram(
            ax, [summary_without_charges], atomic_number=1
        )

    assert returned_ax is ax
    assert not ax.patches
    assert ax.get_xlabel()
    plt.close(figure)


def test_force_histogram_supports_norm_and_component_modes(
    dataset_summary: DatasetSummary,
) -> None:
    element_norm_figure, element_norm_ax = plt.subplots()
    element_component_figure, element_component_ax = plt.subplots()
    xz_component_figure, xz_component_ax = plt.subplots()

    plot_dataset_comparison_force_histogram(
        element_norm_ax,
        [dataset_summary],
        mode=VectorMode.norm,
        atomic_number=1,
        histogram_plot_options=FOUR_BIN_HISTOGRAM_OPTIONS,
    )
    plot_dataset_comparison_force_histogram(
        element_component_ax,
        [dataset_summary],
        mode=VectorMode.component,
        atomic_number=1,
        histogram_plot_options=FOUR_BIN_HISTOGRAM_OPTIONS,
    )
    plot_dataset_comparison_force_histogram(
        xz_component_ax,
        [dataset_summary],
        mode=VectorMode.component,
        components=[ForceComponent.x, ForceComponent.z],
        histogram_plot_options=FOUR_BIN_HISTOGRAM_OPTIONS,
    )

    assert element_norm_ax.patches
    assert element_component_ax.patches
    assert xz_component_ax.patches

    assert element_norm_ax.get_xlabel() == FORCE_NORM_AXIS_LABEL
    assert element_component_ax.get_xlabel() == FORCE_COMPONENT_AXIS_LABEL
    assert xz_component_ax.get_xlabel() == FORCE_COMPONENT_AXIS_LABEL

    plt.close(element_norm_figure)
    plt.close(xz_component_figure)
    plt.close(element_component_figure)


def test_stress_histogram_supports_norm_and_component_modes(
    dataset_summary: DatasetSummary,
) -> None:
    norm_figure, norm_ax = plt.subplots()
    component_figure, component_ax = plt.subplots()

    plot_dataset_comparison_stress_histogram(
        norm_ax,
        [dataset_summary],
        mode=VectorMode.norm,
        histogram_plot_options=FOUR_BIN_HISTOGRAM_OPTIONS,
    )
    plot_dataset_comparison_stress_histogram(
        component_ax,
        [dataset_summary],
        mode=VectorMode.component,
        components=[StressComponent.xx, StressComponent.xz],
        histogram_plot_options=FOUR_BIN_HISTOGRAM_OPTIONS,
    )

    assert norm_ax.patches
    assert component_ax.patches

    assert norm_ax.get_xlabel() == STRESS_NORM_AXIS_LABEL
    assert component_ax.get_xlabel() == STRESS_COMPONENT_AXIS_LABEL

    plt.close(norm_figure)
    plt.close(component_figure)


@pytest.mark.parametrize(
    ("plotter", "components"),
    [
        (plot_dataset_comparison_force_histogram, [ForceComponent.x]),
        (plot_dataset_comparison_stress_histogram, [StressComponent.xx]),
    ],
    ids=["force", "stress"],
)
def test_vector_histograms_reject_components_in_norm_mode(
    dataset_summary: DatasetSummary,
    plotter: Callable[..., Axes],
    components: Sequence[object],
) -> None:
    figure, ax = plt.subplots()

    with pytest.raises(
        ValueError,
        match=r"components can only be set when mode is VectorMode.component",
    ):
        plotter(ax, [dataset_summary], mode=VectorMode.norm, components=components)

    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [plot_dataset_comparison_force_histogram, plot_dataset_comparison_stress_histogram],
    ids=["force", "stress"],
)
def test_vector_histograms_reject_unsupported_modes(
    dataset_summary: DatasetSummary, plotter: Callable[..., Axes]
) -> None:
    figure, ax = plt.subplots()

    with pytest.raises(ValueError, match=r"Unsupported vector mode"):
        plotter(ax, [dataset_summary], mode="unsupported")

    plt.close(figure)


def test_force_histogram_rejects_component_selection_with_element_filter(
    dataset_summary: DatasetSummary,
) -> None:
    figure, ax = plt.subplots()

    with pytest.raises(
        ValueError, match=r"components and atomic_number cannot both be set."
    ):
        plot_dataset_comparison_force_histogram(
            ax,
            [dataset_summary],
            mode=VectorMode.component,
            components=[ForceComponent.x],
            atomic_number=1,
        )

    plt.close(figure)


def test_normalised_element_count_plot_shows_observed_element_symbols_by_default(
    dataset_summaries: list[DatasetSummary],
) -> None:
    figure, ax = plt.subplots()

    plot_dataset_comparison_normalised_element_counts(ax, dataset_summaries)

    assert ax.get_xlabel() == ELEMENT_AXIS_LABEL
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["H", "He"]
    assert ax.get_xlim() == (
        0 - COUNT_DISTRIBUTION_X_AXIS_PADDING,
        1 + COUNT_DISTRIBUTION_X_AXIS_PADDING,
    )
    bar_heights = _get_bar_heights(ax)
    assert bar_heights[:2] == [0.75, 0.25]
    assert bar_heights[2:] == [0.5, 0.5]
    plt.close(figure)


def test_normalised_element_count_plot_can_show_observed_atomic_numbers(
    dataset_summaries: list[DatasetSummary],
) -> None:
    figure, ax = plt.subplots()

    plot_dataset_comparison_normalised_element_counts(
        ax, dataset_summaries, show_element_symbols=False
    )

    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["1", "2"]
    plt.close(figure)


def test_property_predictions_creates_expected_histogram_grid(
    dataset_summaries: list[DatasetSummary],
) -> None:
    figure, axes = plot_dataset_comparison_property_predictions(
        dataset_summaries, histogram_plot_options=FOUR_BIN_HISTOGRAM_OPTIONS
    )

    assert axes.shape == (2, 2)
    assert figure is axes[0, 0].figure

    cohesive_energy_ax = axes[0, 0]
    force_ax = axes[0, 1]
    stress_ax = axes[1, 0]
    charge_ax = axes[1, 1]

    assert cohesive_energy_ax.get_xlabel() == COHESIVE_ENERGY_AXIS_LABEL
    assert force_ax.get_xlabel() == FORCE_NORM_AXIS_LABEL
    assert stress_ax.get_xlabel() == STRESS_NORM_AXIS_LABEL
    assert charge_ax.get_xlabel() == CHARGE_AXIS_LABEL

    assert cohesive_energy_ax.patches
    assert force_ax.patches
    assert stress_ax.patches
    assert charge_ax.patches

    assert _get_legend_labels(cohesive_energy_ax) == []
    assert _get_legend_labels(force_ax) == [FIRST_DATASET_LABEL, SECOND_DATASET_LABEL]
    assert _get_legend_labels(stress_ax) == []
    assert _get_legend_labels(charge_ax) == []
    plt.close(figure)


def test_property_predictions_rejects_wrong_axes_shape(
    dataset_summaries: list[DatasetSummary],
) -> None:
    figure, axs = plt.subplots(nrows=1, ncols=2)

    with pytest.raises(ValueError, match=r"axs must have shape \(2, 2\), got \(2,\)"):
        plot_dataset_comparison_property_predictions(dataset_summaries, axs=axs)

    plt.close(figure)


def _get_legend_labels(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    if legend is None:
        return []
    return [text.get_text() for text in legend.get_texts()]


def _get_bar_heights(ax: Axes) -> list[float]:
    heights = []
    for patch in ax.patches:
        assert isinstance(patch, Rectangle)
        heights.append(patch.get_height())
    return heights
