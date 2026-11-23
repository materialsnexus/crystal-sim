"""Plotting helpers for comparing dataset summaries."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TypeAlias, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from numpy.typing import NDArray

from crystal.helpers.dataset import DatasetSummary
from crystal.schemas.arrays import FloatArray, IntArray
from crystal.schemas.elements import AtomicNumber, atomic_number_to_element_str
from crystal.schemas.properties import ForceComponent, StressComponent, VectorMode

Color: TypeAlias = str | tuple[float, float, float] | tuple[float, float, float, float]
HistogramBins: TypeAlias = int | FloatArray

DEFAULT_FILL_ALPHA = 0.45
DEFAULT_OUTLINE_LINEWIDTH = 2.0
DEFAULT_COMPARISON_FIGSIZE = (10.0, 8.0)
DEFAULT_HISTOGRAM_BINS = 50
GROUPED_BAR_WIDTH_FRACTION = 0.8
COUNT_DISTRIBUTION_X_AXIS_PADDING = 0.5

COHESIVE_ENERGY_AXIS_LABEL = r"$\mathrm{Cohesive\ energy}\ [\mathrm{eV}/\mathrm{atom}]$"
FORCE_NORM_AXIS_LABEL = r"$\mathrm{Force\ norm}\ [\mathrm{eV}/\mathrm{A}]$"
FORCE_COMPONENT_AXIS_LABEL = r"$\mathrm{Force\ component}\ [\mathrm{eV}/\mathrm{A}]$"
STRESS_NORM_AXIS_LABEL = r"$\mathrm{Stress\ norm}\ [\mathrm{eV}/\mathrm{A}^{3}]$"
STRESS_COMPONENT_AXIS_LABEL = (
    r"$\mathrm{Stress\ component}\ [\mathrm{eV}/\mathrm{A}^{3}]$"
)
CHARGE_AXIS_LABEL = r"$\mathrm{Charge}\ [\mathrm{e}]$"
ATOM_COUNT_AXIS_LABEL = "Number of atoms"
UNIQUE_ELEMENT_COUNT_AXIS_LABEL = "Number of unique elements"
ATOMIC_NUMBER_AXIS_LABEL = "Atomic number"
ELEMENT_AXIS_LABEL = "Element"
DENSITY_AXIS_LABEL = "Density"
COUNT_AXIS_LABEL = "Count"
FRACTION_AXIS_LABEL = "Fraction"


@dataclass(frozen=True, slots=True)
class HistogramPlotOptions:
    """Shared options for rendering histogram plots."""

    bins: HistogramBins = DEFAULT_HISTOGRAM_BINS
    density: bool = True
    show_legend: bool = True
    fill_alpha: float = DEFAULT_FILL_ALPHA


DEFAULT_HISTOGRAM_PLOT_OPTIONS = HistogramPlotOptions()


@dataclass(frozen=True, slots=True)
class HistogramEntry:
    """One histogram series plus display metadata."""

    values: FloatArray
    label: str
    color: Color | None = None


@dataclass(frozen=True, slots=True)
class CountDistributionEntry:
    """One integer count-distribution series plus display metadata."""

    values: IntArray
    label: str
    color: Color | None = None


PlotEntry: TypeAlias = HistogramEntry | CountDistributionEntry


@overload
def _prepare_plot_entries(
    entries: Sequence[HistogramEntry],
) -> tuple[list[HistogramEntry], list[str]]: ...


@overload
def _prepare_plot_entries(
    entries: Sequence[CountDistributionEntry],
) -> tuple[list[CountDistributionEntry], list[str]]: ...


def _prepare_plot_entries(
    entries: Sequence[PlotEntry],
) -> tuple[list[PlotEntry], list[str]]:
    """Return non-empty entries with colors and labels for skipped empty entries."""
    non_empty_entries: list[PlotEntry] = []
    empty_entry_labels: list[str] = []
    for entry_index, entry in enumerate(entries):
        if entry.values.size > 0:
            non_empty_entries.append(
                entry
                if entry.color is not None
                else replace(entry, color=f"C{entry_index}")
            )
        else:
            empty_entry_labels.append(entry.label)
    return non_empty_entries, empty_entry_labels


def plot_histogram_entries(
    ax: Axes,
    histogram_entries: Sequence[HistogramEntry],
    *,
    x_axis_label: str,
    histogram_plot_options: HistogramPlotOptions = DEFAULT_HISTOGRAM_PLOT_OPTIONS,
) -> Axes:
    """Plot histogram entries on the input axis.

    Empty entries are skipped with a warning. Entries without an explicit color use
    the Matplotlib default cycle color for their original input position.
    """
    valid_entries_with_colors, empty_entry_labels = _prepare_plot_entries(
        histogram_entries
    )
    if empty_entry_labels:
        warnings.warn(
            f"Skipping empty histogram data for {', '.join(empty_entry_labels)}.",
            stacklevel=2,
        )

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(
        DENSITY_AXIS_LABEL if histogram_plot_options.density else COUNT_AXIS_LABEL
    )
    if not valid_entries_with_colors:
        return ax

    shared_bins = _build_shared_histogram_bins(
        values=[entry.values for entry in valid_entries_with_colors],
        bins=histogram_plot_options.bins,
    )
    for entry in valid_entries_with_colors:
        assert entry.color is not None
        ax.hist(
            entry.values,
            bins=shared_bins,  # type: ignore[arg-type]
            density=histogram_plot_options.density,
            facecolor=to_rgba(entry.color, alpha=histogram_plot_options.fill_alpha),
            edgecolor="none",
            linewidth=0,
            label=entry.label,
        )
        ax.hist(
            entry.values,
            bins=shared_bins,  # type: ignore[arg-type]
            density=histogram_plot_options.density,
            histtype="step",
            color=entry.color,
            linewidth=DEFAULT_OUTLINE_LINEWIDTH,
        )
    if histogram_plot_options.show_legend:
        ax.legend()
    return ax


def plot_integer_count_distribution_entries(
    ax: Axes,
    count_distribution_entries: Sequence[CountDistributionEntry],
    *,
    x_axis_label: str,
    normalize: bool = True,
    fill_alpha: float = DEFAULT_FILL_ALPHA,
    include_unobserved_counts: bool = True,
    x_tick_labels: Sequence[str] | None = None,
) -> Axes:
    """Plot integer count distributions as grouped bars on an existing axis.

    Empty entries are skipped with a warning. Entries without an explicit color use
    the Matplotlib default cycle color for their original input position. If custom
    tick labels are provided, their length must match the plotted x positions.
    """
    valid_entries_with_colors, empty_entry_labels = _prepare_plot_entries(
        count_distribution_entries
    )
    if empty_entry_labels:
        warnings.warn(
            f"Skipping empty count distribution data for "
            f"{', '.join(empty_entry_labels)}.",
            stacklevel=2,
        )

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(FRACTION_AXIS_LABEL if normalize else COUNT_AXIS_LABEL)
    if not valid_entries_with_colors:
        return ax

    observed_counts = np.unique(
        [count for entry in valid_entries_with_colors for count in entry.values]
    )
    if include_unobserved_counts:
        x_positions = np.arange(observed_counts[0], observed_counts[-1] + 1)
        counts_at_x_positions = x_positions
    else:
        x_positions = np.arange(len(observed_counts), dtype=np.int_)
        counts_at_x_positions = observed_counts

    if x_tick_labels is not None and len(x_tick_labels) != len(x_positions):
        error_msg = (
            "x_tick_labels must have the same length as the plotted count values: "
            f"{len(x_tick_labels)} != {len(x_positions)}."
        )
        raise ValueError(error_msg)
    bar_width = GROUPED_BAR_WIDTH_FRACTION / len(valid_entries_with_colors)
    offset_start = -GROUPED_BAR_WIDTH_FRACTION / 2 + bar_width / 2

    for dataset_index, entry in enumerate(valid_entries_with_colors):
        entry_counts, entry_heights = np.unique(entry.values, return_counts=True)
        plotted_heights = (
            entry_heights / entry.values.size if normalize else entry_heights
        )
        height_by_count = dict(zip(entry_counts, plotted_heights, strict=True))
        heights = np.asarray(
            [height_by_count.get(count, 0.0) for count in counts_at_x_positions],
            dtype=np.float64,
        )
        assert entry.color is not None
        ax.bar(
            x_positions + offset_start + dataset_index * bar_width,
            heights,
            width=bar_width,
            facecolor=to_rgba(entry.color, alpha=fill_alpha),
            edgecolor=entry.color,
            linewidth=DEFAULT_OUTLINE_LINEWIDTH,
            label=entry.label,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        list(x_tick_labels)
        if x_tick_labels is not None
        else [str(count) for count in counts_at_x_positions]
    )
    ax.set_xlim(
        float(np.min(x_positions)) - COUNT_DISTRIBUTION_X_AXIS_PADDING,
        float(np.max(x_positions)) + COUNT_DISTRIBUTION_X_AXIS_PADDING,
    )
    ax.legend()
    return ax


def plot_dataset_comparison_cohesive_energy_histogram(
    ax: Axes,
    dataset_summaries: Sequence[DatasetSummary],
    *,
    histogram_plot_options: HistogramPlotOptions = DEFAULT_HISTOGRAM_PLOT_OPTIONS,
) -> Axes:
    """Plot cohesive-energy-per-atom histograms for dataset summaries.

    Dataset summaries without isolated atom energies are skipped with a warning.
    The x-axis label is set to the standard cohesive-energy label.
    """
    return plot_histogram_entries(
        ax,
        [
            HistogramEntry(
                values=_get_cohesive_energy_values(summary=summary), label=summary.label
            )
            for summary in dataset_summaries
        ],
        x_axis_label=COHESIVE_ENERGY_AXIS_LABEL,
        histogram_plot_options=histogram_plot_options,
    )


def plot_dataset_comparison_force_histogram(
    ax: Axes,
    dataset_summaries: Sequence[DatasetSummary],
    *,
    mode: VectorMode | str = VectorMode.norm,
    components: Sequence[ForceComponent] | None = None,
    atomic_number: AtomicNumber | None = None,
    histogram_plot_options: HistogramPlotOptions = DEFAULT_HISTOGRAM_PLOT_OPTIONS,
) -> Axes:
    """Plot force norm or force component histograms for dataset summaries.

    Use `mode=VectorMode.norm` to plot force norms. Use
    `mode=VectorMode.component` to plot force components, optionally restricted by
    `components`. Setting `components` in norm mode raises `ValueError`. Setting
    both `components` and `atomic_number` also raises `ValueError`, because those
    filters select incompatible force views.
    """
    match mode:
        case VectorMode.norm:
            component_labels = None
            x_axis_label = FORCE_NORM_AXIS_LABEL
        case VectorMode.component:
            component_labels = (
                [component.name for component in components]
                if components is not None
                else None
            )
            x_axis_label = FORCE_COMPONENT_AXIS_LABEL
        case _:
            error_msg = f"Unsupported vector mode: {mode!r}."
            raise ValueError(error_msg)

    return plot_histogram_entries(
        ax,
        [
            HistogramEntry(
                values=_get_force_values(
                    summary=summary,
                    mode=mode,
                    components=components,
                    atomic_number=atomic_number,
                ),
                label=_build_histogram_label(
                    dataset_label=summary.label,
                    atomic_number=atomic_number,
                    component_labels=component_labels,
                ),
            )
            for summary in dataset_summaries
        ],
        x_axis_label=x_axis_label,
        histogram_plot_options=histogram_plot_options,
    )


def plot_dataset_comparison_stress_histogram(
    ax: Axes,
    dataset_summaries: Sequence[DatasetSummary],
    *,
    mode: VectorMode | str = VectorMode.norm,
    components: Sequence[StressComponent] | None = None,
    histogram_plot_options: HistogramPlotOptions = DEFAULT_HISTOGRAM_PLOT_OPTIONS,
) -> Axes:
    """Plot stress norm or stress component histograms for dataset summaries.

    Use `mode=VectorMode.norm` to plot stress norms. Use
    `mode=VectorMode.component` to plot stress components, optionally restricted by
    `components`. Setting `components` in norm mode raises `ValueError`.
    """
    match mode:
        case VectorMode.norm:
            component_labels = None
            x_axis_label = STRESS_NORM_AXIS_LABEL
        case VectorMode.component:
            component_labels = (
                [component.name for component in components]
                if components is not None
                else None
            )
            x_axis_label = STRESS_COMPONENT_AXIS_LABEL
        case _:
            error_msg = f"Unsupported vector mode: {mode!r}."
            raise ValueError(error_msg)

    return plot_histogram_entries(
        ax,
        [
            HistogramEntry(
                values=_get_stress_values(
                    summary=summary, mode=mode, components=components
                ),
                label=_build_histogram_label(
                    dataset_label=summary.label, component_labels=component_labels
                ),
            )
            for summary in dataset_summaries
        ],
        x_axis_label=x_axis_label,
        histogram_plot_options=histogram_plot_options,
    )


def plot_dataset_comparison_charge_histogram(
    ax: Axes,
    dataset_summaries: Sequence[DatasetSummary],
    *,
    atomic_number: AtomicNumber | None = None,
    histogram_plot_options: HistogramPlotOptions = DEFAULT_HISTOGRAM_PLOT_OPTIONS,
) -> Axes:
    """Plot charge histograms for dataset summaries.

    Dataset summaries without charges are skipped with a warning. Set
    `atomic_number` to compare charges only for one element.
    """
    return plot_histogram_entries(
        ax,
        [
            HistogramEntry(
                values=_get_charge_values(summary=summary, atomic_number=atomic_number),
                label=_build_histogram_label(
                    dataset_label=summary.label, atomic_number=atomic_number
                ),
            )
            for summary in dataset_summaries
        ],
        x_axis_label=CHARGE_AXIS_LABEL,
        histogram_plot_options=histogram_plot_options,
    )


def plot_dataset_comparison_normalised_atom_count_distribution(
    ax: Axes,
    dataset_summaries: Sequence[DatasetSummary],
    *,
    fill_alpha: float = DEFAULT_FILL_ALPHA,
    include_unobserved_atom_counts: bool = True,
) -> Axes:
    """Plot normalised atom-count distributions as grouped bars.

    By default, missing atom counts between the minimum and maximum observed counts
    are included as zero-height bars. Set `include_unobserved_atom_counts=False` to skip those counts.
    """
    return plot_integer_count_distribution_entries(
        ax,
        count_distribution_entries=[
            CountDistributionEntry(values=summary.atom_counts, label=summary.label)
            for summary in dataset_summaries
        ],
        x_axis_label=ATOM_COUNT_AXIS_LABEL,
        fill_alpha=fill_alpha,
        include_unobserved_counts=include_unobserved_atom_counts,
    )


def plot_dataset_comparison_normalised_unique_element_count_distribution(
    ax: Axes,
    dataset_summaries: Sequence[DatasetSummary],
    *,
    fill_alpha: float = DEFAULT_FILL_ALPHA,
) -> Axes:
    """Plot normalised unique-element-count distributions as grouped bars."""
    return plot_integer_count_distribution_entries(
        ax,
        count_distribution_entries=[
            CountDistributionEntry(
                values=summary.unique_element_counts, label=summary.label
            )
            for summary in dataset_summaries
        ],
        x_axis_label=UNIQUE_ELEMENT_COUNT_AXIS_LABEL,
        fill_alpha=fill_alpha,
    )


def plot_dataset_comparison_normalised_element_counts(
    ax: Axes,
    dataset_summaries: Sequence[DatasetSummary],
    *,
    fill_alpha: float = DEFAULT_FILL_ALPHA,
    show_element_symbols: bool = True,
) -> Axes:
    """Plot normalised element atom fractions as grouped bars.

    Only observed elements are plotted. By default, tick labels show element
    symbols; set `show_element_symbols=False` to show atomic numbers instead.
    """
    observed_atomic_numbers = np.unique(
        np.asarray(
            [
                number
                for summary in dataset_summaries
                for number in summary.atomic_numbers
            ],
            dtype=np.int_,
        )
    )
    return plot_integer_count_distribution_entries(
        ax,
        count_distribution_entries=[
            CountDistributionEntry(values=summary.atomic_numbers, label=summary.label)
            for summary in dataset_summaries
        ],
        x_axis_label=(
            ELEMENT_AXIS_LABEL if show_element_symbols else ATOMIC_NUMBER_AXIS_LABEL
        ),
        normalize=True,
        fill_alpha=fill_alpha,
        include_unobserved_counts=False,
        x_tick_labels=(
            [
                atomic_number_to_element_str(atomic_number)
                for atomic_number in observed_atomic_numbers
            ]
            if show_element_symbols
            else None
        ),
    )


def plot_dataset_comparison_property_predictions(
    dataset_summaries: Sequence[DatasetSummary],
    *,
    force_mode: VectorMode | str = VectorMode.norm,
    stress_mode: VectorMode | str = VectorMode.norm,
    histogram_plot_options: HistogramPlotOptions = DEFAULT_HISTOGRAM_PLOT_OPTIONS,
    axs: NDArray[np.object_] | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Create a 2x2 figure for core dataset property comparisons.

    The panels are cohesive energy, force, stress, and charge histograms. The force
    and stress panels use the same mode rules as
    `plot_dataset_comparison_force_histogram` and
    `plot_dataset_comparison_stress_histogram`.
    """
    if axs is None:
        figure, axs = plt.subplots(nrows=2, ncols=2, figsize=DEFAULT_COMPARISON_FIGSIZE)
    elif axs.shape != (2, 2):
        error_msg = f"axs must have shape (2, 2), got {axs.shape}."
        raise ValueError(error_msg)

    cohesive_energy_ax: Axes = axs[0, 0]
    force_ax: Axes = axs[0, 1]
    stress_ax: Axes = axs[1, 0]
    charge_ax: Axes = axs[1, 1]
    legend_options = histogram_plot_options
    no_legend_options = replace(histogram_plot_options, show_legend=False)

    plot_dataset_comparison_cohesive_energy_histogram(
        cohesive_energy_ax, dataset_summaries, histogram_plot_options=no_legend_options
    )
    plot_dataset_comparison_force_histogram(
        force_ax,
        dataset_summaries,
        mode=force_mode,
        histogram_plot_options=legend_options,
    )
    plot_dataset_comparison_stress_histogram(
        stress_ax,
        dataset_summaries,
        mode=stress_mode,
        histogram_plot_options=no_legend_options,
    )
    plot_dataset_comparison_charge_histogram(
        charge_ax, dataset_summaries, histogram_plot_options=no_legend_options
    )
    figure.tight_layout()
    return figure, axs


def _get_cohesive_energy_values(*, summary: DatasetSummary) -> FloatArray:
    if summary.isolated_atom_energies is None:
        warnings.warn(
            f"Skipping cohesive energy data for {summary.label}: "
            "isolated atom energies are missing.",
            UserWarning,
            stacklevel=2,
        )
        return np.array([], dtype=np.float64)
    return summary.cohesive_energies_per_atom()


def _get_force_values(
    *,
    summary: DatasetSummary,
    mode: VectorMode | str,
    components: Sequence[ForceComponent] | None,
    atomic_number: AtomicNumber | None,
) -> FloatArray:
    match mode:
        case VectorMode.norm:
            if components is not None:
                error_msg = (
                    "components can only be set when mode is VectorMode.component."
                )
                raise ValueError(error_msg)
            if atomic_number is None:
                return summary.force_norms
            return summary.force_norms_by_element.get(
                atomic_number, np.array([], dtype=np.float64)
            )
        case VectorMode.component:
            if atomic_number is not None and components is not None:
                error_msg = "components and atomic_number cannot both be set."
                raise ValueError(error_msg)
            if atomic_number is None:
                return summary.select_force_components(components=components)
            force_components = summary.force_components_by_element.get(
                atomic_number, np.empty((0, 3), dtype=np.float64)
            )
            return force_components.ravel()
        case _:
            error_msg = f"Unsupported vector mode: {mode!r}."
            raise ValueError(error_msg)


def _get_stress_values(
    *,
    summary: DatasetSummary,
    mode: VectorMode | str,
    components: Sequence[StressComponent] | None,
) -> FloatArray:
    match mode:
        case VectorMode.norm:
            if components is not None:
                error_msg = (
                    "components can only be set when mode is VectorMode.component."
                )
                raise ValueError(error_msg)
            return summary.stress_norms
        case VectorMode.component:
            return summary.select_stress_components(components=components)
        case _:
            error_msg = f"Unsupported vector mode: {mode!r}."
            raise ValueError(error_msg)


def _get_charge_values(
    *, summary: DatasetSummary, atomic_number: AtomicNumber | None
) -> FloatArray:
    if summary.charges is None:
        warnings.warn(
            f"Skipping charge data for {summary.label}: charges are missing.",
            UserWarning,
            stacklevel=2,
        )
        return np.array([], dtype=np.float64)
    if atomic_number is None:
        return summary.charges
    charges_by_element = summary.charges_by_element
    if charges_by_element is None:
        return np.array([], dtype=np.float64)
    return charges_by_element.get(atomic_number, np.array([], dtype=np.float64))


def _build_shared_histogram_bins(
    *, values: Sequence[FloatArray], bins: HistogramBins
) -> NDArray[np.float64]:
    if not isinstance(bins, int):
        return np.asarray(bins)
    all_values = np.concatenate(values)
    minimum_value = all_values.min()
    maximum_value = all_values.max()
    if minimum_value == maximum_value:
        return np.asarray([minimum_value - 0.5, maximum_value + 0.5], dtype=np.float64)
    generated_bins: FloatArray = np.linspace(minimum_value, maximum_value, bins + 1)
    return generated_bins


def _build_histogram_label(
    *,
    dataset_label: str,
    atomic_number: AtomicNumber | None = None,
    component_labels: Sequence[str] | None = None,
) -> str:
    label_parts = [dataset_label]
    if atomic_number is not None:
        label_parts.append(atomic_number_to_element_str(atomic_number))
    if component_labels is not None:
        label_parts.append(",".join(component_labels))
    return " ".join(label_parts)
