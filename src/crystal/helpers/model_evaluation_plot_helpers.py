"""Plotting helpers for MLIP model-evaluation results."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Annotated, Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, validate_call

from crystal.helpers.model_evaluator import (
    ModelEvaluationResult,
    ReferencePredictionPair,
)
from crystal.schemas.properties import VectorMode

DEFAULT_EVALUATION_FIGSIZE = (10.0, 10.0)
DEFAULT_REFERENCE_LINE_COLOR = "black"
DEFAULT_REFERENCE_LINE_STYLE = "--"
MIN_SCATTER_ALPHA = 0.05
SCATTER_ALPHA_KWARG = "alpha"

COHESIVE_ENERGY_LABEL = r"$\mathrm{Cohesive\ energy}\ [\mathrm{eV}/\mathrm{atom}]$"
FORCE_NORM_LABEL = r"$\mathrm{Force\ norm}\ [\mathrm{eV}/\mathrm{A}]$"
FORCE_COMPONENT_LABEL = r"$\mathrm{Force\ component}\ [\mathrm{eV}/\mathrm{A}]$"
STRESS_NORM_LABEL = r"$\mathrm{Stress\ norm}\ [\mathrm{eV}/\mathrm{A}^{3}]$"
STRESS_COMPONENT_LABEL = r"$\mathrm{Stress\ component}\ [\mathrm{eV}/\mathrm{A}^{3}]$"
CHARGE_LABEL = r"$\mathrm{Charge}\ [\mathrm{e}]$"


@dataclass(frozen=True, slots=True)
class ReferencePredictionEntry:
    """One paired reference/prediction series plus its label and `Axes.scatter` keyword arguments."""

    values: ReferencePredictionPair
    label: str
    scatter_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the paired values are non-empty."""
        if self.values.reference.size == 0:
            error_msg = "Plot entry values must contain at least one value."
            raise ValueError(error_msg)


@dataclass(frozen=True, slots=True)
class ModelEvaluationEntry:
    """One model-evaluation result plus scatter-plot display metadata."""

    result: ModelEvaluationResult
    label: str
    scatter_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_cohesive_energy_entry(self) -> ReferencePredictionEntry:
        """Return a styled cohesive-energy reference/prediction entry."""
        return self._to_reference_prediction_entry(
            values=self.result.cohesive_energies_per_atom
        )

    def to_force_entry(
        self, *, mode: VectorMode, adaptive_alpha: bool = False
    ) -> ReferencePredictionEntry:
        """Return a styled force reference/prediction entry in the requested mode."""
        match mode:
            case VectorMode.norm:
                values = self.result.force_norms
            case VectorMode.component:
                values = self.result.forces
        return self._to_reference_prediction_entry(
            values=values, adaptive_alpha=adaptive_alpha
        )

    def to_stress_entry(
        self, *, mode: VectorMode, adaptive_alpha: bool = False
    ) -> ReferencePredictionEntry:
        """Return a styled stress reference/prediction entry in the requested mode."""
        match mode:
            case VectorMode.norm:
                values = self.result.stress_norms
            case VectorMode.component:
                values = self.result.stress
        return self._to_reference_prediction_entry(
            values=values, adaptive_alpha=adaptive_alpha
        )

    def to_charge_entry(
        self, *, adaptive_alpha: bool = False
    ) -> ReferencePredictionEntry:
        """Return a styled charge reference/prediction entry."""
        return self._to_reference_prediction_entry(
            values=self.result.charges, adaptive_alpha=adaptive_alpha
        )

    def _to_reference_prediction_entry(
        self, *, values: ReferencePredictionPair, adaptive_alpha: bool = False
    ) -> ReferencePredictionEntry:
        if adaptive_alpha:
            scatter_kwargs = self._scatter_kwargs_with_scaled_alpha(
                marker_count=values.reference.size
            )
        else:
            scatter_kwargs = dict(self.scatter_kwargs)

        return ReferencePredictionEntry(
            values=values, label=self.label, scatter_kwargs=scatter_kwargs
        )

    def _scatter_kwargs_with_scaled_alpha(self, *, marker_count: int) -> dict[str, Any]:
        scaled_scatter_kwargs = dict(self.scatter_kwargs)
        baseline_marker_count = len(self.result.reference_structure_summaries)
        if baseline_marker_count == 0:
            return scaled_scatter_kwargs

        base_alpha = float(scaled_scatter_kwargs.get(SCATTER_ALPHA_KWARG, 1.0))
        marker_ratio = max(1.0, marker_count / baseline_marker_count)
        scaled_alpha = base_alpha / sqrt(marker_ratio)
        scaled_scatter_kwargs[SCATTER_ALPHA_KWARG] = max(
            MIN_SCATTER_ALPHA, scaled_alpha
        )
        return scaled_scatter_kwargs


ReferencePredictionEntrySequence: TypeAlias = Annotated[
    Sequence[ReferencePredictionEntry], Field(min_length=1)
]
ModelEvaluationEntrySequence: TypeAlias = Annotated[
    Sequence[ModelEvaluationEntry], Field(min_length=1)
]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_reference_prediction_sequence_as_parity(
    ax: Axes,
    entries: ReferencePredictionEntrySequence,
    *,
    property_label: str,
    show_legend: bool = True,
    show_rmse: bool = False,
) -> Axes:
    """Plot predicted values against references on an existing axis."""
    for entry in entries:
        scatter_kwargs = {**entry.scatter_kwargs, "label": entry.label}
        ax.scatter(
            entry.values.reference.ravel(),
            entry.values.predicted.ravel(),
            **scatter_kwargs,
        )
    if show_legend:
        ax.legend()
    if show_rmse:
        _add_rmse_panel(ax=ax, entries=entries)
    ax.set_xlabel(f"True {property_label}")
    ax.set_ylabel(f"Pred {property_label}")

    lower_limit, upper_limit = _get_parity_limits(ax=ax)
    ax.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        color=DEFAULT_REFERENCE_LINE_COLOR,
        linestyle=DEFAULT_REFERENCE_LINE_STYLE,
    )
    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_box_aspect(1)
    return ax


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_reference_prediction_sequence_as_residuals(
    ax: Axes,
    entries: ReferencePredictionEntrySequence,
    *,
    property_label: str,
    show_legend: bool = True,
    show_rmse: bool = False,
) -> Axes:
    """Plot signed prediction errors against references on an existing axis."""
    for entry in entries:
        scatter_kwargs = {**entry.scatter_kwargs, "label": entry.label}
        ax.scatter(
            entry.values.reference.ravel(),
            entry.values.difference.ravel(),
            **scatter_kwargs,
        )
    if show_legend:
        ax.legend()
    if show_rmse:
        _add_rmse_panel(ax=ax, entries=entries)
    ax.set_xlabel(f"True {property_label}")
    ax.set_ylabel(f"Pred-True {property_label}")
    ax.axhline(
        0.0, color=DEFAULT_REFERENCE_LINE_COLOR, linestyle=DEFAULT_REFERENCE_LINE_STYLE
    )
    ax.set_box_aspect(1)
    return ax


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, strict=True))
def plot_model_comparison_parity_plots(
    entries: ModelEvaluationEntrySequence,
    *,
    force_mode: VectorMode = VectorMode.norm,
    stress_mode: VectorMode = VectorMode.norm,
    adaptive_alpha: bool = True,
    show_rmse: bool = True,
    axs: NDArray[np.object_] | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Create a 2x2 figure containing all model-comparison parity plots.

    The panels show cohesive energy per atom, forces, stresses, and charges in
    row-major order. Forces and stresses default to norms and can instead show
    components by setting their respective modes to `VectorMode.component`. Pass
    `axs` to plot on an existing 2x2 axes array. Each entry's label and scatter
    keyword arguments are applied consistently across all four panels. The charge
    panel is skipped if any model entry's reference structures lack charges. Set
    `adaptive_alpha=False` to preserve each entry's original opacity across panels
    and `show_rmse=False` to hide the lower-left RMSE summaries.
    """
    match force_mode:
        case VectorMode.norm:
            force_label = FORCE_NORM_LABEL
        case VectorMode.component:
            force_label = FORCE_COMPONENT_LABEL

    match stress_mode:
        case VectorMode.norm:
            stress_label = STRESS_NORM_LABEL
        case VectorMode.component:
            stress_label = STRESS_COMPONENT_LABEL

    if axs is None:
        figure, axs = plt.subplots(nrows=2, ncols=2, figsize=DEFAULT_EVALUATION_FIGSIZE)
    elif axs.shape != (2, 2):
        error_msg = f"axs must have shape (2, 2), got {axs.shape}."
        raise ValueError(error_msg)
    else:
        figure = axs[0, 0].figure

    cohesive_energy_ax: Axes = axs[0, 0]
    force_ax: Axes = axs[0, 1]
    stress_ax: Axes = axs[1, 0]
    charge_ax: Axes = axs[1, 1]

    plot_reference_prediction_sequence_as_parity(
        cohesive_energy_ax,
        [entry.to_cohesive_energy_entry() for entry in entries],
        property_label=COHESIVE_ENERGY_LABEL,
        show_legend=False,
        show_rmse=show_rmse,
    )
    plot_reference_prediction_sequence_as_parity(
        force_ax,
        [
            entry.to_force_entry(mode=force_mode, adaptive_alpha=adaptive_alpha)
            for entry in entries
        ],
        property_label=force_label,
        show_legend=False,
        show_rmse=show_rmse,
    )
    legend_handles, legend_labels = cohesive_energy_ax.get_legend_handles_labels()
    force_ax.legend(legend_handles, legend_labels)
    plot_reference_prediction_sequence_as_parity(
        stress_ax,
        [
            entry.to_stress_entry(mode=stress_mode, adaptive_alpha=adaptive_alpha)
            for entry in entries
        ],
        property_label=stress_label,
        show_legend=False,
        show_rmse=show_rmse,
    )
    if _warn_if_reference_charges_missing(entries=entries):
        charge_ax.set_xlabel(f"True {CHARGE_LABEL}")
        charge_ax.set_ylabel(f"Pred {CHARGE_LABEL}")
        charge_ax.set_box_aspect(1)
    else:
        plot_reference_prediction_sequence_as_parity(
            charge_ax,
            [entry.to_charge_entry(adaptive_alpha=adaptive_alpha) for entry in entries],
            property_label=CHARGE_LABEL,
            show_legend=False,
            show_rmse=show_rmse,
        )
    figure.tight_layout()
    return figure, axs


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, strict=True))
def plot_model_comparison_residual_plots(
    entries: ModelEvaluationEntrySequence,
    *,
    force_mode: VectorMode = VectorMode.norm,
    stress_mode: VectorMode = VectorMode.norm,
    adaptive_alpha: bool = True,
    show_rmse: bool = True,
    axs: NDArray[np.object_] | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Create a 2x2 figure containing all model-comparison residual plots.

    The panels show cohesive energy per atom, forces, stresses, and charges in
    row-major order. Forces and stresses default to norms and can instead show
    components by setting their respective modes to `VectorMode.component`. Pass
    `axs` to plot on an existing 2x2 axes array. Each entry's label and scatter
    keyword arguments are applied consistently across all four panels. The charge
    panel is skipped if any model entry's reference structures lack charges. Set
    `adaptive_alpha=False` to preserve each entry's original opacity across panels
    and `show_rmse=False` to hide the lower-left RMSE summaries.
    """
    match force_mode:
        case VectorMode.norm:
            force_label = FORCE_NORM_LABEL
        case VectorMode.component:
            force_label = FORCE_COMPONENT_LABEL

    match stress_mode:
        case VectorMode.norm:
            stress_label = STRESS_NORM_LABEL
        case VectorMode.component:
            stress_label = STRESS_COMPONENT_LABEL

    if axs is None:
        figure, axs = plt.subplots(nrows=2, ncols=2, figsize=DEFAULT_EVALUATION_FIGSIZE)
    elif axs.shape != (2, 2):
        error_msg = f"axs must have shape (2, 2), got {axs.shape}."
        raise ValueError(error_msg)
    else:
        figure = axs[0, 0].figure

    cohesive_energy_ax: Axes = axs[0, 0]
    force_ax: Axes = axs[0, 1]
    stress_ax: Axes = axs[1, 0]
    charge_ax: Axes = axs[1, 1]

    plot_reference_prediction_sequence_as_residuals(
        cohesive_energy_ax,
        [entry.to_cohesive_energy_entry() for entry in entries],
        property_label=COHESIVE_ENERGY_LABEL,
        show_legend=False,
        show_rmse=show_rmse,
    )
    plot_reference_prediction_sequence_as_residuals(
        force_ax,
        [
            entry.to_force_entry(mode=force_mode, adaptive_alpha=adaptive_alpha)
            for entry in entries
        ],
        property_label=force_label,
        show_legend=False,
        show_rmse=show_rmse,
    )
    legend_handles, legend_labels = cohesive_energy_ax.get_legend_handles_labels()
    force_ax.legend(legend_handles, legend_labels)
    plot_reference_prediction_sequence_as_residuals(
        stress_ax,
        [
            entry.to_stress_entry(mode=stress_mode, adaptive_alpha=adaptive_alpha)
            for entry in entries
        ],
        property_label=stress_label,
        show_legend=False,
        show_rmse=show_rmse,
    )
    if _warn_if_reference_charges_missing(entries=entries):
        charge_ax.set_xlabel(f"True {CHARGE_LABEL}")
        charge_ax.set_ylabel(f"Pred-True {CHARGE_LABEL}")
        charge_ax.set_box_aspect(1)
    else:
        plot_reference_prediction_sequence_as_residuals(
            charge_ax,
            [entry.to_charge_entry(adaptive_alpha=adaptive_alpha) for entry in entries],
            property_label=CHARGE_LABEL,
            show_legend=False,
            show_rmse=show_rmse,
        )
    figure.tight_layout()
    return figure, axs


def _warn_if_reference_charges_missing(
    *, entries: Sequence[ModelEvaluationEntry]
) -> bool:
    missing_charge_labels = [
        entry.label
        for entry in entries
        if any(
            structure.charges is None
            for structure in entry.result.reference_structure_summaries
        )
    ]
    if not missing_charge_labels:
        return False

    warnings.warn(
        "Skipping charge plot because reference charges are missing for: "
        f"{', '.join(missing_charge_labels)}.",
        UserWarning,
        stacklevel=2,
    )
    return True


def _add_rmse_panel(*, ax: Axes, entries: Sequence[ReferencePredictionEntry]) -> None:
    rmse_summary = "\n".join(
        f"{entry.label} RMSE: {entry.values.rmse:.3g}" for entry in entries
    )
    ax.text(
        0.02,
        0.02,
        rmse_summary,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.9,
        },
    )


def _get_parity_limits(*, ax: Axes) -> tuple[float, float]:
    x_lower, x_upper = ax.get_xlim()
    y_lower, y_upper = ax.get_ylim()
    return min(x_lower, y_lower), max(x_upper, y_upper)
