"""Utility helpers for MD schema data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ase
import numpy as np
from ase.io import read

from crystal.schemas.md.data import DefaultMdLogData, MdSeries, UserDefinedMdLogData

DEFAULT_USER_DEFINED_HEADERS: list[str] = [
    "time_ps",
    "density",
    "max_force_component_ev_per_angstrom",
]
DEFAULT_MD_LOG_MINIMUM_COLUMN_COUNT = 5
TRAJECTORY_SITE_PROPERTIES_INFO_KEY = "pymatgen_site_properties"
DEFAULT_KEYS_TO_STORE = {"interface_label"}


def store_trajectory_site_properties(
    atoms: ase.Atoms, keys: set[str] = DEFAULT_KEYS_TO_STORE
) -> None:
    """Store ASE trajectory-unsafe per-site arrays in metadata."""
    site_properties: dict[str, list[Any]] = {}
    for property_name, values in atoms.arrays.items():
        if property_name in keys:
            site_properties[property_name] = values.tolist()

    if site_properties:
        atoms.info[TRAJECTORY_SITE_PROPERTIES_INFO_KEY] = site_properties


def restore_trajectory_site_properties(atoms: ase.Atoms) -> None:  # noqa: D103
    site_properties = atoms.info.get(TRAJECTORY_SITE_PROPERTIES_INFO_KEY)
    if site_properties is None:
        return
    if not isinstance(site_properties, dict):
        error_msg = (
            "Expected trajectory site property metadata to be a dictionary keyed by "
            "property name."
        )
        raise TypeError(error_msg)

    for property_name, values in site_properties.items():
        if property_name in atoms.arrays:
            continue
        if not isinstance(values, list) or len(values) != len(atoms):
            error_msg = (
                "Expected trajectory site property values to be lists with one value "
                "per atom."
            )
            raise ValueError(error_msg)
        atoms.arrays[property_name] = np.asarray(values)


def _parse_float_tokens(tokens: list[str]) -> list[float] | None:
    try:
        return [float(token) for token in tokens]
    except ValueError:
        return None


def parse_default_md_log(log_path: Path) -> DefaultMdLogData:
    """Parse an ASE default MD log into typed time-series data."""
    time_ps: list[float] = []
    total_energy_ev: list[float] = []
    temperature_k: list[float] = []

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        tokens = stripped_line.split()
        if len(tokens) < DEFAULT_MD_LOG_MINIMUM_COLUMN_COUNT:
            continue
        float_tokens = _parse_float_tokens(tokens)
        if float_tokens is None:
            continue
        time_ps.append(float_tokens[0])
        total_energy_ev.append(float_tokens[1])
        temperature_k.append(float_tokens[4])

    if not time_ps:
        error_msg = f"No MD samples were parsed from default log: {log_path}"
        raise ValueError(error_msg)

    return DefaultMdLogData(
        time_ps=time_ps, total_energy_ev=total_energy_ev, temperature_k=temperature_k
    )


def parse_user_defined_md_log(log_path: Path) -> UserDefinedMdLogData:
    """Parse the custom CSV-style MD log into typed time-series data."""
    time_ps: list[float] = []
    density: list[float] = []
    max_force_component_ev_per_angstrom: list[float] = []

    for line_number, raw_line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines()
    ):
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        if line_number == 0 and stripped_line.startswith("time_ps"):
            continue
        values = [value.strip() for value in stripped_line.split(",")]
        if len(values) != len(DEFAULT_USER_DEFINED_HEADERS):
            error_msg = (
                "Expected user-defined MD log rows to contain "
                f"{len(DEFAULT_USER_DEFINED_HEADERS)} columns."
            )
            raise ValueError(error_msg)
        time_value, density_value, max_force_value = (float(value) for value in values)
        time_ps.append(time_value)
        density.append(density_value)
        max_force_component_ev_per_angstrom.append(max_force_value)

    if not time_ps:
        error_msg = f"No MD samples were parsed from user-defined log: {log_path}"
        raise ValueError(error_msg)

    return UserDefinedMdLogData(
        time_ps=time_ps,
        density=density,
        max_force_component=max_force_component_ev_per_angstrom,
    )


def compute_running_average_series(
    *, series: MdSeries, time_averaging_window: float
) -> MdSeries:
    """Compute a running-average series over the provided time window in ps."""
    if time_averaging_window <= 0.0:
        error_msg = "time_averaging_window must be greater than zero."
        raise ValueError(error_msg)

    averaged_values: list[float] = []
    for index, current_time in enumerate(series.time_ps):
        window_start_time = current_time - time_averaging_window
        window_values = [
            value
            for sample_time, value in zip(
                series.time_ps[: index + 1], series.values[: index + 1], strict=True
            )
            if sample_time >= window_start_time
        ]
        averaged_values.append(float(np.mean(window_values)))

    return MdSeries(time_ps=series.time_ps, values=averaged_values)


def read_last_trajectory_frame(trajectory_path: Path) -> ase.Atoms:  # noqa: D103
    if not trajectory_path.is_file():
        error_msg = f"Trajectory file does not exist: {trajectory_path}"
        raise ValueError(error_msg)
    last_frame = read(trajectory_path, index=-1)
    if not isinstance(last_frame, ase.Atoms):
        error_msg = f"Expected a single trajectory frame from {trajectory_path}."
        raise TypeError(error_msg)
    restore_trajectory_site_properties(atoms=last_frame)
    return last_frame
