"""Tests for QE -> pp.x -> Chargemol workflow payloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from crystal.schemas.chargemol import QePpToChargemolWorkflowInputs
from tests.helpers.test_chargemol_factories import (
    build_default_qe_pp_input_settings,
    build_default_qe_pp_plot_settings,
    build_test_pseudopotential_config,
)
from tests.helpers.test_paths import TEST_CUBES_DIR, TEST_O_PBE_PSEUDOPOTENTIAL_FILENAME


def test_qe_pp_to_chargemol_workflow_inputs_builds_from_required_fields() -> None:
    """Workflow inputs should serialize required QE pp.x handoff fields."""
    payload = QePpToChargemolWorkflowInputs(
        cube_output_directory=TEST_CUBES_DIR,
        input_settings=build_default_qe_pp_input_settings(
            outdir=TEST_CUBES_DIR.as_posix()
        ),
        plot_settings=build_default_qe_pp_plot_settings(),
        pseudopotential_config=build_test_pseudopotential_config(
            Path(__file__),
            pseudopotentials_file_map={"O": TEST_O_PBE_PSEUDOPOTENTIAL_FILENAME},
        ),
    )
    assert payload.cube_output_directory == TEST_CUBES_DIR


def test_qe_pp_to_chargemol_workflow_inputs_requires_cube_output_directory() -> None:
    """Workflow inputs should require the cube output directory."""
    with pytest.raises(ValueError, match="cube_output_directory"):
        QePpToChargemolWorkflowInputs.model_validate(
            {
                "input_settings": build_default_qe_pp_input_settings(
                    outdir=TEST_CUBES_DIR.as_posix()
                ),
                "plot_settings": build_default_qe_pp_plot_settings(),
                "pseudopotential_config": build_test_pseudopotential_config(
                    Path(__file__),
                    pseudopotentials_file_map={
                        "O": TEST_O_PBE_PSEUDOPOTENTIAL_FILENAME
                    },
                ),
            }
        )
