from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from jobflow.managers.local import run_locally

from crystal.calculators.gpaw.scf_chargemol_maker import GpawScfChargemolMaker
from crystal.schemas.gpaw import (
    GpawChargemolSimulationSettings,
    GpawChargemolTaskDocument,
    GpawScfChargemolSimulationSettings,
    GpawScfChargemolTaskDocument,
    GpawScfSimulationSettings,
)
from tests.helpers.constants.structure import (
    TEST_SI_STRUCTURE,
    TEST_SINGLE_RESULT_COUNT,
)
from tests.helpers.gpaw_chargemol_integration import (
    assert_external_gpaw_chargemol_configuration_is_present,
    assert_successful_silicon_gpaw_chargemol_labelling,
)
from tests.helpers.jobflow import extract_outputs_of_type_from_run


@pytest.mark.needs_third_party_executables
def test_run_gpaw_scf_chargemol_maker_with_external_runtime(tmp_path: Path) -> None:
    assert_external_gpaw_chargemol_configuration_is_present()
    simulation_settings = GpawScfChargemolSimulationSettings(
        gpaw_scf_simulation_settings=GpawScfSimulationSettings(),
        gpaw_chargemol_simulation_settings=GpawChargemolSimulationSettings(),
    )
    maker = GpawScfChargemolMaker(simulation_settings=simulation_settings)
    flow = maker.make(structure=deepcopy(TEST_SI_STRUCTURE))
    run_flow = run_locally(flow, root_dir=tmp_path, create_folders=True)
    output_docs = extract_outputs_of_type_from_run(run_flow, GpawChargemolTaskDocument)
    assert len(output_docs) == TEST_SINGLE_RESULT_COUNT
    output_doc = output_docs[0]
    assert isinstance(output_doc, GpawScfChargemolTaskDocument)
    assert output_doc.gpaw_scf_simulation_settings == (
        simulation_settings.gpaw_scf_simulation_settings
    )
    assert_successful_silicon_gpaw_chargemol_labelling(
        task_document=output_doc, source_structure=TEST_SI_STRUCTURE
    )
