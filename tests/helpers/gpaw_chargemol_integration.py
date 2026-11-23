"""Shared assertions for real GPAW-to-Chargemol integration coverage."""

from __future__ import annotations

from pathlib import Path

from pymatgen.core.structure import Structure as PMG_Structure

from crystal import CRYSTAL_CONFIG
from crystal.constants.chargemol import CHARGEMOL_JOB_CONTROL_FILENAME
from crystal.constants.gpaw import (
    GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES,
    GPAW_TEXT_LOG_FILENAME,
    GPAW_TOTAL_DENSITY_CUBE_FILENAME,
)
from crystal.helpers.chargemol.charge_parser import (
    CHARGEMOL_NET_ATOMIC_CHARGES_PROPERTY,
)
from crystal.schemas.gpaw import GpawScfChargemolTaskDocument
from tests.helpers.external_gpaw_runtime import (
    assert_external_gpaw_runtime_configuration_is_present,
)

GPAW_PAW_SETUP_SHA256_DIGEST_LENGTH = 64


def assert_external_gpaw_chargemol_configuration_is_present() -> None:
    """Assert that the user-local external program paths are configured."""
    assert_external_gpaw_runtime_configuration_is_present()
    assert CRYSTAL_CONFIG.programs.chargemol_dir is not None


def assert_successful_silicon_gpaw_chargemol_labelling(
    task_document: GpawScfChargemolTaskDocument, source_structure: PMG_Structure
) -> None:
    """Assert successful scientific outputs and materialized runner artifacts."""
    assert task_document.converged
    assert task_document.properties is not None
    assert task_document.structure is not None
    assert task_document.chargemol_run_info is not None
    assert task_document.chargemol_run_info.exit_code_ok
    assert task_document.runtime_provenance is not None
    assert task_document.runtime_provenance.gpaw_version
    assert task_document.runtime_provenance.elements == frozenset({"Si"})
    assert (
        len(
            task_document.runtime_provenance.atomic_data_by_element[
                "Si"
            ].paw_setup_sha256_digest
        )
        == GPAW_PAW_SETUP_SHA256_DIGEST_LENGTH
    )
    assert task_document.properties.charges is not None
    assert len(task_document.properties.charges) == len(source_structure)
    assert (
        task_document.structure.site_properties[CHARGEMOL_NET_ATOMIC_CHARGES_PROPERTY]
        == task_document.properties.charges
    )
    assert task_document.dir_name is not None
    run_directory = Path(task_document.dir_name)
    assert (run_directory / CHARGEMOL_JOB_CONTROL_FILENAME).is_file()
    assert not (run_directory / GPAW_TOTAL_DENSITY_CUBE_FILENAME).exists()
    assert (run_directory / GPAW_TEXT_LOG_FILENAME).is_file()
    assert all(
        (run_directory / output_filename).is_file()
        for output_filename in GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES
    )
