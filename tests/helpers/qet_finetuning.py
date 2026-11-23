"""Shared factories for QET fine-tuning tests."""

from __future__ import annotations

from collections.abc import Sequence

from jobflow.core.job import job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.schemas.atomic_data import GpawAtomicDataInfo
from crystal.schemas.gpaw import (
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
    GpawScfChargemolTaskDocument,
    GpawScfSimulationSettings,
)
from crystal.schemas.gpaw_isolated_atom_reference import (
    IsolatedAtomEnergy,
    IsolatedAtomEnergyReference,
)
from crystal.schemas.properties import SinglePointProperties
from crystal.schemas.qet_finetuning import QetFineTuningRecord
from tests.helpers.builders.process_run_info import (
    build_successful_process_run_info_for_test,
)
from tests.helpers.constants.qet_finetuning import TEST_QET_FINE_TUNING_CHARGES
from tests.helpers.constants.structure import (
    TEST_RELAX_ENERGY_PLACEHOLDER,
    TEST_RELAX_STEP_INDEX,
    TEST_RELAX_STRESS_VECTOR_ZERO,
    TEST_RELAX_ZERO_FORCE_VECTOR,
    TEST_SI_STRUCTURE,
)

TEST_GPAW_VERSION = "25.7.0"
TEST_SILICON_SETUP_DIGEST = "a" * 64
TEST_SILICON_ISOLATED_ATOM_ENERGY = -5.0


def create_qet_fine_tuning_record(
    energy: float,
    *,
    structure: PMG_Structure = TEST_SI_STRUCTURE,
    stress: Sequence[float] = TEST_RELAX_STRESS_VECTOR_ZERO,
) -> QetFineTuningRecord:
    """Create a charge-complete QET fine-tuning record for tests."""
    return QetFineTuningRecord(
        structure=structure,
        energy=energy,
        forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in structure],
        stress=stress,
        charges=TEST_QET_FINE_TUNING_CHARGES,
    )


def create_isolated_atom_reference_for_test() -> IsolatedAtomEnergyReference:
    """Create the silicon isolated-atom reference used by QET maker tests."""
    return IsolatedAtomEnergyReference(
        simulation_settings=GpawIsolatedAtomSimulationSettings(),
        isolated_atom_energies=(
            IsolatedAtomEnergy(
                element="Si",
                energy=TEST_SILICON_ISOLATED_ATOM_ENERGY,
                runtime_provenance=GpawRuntimeProvenance(
                    gpaw_version=TEST_GPAW_VERSION,
                    atomic_data=(
                        GpawAtomicDataInfo(
                            element_symbol="Si",
                            paw_setup_sha256_digest=TEST_SILICON_SETUP_DIGEST,
                        ),
                    ),
                ),
            ),
        ),
    )


def create_gpaw_label_task_document(
    *,
    index: int = 0,
    energy: float | None = None,
    errors: str = "",
    charges: Sequence[float] | None = TEST_QET_FINE_TUNING_CHARGES,
    structure: PMG_Structure = TEST_SI_STRUCTURE,
) -> GpawScfChargemolTaskDocument:
    """Create a GPAW SCF plus Chargemol label document for tests."""
    if errors:
        return GpawScfChargemolTaskDocument(
            dir_name=f"label-{index}",
            errors=errors,
            properties=None,
            structure=None,
            chargemol_run_info=None,
        )

    return GpawScfChargemolTaskDocument(
        dir_name=f"label-{index}",
        errors="",
        properties=SinglePointProperties(
            step=TEST_RELAX_STEP_INDEX,
            structure=structure,
            energy=(
                TEST_RELAX_ENERGY_PLACEHOLDER + index if energy is None else energy
            ),
            forces=[TEST_RELAX_ZERO_FORCE_VECTOR for _ in structure],
            stress=TEST_RELAX_STRESS_VECTOR_ZERO,
            charges=charges,
        ),
        structure=structure,
        chargemol_run_info=build_successful_process_run_info_for_test(),
        gpaw_scf_simulation_settings=GpawScfSimulationSettings(),
        runtime_provenance=GpawRuntimeProvenance(
            gpaw_version=TEST_GPAW_VERSION,
            atomic_data=(
                GpawAtomicDataInfo(
                    element_symbol="Si",
                    paw_setup_sha256_digest=TEST_SILICON_SETUP_DIGEST,
                ),
            ),
        ),
    )


@job(output_schema=GpawScfChargemolTaskDocument)
def store_gpaw_label_task_document_for_test(
    task_document: GpawScfChargemolTaskDocument,
) -> GpawScfChargemolTaskDocument:
    """Return one label through Jobflow so tests can persist its output."""
    return task_document
