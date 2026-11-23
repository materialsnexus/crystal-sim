"""Integration tests for GPAW using the configured external runtime."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite, prod
from pathlib import Path

import pytest
from ase.io.cube import read_cube_data
from ase.units import Bohr
from jobflow.managers.local import run_locally

from crystal.calculators.gpaw.chargemol_maker import GpawChargemolMaker
from crystal.calculators.gpaw.gpaw_runner import run_gpaw_scf
from crystal.constants.chargemol import (
    CHARGEMOL_BOND_ORDERS_FILENAME,
    CHARGEMOL_JOB_CONTROL_FILENAME,
    CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME,
    CHARGEMOL_OVERLAP_POPULATIONS_FILENAME,
    CHARGEMOL_TOTAL_CUBE_ANALYSIS_OUTPUT_FILENAME,
)
from crystal.constants.gpaw import (
    GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES,
    GPAW_TOTAL_DENSITY_CUBE_FILENAME,
)
from crystal.helpers.chargemol.charge_parser import ChargemolNetAtomicChargesParser
from crystal.schemas.gpaw import (
    GpawBulkOrSlab,
    GpawChargemolSimulationSettings,
    GpawChargemolTaskDocument,
    GpawConvergenceSettings,
    GpawScfSimulationSettings,
    GpawSettings,
)
from crystal.schemas.properties import SinglePointProperties
from tests.helpers.constants.gpaw import (
    GPAW_BULK_OR_SLAB_SLAB_INTERFACE_VALUE,
    SLAB_INTERFACE_GPAW_DIPOLELAYER,
    SLAB_INTERFACE_GPAW_PBC,
    TEST_GPAW_EXTERNAL_SI_NET_CHARGE_ABS_TOLERANCE,
    TEST_GPAW_EXTERNAL_SI_SCF_ENERGY_ABS_TOLERANCE_EV,
    TEST_GPAW_EXTERNAL_SI_SCF_EXPECTED_ENERGY_EV,
)
from tests.helpers.constants.structure import TEST_001_FE_SLAB, TEST_SI_STRUCTURE
from tests.helpers.external_gpaw_runtime import (
    assert_external_gpaw_runtime_configuration_is_present,
)
from tests.helpers.gpaw_chargemol_integration import (
    assert_external_gpaw_chargemol_configuration_is_present,
)


@pytest.mark.needs_third_party_executables
def test_run_gpaw_scf_with_external_runtime(tmp_path: Path) -> None:
    assert_external_gpaw_runtime_configuration_is_present()
    scf_result = run_gpaw_scf(
        structure=deepcopy(TEST_SI_STRUCTURE),
        simulation_settings=GpawScfSimulationSettings(),
        run_directory=tmp_path,
    )
    properties = scf_result.properties
    assert isinstance(properties, SinglePointProperties)
    assert properties.energy == pytest.approx(
        TEST_GPAW_EXTERNAL_SI_SCF_EXPECTED_ENERGY_EV,
        abs=TEST_GPAW_EXTERNAL_SI_SCF_ENERGY_ABS_TOLERANCE_EV,
    )
    assert properties.charges is None
    assert len(properties.forces) == len(TEST_SI_STRUCTURE)
    assert len(properties.stress) == 6
    assert scf_result.runtime_provenance.gpaw_version
    assert scf_result.runtime_provenance.elements == frozenset({"Si"})
    assert (
        len(
            scf_result.runtime_provenance.atomic_data_by_element[
                "Si"
            ].paw_setup_sha256_digest
        )
        == 64
    )
    cube_path = tmp_path / GPAW_TOTAL_DENSITY_CUBE_FILENAME
    assert cube_path.is_file()
    density_values, cube_atoms = read_cube_data(cube_path.as_posix())  # type: ignore[no-untyped-call]
    voxel_volume_angstrom3 = cube_atoms.get_volume() / float(prod(density_values.shape))
    integrated_total_density = float(density_values.sum() * voxel_volume_angstrom3)
    integrated_total_electrons = integrated_total_density / (Bohr**3)
    assert integrated_total_electrons == pytest.approx(
        sum(site.specie.Z for site in TEST_SI_STRUCTURE), abs=1e-3
    )


@pytest.mark.needs_third_party_executables
def test_run_gpaw_scf_then_gpaw_chargemol_maker_with_external_runtime(
    tmp_path: Path,
) -> None:
    assert_external_gpaw_chargemol_configuration_is_present()
    scf_result = run_gpaw_scf(
        structure=deepcopy(TEST_SI_STRUCTURE),
        simulation_settings=GpawScfSimulationSettings(),
        run_directory=tmp_path,
    )
    properties = scf_result.properties
    run_directory = tmp_path
    assert (run_directory / GPAW_TOTAL_DENSITY_CUBE_FILENAME).is_file()
    assert (run_directory / CHARGEMOL_JOB_CONTROL_FILENAME).is_file()

    chargemol_maker = GpawChargemolMaker(
        simulation_settings=GpawChargemolSimulationSettings()
    )
    job = chargemol_maker.make(
        job_control_directory=run_directory,
        structure=properties.structure,
        properties=properties,
    )
    run_job = run_locally(job, root_dir=run_directory, create_folders=True)
    output_doc = run_job[job.uuid][1].output

    assert isinstance(output_doc, GpawChargemolTaskDocument)
    assert output_doc.converged
    assert output_doc.chargemol_run_info is not None
    assert output_doc.properties is not None
    assert output_doc.structure is not None
    expected_output_filenames = set(GPAW_CHARGEMOL_EXPECTED_OUTPUT_FILENAMES)
    generated_output_filenames = {
        path.name for path in run_directory.iterdir() if path.is_file()
    }
    assert expected_output_filenames <= generated_output_filenames
    assert (run_directory / CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME).is_file()
    assert (run_directory / CHARGEMOL_BOND_ORDERS_FILENAME).is_file()
    assert (run_directory / CHARGEMOL_OVERLAP_POPULATIONS_FILENAME).is_file()
    assert (run_directory / CHARGEMOL_TOTAL_CUBE_ANALYSIS_OUTPUT_FILENAME).is_file()
    net_atomic_charges = ChargemolNetAtomicChargesParser(
        chargemol_net_atomic_charges_path=(
            run_directory / CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME
        )
    ).get_chargemol_atomic_charges()
    assert net_atomic_charges.atom_count == len(TEST_SI_STRUCTURE)
    assert net_atomic_charges.total_charge == pytest.approx(
        0.0, abs=TEST_GPAW_EXTERNAL_SI_NET_CHARGE_ABS_TOLERANCE
    )


@pytest.mark.needs_third_party_executables
def test_run_gpaw_scf_with_external_runtime_for_slab_interface(tmp_path: Path) -> None:
    assert_external_gpaw_runtime_configuration_is_present()
    slab_simulation_settings = GpawScfSimulationSettings(
        gpaw_settings=GpawSettings(
            bulk_or_slab=GpawBulkOrSlab(GPAW_BULK_OR_SLAB_SLAB_INTERFACE_VALUE),
            pbc=SLAB_INTERFACE_GPAW_PBC,
            poissonsolver={"dipolelayer": SLAB_INTERFACE_GPAW_DIPOLELAYER},
            cutoff_ev=250.0,
            kpoint_density=2.0,
            maxiter=80,
            convergence=GpawConvergenceSettings(
                density=1e-2, energy=5e-2, forces=0.2, eigenstates=1e-3
            ),
        )
    )
    scf_result = run_gpaw_scf(
        structure=deepcopy(TEST_001_FE_SLAB),
        simulation_settings=slab_simulation_settings,
        run_directory=tmp_path,
    )
    properties = scf_result.properties
    assert isinstance(properties, SinglePointProperties)
    assert isfinite(properties.energy)
    assert properties.charges is None
    assert len(properties.forces) == len(TEST_001_FE_SLAB)
    assert len(properties.stress) == 6
