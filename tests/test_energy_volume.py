"""Tests for energy-volume flow construction and processing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from jobflow.managers.local import run_locally

from crystal.calculators.relax_maker_factory import build_relax_maker
from crystal.calculators.single_point_maker_factory import build_single_point_maker
from crystal.flows.energy_volume import (
    _process_single_points_for_energy_volume,
    compute_energy_volume_curves,
)
from crystal.helpers.structure import get_strained_structures
from crystal.schemas.energy_volume import EnergyVolumeOutput
from crystal.schemas.mlip import MlipSinglePointTaskDocument
from tests.helpers.builders.properties import build_single_point_properties_for_test
from tests.helpers.builders.relax_maker_settings import (
    build_mlip_relax_maker_settings_for_test,
)
from tests.helpers.builders.single_point_maker_settings import (
    build_mlip_single_point_maker_settings_for_test,
)
from tests.helpers.constants.structure import (
    TEST_RELAX_ENERGY_HIGH,
    TEST_RELAX_ENERGY_LOW,
    TEST_SI_STRUCTURE,
)

TEST_LINEAR_STRAIN = (-0.01, 0.01)
TEST_STRAIN_STEPS = 3

TEST_MLIP_SINGLE_POINT_MAKER = build_single_point_maker(
    single_point_maker_settings=build_mlip_single_point_maker_settings_for_test()
)
TEST_MLIP_RELAX_MAKER = build_relax_maker(
    relax_maker_settings=build_mlip_relax_maker_settings_for_test()
)
TEST_STRAINS = np.linspace(
    TEST_LINEAR_STRAIN[0], TEST_LINEAR_STRAIN[1], TEST_STRAIN_STEPS
)


def test_energy_volume_output_from_single_point_task_doc() -> None:
    single_point_properties = build_single_point_properties_for_test(
        TEST_SI_STRUCTURE, energy=TEST_RELAX_ENERGY_LOW
    )
    task_document = MlipSinglePointTaskDocument(
        properties=single_point_properties, errors=""
    )

    energy_volume_output = EnergyVolumeOutput.from_single_point_task_doc(task_document)

    assert energy_volume_output.strained_structure == single_point_properties.structure
    assert energy_volume_output.volume == single_point_properties.structure.volume
    assert energy_volume_output.energy == TEST_RELAX_ENERGY_LOW
    assert energy_volume_output.forces == single_point_properties.forces
    assert energy_volume_output.stresses == single_point_properties.stress
    assert energy_volume_output.charges == single_point_properties.charges


def test_process_single_points_for_energy_volume_skips_failed_task_documents(
    tmp_path: Path,
) -> None:
    strained_structures = get_strained_structures(
        structure=TEST_SI_STRUCTURE, strains=TEST_STRAINS
    )
    task_documents = [
        MlipSinglePointTaskDocument(
            properties=build_single_point_properties_for_test(
                strained_structures[0], energy=TEST_RELAX_ENERGY_LOW
            ),
            errors="",
        ),
        MlipSinglePointTaskDocument(
            properties=None, errors="mock single point failure"
        ),
        MlipSinglePointTaskDocument(
            properties=build_single_point_properties_for_test(
                strained_structures[2], energy=TEST_RELAX_ENERGY_HIGH
            ),
            errors="",
        ),
    ]

    processing_job = _process_single_points_for_energy_volume(
        single_points=task_documents
    )
    run_output = run_locally(processing_job, root_dir=tmp_path, create_folders=True)

    energy_volume_outputs = run_output[processing_job.uuid][1].output
    assert len(energy_volume_outputs) == TEST_STRAIN_STEPS - 1
    assert [output.energy for output in energy_volume_outputs] == [
        TEST_RELAX_ENERGY_LOW,
        TEST_RELAX_ENERGY_HIGH,
    ]
    assert [output.volume for output in energy_volume_outputs] == sorted(
        output.volume for output in energy_volume_outputs
    )


def test_energy_volume_flow_with_mlip_relax_and_mlip_single_point(
    tmp_path: Path,
) -> None:
    energy_volume_flow = compute_energy_volume_curves(
        structure=deepcopy(TEST_SI_STRUCTURE),
        linear_strain=TEST_LINEAR_STRAIN,
        strain_steps=TEST_STRAIN_STEPS,
        single_point_maker=deepcopy(TEST_MLIP_SINGLE_POINT_MAKER),
        relax_maker=deepcopy(TEST_MLIP_RELAX_MAKER),
    )

    assert len(energy_volume_flow.jobs) == TEST_STRAIN_STEPS + 3
    [relax_job, strain_job, *single_point_jobs, processing_job] = (
        energy_volume_flow.jobs
    )
    assert relax_job.name == "Relax Structure"
    assert strain_job.name == "Strain Structure"
    assert all(
        "Single point for strain" in single_point_job.name
        for single_point_job in single_point_jobs
    )
    assert processing_job.name == "Extract Energy Volume Output"

    energy_volume_flow_response = run_locally(
        energy_volume_flow, root_dir=tmp_path, create_folders=True
    )

    energy_volume_outputs = energy_volume_flow_response[
        energy_volume_flow.jobs[-1].uuid
    ][1].output
    assert len(energy_volume_outputs) == TEST_STRAIN_STEPS
    assert all(
        isinstance(output, EnergyVolumeOutput) for output in energy_volume_outputs
    )
    assert [output.volume for output in energy_volume_outputs] == sorted(
        output.volume for output in energy_volume_outputs
    )
    assert energy_volume_outputs[0].volume < TEST_SI_STRUCTURE.volume
    assert energy_volume_outputs[-1].volume > TEST_SI_STRUCTURE.volume

    for energy_volume_output in energy_volume_outputs:
        assert isinstance(energy_volume_output.energy, float)
        assert len(energy_volume_output.forces) == len(TEST_SI_STRUCTURE)
        assert len(energy_volume_output.stresses) == 6
        assert energy_volume_output.charges is not None
