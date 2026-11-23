from pathlib import Path

import pytest
from jobflow.managers.local import run_locally
from pymatgen.core import Lattice, Structure  # type: ignore [attr-defined]

from crystal.helpers.temp import run_in_temp_dir
from crystal.schemas.filtering import (
    SinglePointStructureFilteringSettings,
    SinglePointStructureFilteringTaskDocument,
)
from crystal.schemas.isolated_atom_energies import IsolatedAtomEnergiesRef
from crystal.schemas.properties import (
    SinglePointProperties,
    SinglePointStructureFilteringProperties,
)
from crystal.structure.handlers.filter_maker import SinglePointStructureFilteringMaker
from tests.helpers.constants.structure import TEST_SI_STRUCTURE as MOCK_SI_STRUCTURE

FilteringTaskDocument = SinglePointStructureFilteringTaskDocument

ONE_ATOM_STRUCTURE_SPECIES = ["Li"]
TWO_ATOM_STRUCTURE_SPECIES = ["Li", "Li"]
MOCK_ISOLATED_SI_ATOM_ENERGY = -1.0
MOCK_SI_STRUCTURE_ENERGY = -6.0
MOCK_SI_STRUCTURE_FORCES = [[0.3, 0.4, 0.0], [0.0, 0.0, 0.2]]
MOCK_SI_STRUCTURE_STRESS = [1.0, 2.0, 2.0, 0.0, 0.0, 0.0]
PASSING_COHESIVE_SINGLE_POINT_ENERGY = -4.0
PASSING_FORCE_NORM_FORCES = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
PASSING_STRESS_NORM_STRESS = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FAILING_COHESIVE_SINGLE_POINT_ENERGY = 100.0
FAILING_FORCE_NORM_FORCES = [[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
FAILING_STRESS_NORM_STRESS = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def get_vector_norm(vector: list[float]) -> float:
    return float(sum(component**2 for component in vector) ** 0.5)


def get_structure(
    lattice_abc: tuple[float, float, float] = (5.0, 5.0, 5.0),
    species: list[str] = ONE_ATOM_STRUCTURE_SPECIES,
    coords: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),),
    *,
    coords_are_cartesian: bool = True,
) -> Structure:
    lattice: Lattice = Lattice.orthorhombic(*lattice_abc)
    return Structure(
        lattice=lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=coords_are_cartesian,
    )


def get_single_point_filtering_maker(
    isolated_atom_energies_ref: IsolatedAtomEnergiesRef,
    *,
    maximum_cohesive_energy_per_atom: float = -1.0,
    maximum_force_norm: float = 1.0,
    maximum_stress_norm: float = 10.0,
) -> SinglePointStructureFilteringMaker:
    return SinglePointStructureFilteringMaker(
        simulation_settings=SinglePointStructureFilteringSettings(
            isolated_atom_energies=isolated_atom_energies_ref,
            maximum_cohesive_energy_per_atom=maximum_cohesive_energy_per_atom,
            maximum_force_norm=maximum_force_norm,
            maximum_stress_norm=maximum_stress_norm,
        )
    )


def write_isolated_atom_energy_artifact(
    tmp_path: Path, *, si_energy: float = MOCK_ISOLATED_SI_ATOM_ENERGY
) -> IsolatedAtomEnergiesRef:
    artifact_dir = tmp_path / "isolated_atom_energies"
    artifact_dir.mkdir()
    isolated_atom_symbol = MOCK_SI_STRUCTURE.elements[0].symbol
    (artifact_dir / "isolated_atom_energies.extxyz").write_text(
        "\n".join(
            [
                "1",
                f'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 energy={si_energy} pbc="T T T"',
                f"{isolated_atom_symbol} 5 5 5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "metadata.yaml").write_text(
        "name: test\npublished_on: 2026-01-01\ndescription: test artifact\n",
        encoding="utf-8",
    )
    return IsolatedAtomEnergiesRef(path=artifact_dir)


def get_single_point_properties(
    *,
    structure: Structure = MOCK_SI_STRUCTURE,
    energy: float = MOCK_SI_STRUCTURE_ENERGY,
    forces: list[list[float]] = MOCK_SI_STRUCTURE_FORCES,
    stress: list[float] = MOCK_SI_STRUCTURE_STRESS,
) -> SinglePointProperties:
    return SinglePointProperties(
        step=0, structure=structure, energy=energy, forces=forces, stress=stress
    )


def run_single_point_filtering_job(
    maker: SinglePointStructureFilteringMaker,
    single_points: list[SinglePointProperties],
    tmp_path: Path,
) -> SinglePointStructureFilteringTaskDocument:
    job = maker.make(single_points=single_points)
    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)
    task_doc = run_job[job.uuid][1].output
    assert isinstance(task_doc, SinglePointStructureFilteringTaskDocument)
    return task_doc


def validate_filtering_task_document(
    task_doc: FilteringTaskDocument,
    expected_properties_type: type[SinglePointStructureFilteringProperties],
) -> SinglePointStructureFilteringProperties:
    assert isinstance(task_doc.properties, expected_properties_type)
    assert task_doc.errors == ""
    return task_doc.properties


def test_single_point_filtering_metrics_use_single_point_outputs_correctly(
    tmp_path: Path,
) -> None:
    maker = get_single_point_filtering_maker(
        isolated_atom_energies_ref=write_isolated_atom_energy_artifact(tmp_path)
    )
    single_point_properties = get_single_point_properties()

    total_reference_energy = (
        len(single_point_properties.structure) * MOCK_ISOLATED_SI_ATOM_ENERGY
    )
    expected_cohesive_energy_per_atom = (
        MOCK_SI_STRUCTURE_ENERGY - total_reference_energy
    ) / len(single_point_properties.structure)
    expected_max_force_norm = max(
        get_vector_norm(force) for force in MOCK_SI_STRUCTURE_FORCES
    )
    expected_stress_norm = get_vector_norm(MOCK_SI_STRUCTURE_STRESS)

    task_doc = run_single_point_filtering_job(
        maker=maker, single_points=[single_point_properties], tmp_path=tmp_path
    )
    properties = validate_filtering_task_document(
        task_doc=task_doc,
        expected_properties_type=SinglePointStructureFilteringProperties,
    )
    [metrics] = properties.all_structures_with_metrics

    assert metrics.structure == single_point_properties.structure
    assert metrics.cohesive_energy_per_atom == pytest.approx(
        expected_cohesive_energy_per_atom
    )
    assert metrics.maximum_force_norm == pytest.approx(expected_max_force_norm)
    assert metrics.stress_norm == pytest.approx(expected_stress_norm)


@pytest.mark.parametrize(
    ("single_point", "expected_mask"),
    [
        pytest.param(get_single_point_properties(), [True], id="passes"),
        pytest.param(
            get_single_point_properties(energy=PASSING_COHESIVE_SINGLE_POINT_ENERGY),
            [True],
            id="cohesive-energy-passes",
        ),
        pytest.param(
            get_single_point_properties(energy=FAILING_COHESIVE_SINGLE_POINT_ENERGY),
            [False],
            id="cohesive-energy-fails",
        ),
        pytest.param(
            get_single_point_properties(forces=PASSING_FORCE_NORM_FORCES),
            [True],
            id="force-passes",
        ),
        pytest.param(
            get_single_point_properties(forces=FAILING_FORCE_NORM_FORCES),
            [False],
            id="force-fails",
        ),
        pytest.param(
            get_single_point_properties(stress=PASSING_STRESS_NORM_STRESS),
            [True],
            id="stress-passes",
        ),
        pytest.param(
            get_single_point_properties(stress=FAILING_STRESS_NORM_STRESS),
            [False],
            id="stress-fails",
        ),
    ],
)
def test_single_point_filtering_mask_for_each_property(
    tmp_path: Path, single_point: SinglePointProperties, expected_mask: list[bool]
) -> None:
    maker = get_single_point_filtering_maker(
        write_isolated_atom_energy_artifact(
            tmp_path, si_energy=MOCK_ISOLATED_SI_ATOM_ENERGY
        )
    )

    task_doc = run_single_point_filtering_job(
        maker=maker, single_points=[single_point], tmp_path=tmp_path
    )
    properties = validate_filtering_task_document(
        task_doc=task_doc,
        expected_properties_type=SinglePointStructureFilteringProperties,
    )
    assert properties.filtering_mask == expected_mask


def test_single_point_filtering_returns_empty_results_for_empty_input(
    tmp_path: Path,
) -> None:
    maker = get_single_point_filtering_maker(
        write_isolated_atom_energy_artifact(tmp_path)
    )

    task_doc = run_single_point_filtering_job(
        maker=maker, single_points=[], tmp_path=tmp_path
    )
    properties = validate_filtering_task_document(
        task_doc=task_doc,
        expected_properties_type=SinglePointStructureFilteringProperties,
    )

    assert properties.all_structures_with_metrics == []
    assert properties.filtering_mask == []
    assert properties.structures == []


@run_in_temp_dir
def test_single_point_filtering_maker_local_job_success(tmp_path: Path) -> None:
    maker = get_single_point_filtering_maker(
        write_isolated_atom_energy_artifact(tmp_path)
    )
    job = maker.make(single_points=[get_single_point_properties()])

    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)
    task_doc = run_job[job.uuid][1].output

    properties = validate_filtering_task_document(
        task_doc=task_doc,
        expected_properties_type=SinglePointStructureFilteringProperties,
    )
    assert properties.filtering_mask == [True]


@run_in_temp_dir
def test_single_point_filtering_maker_fails_closed_for_missing_reference(
    tmp_path: Path,
) -> None:
    maker = get_single_point_filtering_maker(
        write_isolated_atom_energy_artifact(
            tmp_path, si_energy=MOCK_ISOLATED_SI_ATOM_ENERGY
        )
    )
    lithium_single_point = get_single_point_properties(
        structure=get_structure(species=["Li"]), forces=[MOCK_SI_STRUCTURE_FORCES[0]]
    )
    job = maker.make(single_points=[lithium_single_point])

    run_job = run_locally(job, root_dir=tmp_path, create_folders=True)
    task_doc = run_job[job.uuid][1].output

    assert task_doc.properties is None
    assert "Missing isolated atom energy references for elements: Li" in task_doc.errors
