from copy import deepcopy
from pathlib import Path

import ase
import pytest
from ase.io import read, write
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.helpers.md_run_helpers import load_atoms
from crystal.schemas.md.data import MdSeries
from crystal.schemas.md.utils import (
    compute_running_average_series,
    parse_default_md_log,
    parse_user_defined_md_log,
    read_last_trajectory_frame,
    restore_trajectory_site_properties,
    store_trajectory_site_properties,
)
from tests.helpers.constants.md import (
    MOCK_DEFAULT_LOG_CONTENT,
    MOCK_USER_DEFINED_LOG_CONTENT,
    TEST_TRAJECTORY_PATH,
)
from tests.helpers.constants.structure import (
    FLOATING_POINT_TOLERANCE,
    TEST_BCC_FE_STRUCTURE,
)


def test_read_last_trajectory_frame_raises_for_missing_trajectory(
    tmp_path: Path,
) -> None:
    missing_trajectory = tmp_path / "missing.traj"

    with pytest.raises(ValueError, match="Trajectory file does not exist"):
        read_last_trajectory_frame(missing_trajectory)


def test_read_last_trajectory_frame_returns_final_frame() -> None:
    final_frame = read_last_trajectory_frame(TEST_TRAJECTORY_PATH)
    expected_frame = read(TEST_TRAJECTORY_PATH, index=-1)

    assert isinstance(expected_frame, ase.Atoms)
    assert final_frame.positions == pytest.approx(
        expected_frame.positions, abs=FLOATING_POINT_TOLERANCE
    )
    assert final_frame.cell.array == pytest.approx(
        expected_frame.cell.array, abs=FLOATING_POINT_TOLERANCE
    )


def test_pymatgen_site_properties_survive_ase_trajectory_round_trip(
    tmp_path: Path,
) -> None:
    structure = deepcopy(TEST_BCC_FE_STRUCTURE).make_supercell((2, 2, 1))
    interface_labels = ["substrate", "film"] * (len(structure) // 2)
    structure.add_site_property("interface_label", interface_labels)

    atoms: ase.Atoms = AseAtomsAdaptor.get_atoms(structure)
    recreated_structure_from_atoms: PMG_Structure = AseAtomsAdaptor.get_structure(atoms)
    assert "interface_label" in recreated_structure_from_atoms.site_properties

    trajectory_path_without_site_properties = (
        tmp_path / "structure_without_site_properties.traj"
    )
    write(trajectory_path_without_site_properties, atoms)
    recreated_atoms_from_traj_file = read(
        trajectory_path_without_site_properties, index=-1
    )
    assert isinstance(recreated_atoms_from_traj_file, ase.Atoms)
    recreated_structure_from_traj_file: PMG_Structure = AseAtomsAdaptor.get_structure(
        recreated_atoms_from_traj_file
    )
    assert "interface_label" not in recreated_structure_from_traj_file.site_properties

    store_trajectory_site_properties(atoms=atoms)
    trajectory_path_with_site_properties = (
        tmp_path / "structure_with_site_properties.traj"
    )
    write(trajectory_path_with_site_properties, atoms)
    recreated_atoms_from_traj_file_with_site_properties = read(
        trajectory_path_with_site_properties, index=-1
    )
    assert isinstance(recreated_atoms_from_traj_file_with_site_properties, ase.Atoms)
    restore_trajectory_site_properties(
        recreated_atoms_from_traj_file_with_site_properties
    )
    recreated_structure_from_traj_file_with_site_properties: PMG_Structure = (
        AseAtomsAdaptor.get_structure(
            recreated_atoms_from_traj_file_with_site_properties
        )
    )
    assert (
        "interface_label"
        in recreated_structure_from_traj_file_with_site_properties.site_properties
    )
    assert (
        recreated_structure_from_traj_file_with_site_properties.site_properties[
            "interface_label"
        ]
        == structure.site_properties["interface_label"]
    )


def test_pymatgen_site_properties_in_ase_trajectory_for_md_runner_utils(
    tmp_path: Path,
) -> None:
    structure = deepcopy(TEST_BCC_FE_STRUCTURE).make_supercell((2, 2, 1))
    interface_labels = ["substrate", "film"] * (len(structure) // 2)
    structure.add_site_property("interface_label", interface_labels)
    trajectory_path = tmp_path / "structure.traj"

    atoms = load_atoms(structure)
    write(trajectory_path, atoms)
    trajectory_atoms = read_last_trajectory_frame(trajectory_path)
    round_tripped_structure: PMG_Structure = AseAtomsAdaptor.get_structure(
        trajectory_atoms
    )
    assert isinstance(round_tripped_structure, PMG_Structure)
    assert trajectory_atoms.arrays["interface_label"].tolist() == interface_labels
    assert (
        round_tripped_structure.site_properties["interface_label"] == interface_labels
    )


def test_errors_in_restoring_trajectory_site_properties() -> None:
    atoms = ase.Atoms("Fe")
    atoms_for_dict_error = deepcopy(atoms)
    atoms_for_dict_error.info["pymatgen_site_properties"] = ["interface_label"]

    with pytest.raises(TypeError, match="metadata to be a dictionary"):
        restore_trajectory_site_properties(atoms_for_dict_error)

    atoms_with_list_error = deepcopy(atoms)
    atoms_with_list_error.info["pymatgen_site_properties"] = {
        "interface_label": "substrate"
    }
    with pytest.raises(ValueError, match="lists with one value per atom"):
        restore_trajectory_site_properties(atoms_with_list_error)

    atoms_with_wrong_length_in_properties = deepcopy(atoms)
    atoms_with_wrong_length_in_properties.info["pymatgen_site_properties"] = {
        "interface_label": ["substrate", "film"]
    }
    with pytest.raises(ValueError, match="one value per atom"):
        restore_trajectory_site_properties(atoms_with_wrong_length_in_properties)


def test_parse_default_md_log_reads_energy_and_temperature_columns(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "nvt_default.log"
    log_path.write_text(MOCK_DEFAULT_LOG_CONTENT, encoding="utf-8")

    parsed_log = parse_default_md_log(log_path)

    assert parsed_log.time_ps == pytest.approx([0.0, 0.1, 0.2])
    assert parsed_log.total_energy_ev == pytest.approx([-10.0, -9.8, -9.7])
    assert parsed_log.temperature_k == pytest.approx([300.0, 305.0, 302.0])


def test_parse_user_defined_md_log_reads_density_and_force_columns(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "nvt_user_defined.log"
    log_path.write_text(MOCK_USER_DEFINED_LOG_CONTENT, encoding="utf-8")

    parsed_log = parse_user_defined_md_log(log_path)

    assert parsed_log.time_ps == pytest.approx([0.0, 0.1, 0.2])
    assert parsed_log.density == pytest.approx([2.5, 2.45, 2.47])
    assert parsed_log.max_force_component == pytest.approx([0.1, 0.09, 0.08])


def test_compute_running_average_series_returns_windowed_averages() -> None:
    averaged_series = compute_running_average_series(
        series=MdSeries(time_ps=[0.0, 0.1, 0.2, 0.3], values=[10.0, 11.0, 12.0, 13.0]),
        time_averaging_window=0.2,
    )

    assert averaged_series.time_ps == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert averaged_series.values == pytest.approx([10.0, 10.5, 11.0, 12.0])


def test_compute_running_average_series_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        _ = compute_running_average_series(
            series=MdSeries(time_ps=[0.0, 0.1], values=[10.0, 11.0]),
            time_averaging_window=0.0,
        )
