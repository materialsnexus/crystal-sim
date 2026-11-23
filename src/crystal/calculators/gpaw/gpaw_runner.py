"""GPAW SCF runner using an external GPAW runtime environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import ase
from ase.io import write
from pydantic import ConfigDict, Field, PositiveInt, ValidationError
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.calculators.gpaw.external_runner import (
    prepare_external_gpaw_script,
    run_external_gpaw_script,
)
from crystal.constants.chargemol import (
    CHARGEMOL_ATOMIC_DENSITIES_DIRECTORY_PLACEHOLDER,
    CHARGEMOL_JOB_CONTROL_FILENAME,
)
from crystal.constants.gpaw import (
    DEFAULT_GPAW_CHARGEMOL_DENSITY_GRIDREFINEMENT,
    DEFAULT_GPAW_SLAB_INTERFACE_PBC,
    GPAW_EXTERNAL_SCF_RUN_FAILED_MESSAGE,
    GPAW_EXTERNAL_SCF_SCRIPT_FILENAME,
    GPAW_RUNTIME_PROVENANCE_FILENAME,
    GPAW_SCF_INPUT_STRUCTURE_FILENAME,
    GPAW_SCF_KWARGS_FILENAME,
    GPAW_SCF_OUTPUT_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
    GPAW_TOTAL_DENSITY_CUBE_FILENAME,
    GpawExternalScfScriptInputKey,
)
from crystal.constants.process import DEFAULT_NUMBER_OF_PROCESSES
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.helpers.scratch_execution import local_scratch_execution_directory
from crystal.helpers.structure import get_centered_ase_atoms_along_c_axis
from crystal.schemas.chargemol import (
    ChargemolJobControlInputBuilder,
    ChargemolJobControlSettings,
)
from crystal.schemas.gpaw import (
    GpawBulkOrSlab,
    GpawExternalInstallPathsConfig,
    GpawExternalInstallPathsConfigLoader,
    GpawExternalScfOutputPayload,
    GpawRuntimeProvenance,
    GpawScfSimulationSettings,
)
from crystal.schemas.properties import SinglePointProperties
from crystal.schemas.scratch_execution import LocalScratchPolicy
from crystal.schemas.serialization import MSONableBaseModel

_GPAW_EXTERNAL_SCF_SCRIPT_SOURCE_PATH = (
    Path(__file__).resolve().parent
    / "external_runtime"
    / GPAW_EXTERNAL_SCF_SCRIPT_FILENAME
)
GPAW_SCF_SCRATCH_RESULT_FILENAMES = (
    GPAW_EXTERNAL_SCF_SCRIPT_FILENAME,
    GPAW_RUNTIME_PROVENANCE_FILENAME,
    GPAW_SCF_INPUT_STRUCTURE_FILENAME,
    GPAW_SCF_KWARGS_FILENAME,
    GPAW_SCF_OUTPUT_FILENAME,
    GPAW_TOTAL_DENSITY_CUBE_FILENAME,
    CHARGEMOL_JOB_CONTROL_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
)


@dataclass(frozen=True)
class GpawScfRunResult:
    """GPAW SCF output passed directly to the following Chargemol step."""

    properties: SinglePointProperties
    runtime_provenance: GpawRuntimeProvenance


@dataclass(frozen=True)
class GpawExternalScfRunFiles:  # noqa: D101
    run_directory: Path

    @property
    def script_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_EXTERNAL_SCF_SCRIPT_FILENAME

    @property
    def input_structure_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_SCF_INPUT_STRUCTURE_FILENAME

    @property
    def kwargs_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_SCF_KWARGS_FILENAME

    @property
    def output_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_SCF_OUTPUT_FILENAME

    @property
    def runtime_provenance_module_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_RUNTIME_PROVENANCE_FILENAME


def adapt_atoms_to_gpaw_bulk_or_slab(  # noqa: D103
    atoms: ase.Atoms, bulk_or_slab: GpawBulkOrSlab
) -> ase.Atoms:
    adapted_atoms: ase.Atoms = atoms.copy()  # type: ignore[no-untyped-call]
    if bulk_or_slab == GpawBulkOrSlab.slab_interface:
        adapted_atoms.set_pbc(DEFAULT_GPAW_SLAB_INTERFACE_PBC)  # type: ignore[no-untyped-call]
    return get_centered_ase_atoms_along_c_axis(atoms=adapted_atoms)


class GpawExternalScfRunner(MSONableBaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    structure: PMG_Structure
    simulation_settings: GpawScfSimulationSettings
    external_runtime_config: GpawExternalInstallPathsConfig
    number_of_processes: PositiveInt = Field(default=DEFAULT_NUMBER_OF_PROCESSES)

    @cached_property
    def _ase_atoms_adaptor(self) -> AseAtomsAdaptor:
        return AseAtomsAdaptor()

    @cached_property
    def _ase_atoms(self) -> ase.Atoms:
        ase_atoms = self._ase_atoms_adaptor.get_atoms(self.structure)
        return adapt_atoms_to_gpaw_bulk_or_slab(
            atoms=ase_atoms,
            bulk_or_slab=self.simulation_settings.gpaw_settings.bulk_or_slab,
        )

    def _write_runtime_input_files(self, run_files: GpawExternalScfRunFiles) -> None:
        write(run_files.input_structure_path.as_posix(), self._ase_atoms)
        calculator_kwargs = (
            self.simulation_settings.gpaw_settings.build_gpaw_calculator_kwargs()
        )
        calculator_kwargs["txt"] = GPAW_TEXT_LOG_FILENAME
        script_input_payload = {
            GpawExternalScfScriptInputKey.calculator_kwargs: calculator_kwargs,
            GpawExternalScfScriptInputKey.density_gridrefinement: (
                DEFAULT_GPAW_CHARGEMOL_DENSITY_GRIDREFINEMENT
            ),
            GpawExternalScfScriptInputKey.total_density_cube_filename: (
                GPAW_TOTAL_DENSITY_CUBE_FILENAME
            ),
        }
        run_files.kwargs_path.write_text(
            json.dumps(script_input_payload), encoding=DEFAULT_TEXT_ENCODING
        )

    def run_single_point(self, run_directory: Path) -> GpawScfRunResult:
        """Run a single-point GPAW calculation in the external runtime."""
        execution_directory = run_directory.resolve()
        run_files = GpawExternalScfRunFiles(run_directory=execution_directory)
        prepare_external_gpaw_script(
            script_path=run_files.script_path,
            script_source_path=_GPAW_EXTERNAL_SCF_SCRIPT_SOURCE_PATH,
            runtime_provenance_module_path=run_files.runtime_provenance_module_path,
        )
        self._write_runtime_input_files(run_files=run_files)
        try:
            output_path = run_external_gpaw_script(
                external_runtime_config=self.external_runtime_config,
                number_of_processes=self.number_of_processes,
                script_path=run_files.script_path,
                structure_path=run_files.input_structure_path,
                kwargs_path=run_files.kwargs_path,
                output_path=run_files.output_path,
                run_directory=execution_directory,
                failure_message=GPAW_EXTERNAL_SCF_RUN_FAILED_MESSAGE,
            )
            return self._process_scf_output(
                output_path=output_path, run_directory=execution_directory
            )
        except (
            FileNotFoundError,
            PermissionError,
            json.JSONDecodeError,
            ValidationError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(GPAW_EXTERNAL_SCF_RUN_FAILED_MESSAGE) from exc

    def _process_scf_output(
        self, *, output_path: Path, run_directory: Path
    ) -> GpawScfRunResult:
        output_data_object = json.loads(
            output_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
        )
        output_data = GpawExternalScfOutputPayload.model_validate(output_data_object)
        self._write_chargemol_job_control_file(
            run_directory=run_directory, output_data=output_data
        )
        return GpawScfRunResult(
            properties=SinglePointProperties(
                step=0,
                structure=self._ase_atoms_adaptor.get_structure(self._ase_atoms),
                forces=[list(force_vector) for force_vector in output_data.forces],
                stress=list(output_data.stress),
                energy=output_data.energy,
                charges=None,
            ),
            runtime_provenance=output_data.runtime_provenance,
        )

    @staticmethod
    def _write_chargemol_job_control_file(
        *, run_directory: Path, output_data: GpawExternalScfOutputPayload
    ) -> Path:
        input_filename = run_directory / GPAW_TOTAL_DENSITY_CUBE_FILENAME
        if not input_filename.is_file():
            raise FileNotFoundError(input_filename.as_posix())
        pbc = output_data.pbc
        job_control_settings = ChargemolJobControlSettings(
            input_filename=input_filename,
            atomic_densities_directory=Path(
                CHARGEMOL_ATOMIC_DENSITIES_DIRECTORY_PLACEHOLDER
            ),
            periodic_a=pbc[0],
            periodic_b=pbc[1],
            periodic_c=pbc[2],
            core_electrons_by_atomic_number={},
        )
        job_control_path = run_directory / CHARGEMOL_JOB_CONTROL_FILENAME
        job_control_path.write_text(
            ChargemolJobControlInputBuilder(
                settings=job_control_settings
            ).job_control_text,
            encoding=DEFAULT_TEXT_ENCODING,
        )
        return job_control_path


def run_gpaw_scf(  # noqa: D103
    structure: PMG_Structure,
    simulation_settings: GpawScfSimulationSettings,
    run_directory: Path,
    number_of_processes: int = DEFAULT_NUMBER_OF_PROCESSES,
    local_scratch_policy: LocalScratchPolicy | None = None,
) -> GpawScfRunResult:
    external_runtime_config = (
        GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime()
    )
    runner = GpawExternalScfRunner(
        structure=structure,
        simulation_settings=simulation_settings,
        external_runtime_config=external_runtime_config,
        number_of_processes=number_of_processes,
    )
    policy = local_scratch_policy or LocalScratchPolicy()
    with local_scratch_execution_directory(
        policy=policy,
        destination_directory=run_directory,
        result_filenames=GPAW_SCF_SCRATCH_RESULT_FILENAMES,
        text_result_filenames_replacing_execution_directory=(GPAW_TEXT_LOG_FILENAME,),
    ) as execution_directory:
        return runner.run_single_point(run_directory=execution_directory)
