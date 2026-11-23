"""GPAW relax runner using an external GPAW runtime environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import ase
from ase import Atoms
from ase.io import write
from pydantic import ConfigDict, Field, PositiveInt, ValidationError
from pymatgen.core.structure import Structure as PMG_Structure
from pymatgen.io.ase import AseAtomsAdaptor

from crystal.calculators.gpaw.external_runner import (
    prepare_external_gpaw_script,
    run_external_gpaw_script,
)
from crystal.calculators.gpaw.gpaw_runner import adapt_atoms_to_gpaw_bulk_or_slab
from crystal.constants.gpaw import (
    GPAW_EXTERNAL_RELAX_RUN_FAILED_MESSAGE,
    GPAW_EXTERNAL_RELAX_SCRIPT_FILENAME,
    GPAW_SCF_INPUT_STRUCTURE_FILENAME,
    GPAW_SCF_KWARGS_FILENAME,
    GPAW_SCF_OUTPUT_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
    GpawExternalRelaxScriptInputKey,
)
from crystal.constants.process import DEFAULT_NUMBER_OF_PROCESSES
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.helpers.scratch_execution import local_scratch_execution_directory
from crystal.schemas.gpaw import (
    GpawExternalInstallPathsConfig,
    GpawExternalInstallPathsConfigLoader,
    GpawExternalRelaxOutputPayload,
    GpawExternalRelaxStepOutputPayload,
    GpawRelaxSimulationSettings,
)
from crystal.schemas.properties import RelaxProperties, StepProperties
from crystal.schemas.scratch_execution import LocalScratchPolicy
from crystal.schemas.serialization import MSONableBaseModel

_GPAW_EXTERNAL_RELAX_SCRIPT_SOURCE_PATH = (
    Path(__file__).resolve().parent
    / "external_runtime"
    / GPAW_EXTERNAL_RELAX_SCRIPT_FILENAME
)
GPAW_RELAX_SCRATCH_RESULT_FILENAMES = (
    GPAW_EXTERNAL_RELAX_SCRIPT_FILENAME,
    GPAW_SCF_INPUT_STRUCTURE_FILENAME,
    GPAW_SCF_KWARGS_FILENAME,
    GPAW_SCF_OUTPUT_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
)


@dataclass(frozen=True)
class GpawExternalRelaxRunFiles:  # noqa: D101
    run_directory: Path

    @property
    def script_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_EXTERNAL_RELAX_SCRIPT_FILENAME

    @property
    def input_structure_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_SCF_INPUT_STRUCTURE_FILENAME

    @property
    def kwargs_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_SCF_KWARGS_FILENAME

    @property
    def output_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_SCF_OUTPUT_FILENAME


class GpawExternalRelaxRunner(MSONableBaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    structure: PMG_Structure
    simulation_settings: GpawRelaxSimulationSettings
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

    @property
    def _fixed_atom_indices(self) -> list[int]:
        constraints = self.simulation_settings.constraints
        if constraints is None:
            return []
        return constraints.build_fixed_atom_indices(atoms=self._ase_atoms)

    @property
    def _fix_cell(self) -> bool:
        constraints = self.simulation_settings.constraints
        if constraints is None:
            return True
        return constraints.fix_cell

    def _write_runtime_input_files(self, run_files: GpawExternalRelaxRunFiles) -> None:
        write(run_files.input_structure_path.as_posix(), self._ase_atoms)
        calculator_kwargs = (
            self.simulation_settings.gpaw_settings.build_gpaw_calculator_kwargs()
        )
        calculator_kwargs["txt"] = GPAW_TEXT_LOG_FILENAME
        script_input_payload = {
            GpawExternalRelaxScriptInputKey.calculator_kwargs: calculator_kwargs,
            GpawExternalRelaxScriptInputKey.optimizer_type: (
                self.simulation_settings.optimizer_type
            ),
            GpawExternalRelaxScriptInputKey.max_steps: self.simulation_settings.max_steps,
            GpawExternalRelaxScriptInputKey.force_threshold_in_ev_per_angstrom: (
                self.simulation_settings.force_threshold_in_ev_per_angstrom
            ),
            GpawExternalRelaxScriptInputKey.store_interval: (
                self.simulation_settings.store_interval
            ),
            GpawExternalRelaxScriptInputKey.fix_cell: self._fix_cell,
            GpawExternalRelaxScriptInputKey.fixed_atom_indices: self._fixed_atom_indices,
        }
        run_files.kwargs_path.write_text(
            json.dumps(script_input_payload), encoding=DEFAULT_TEXT_ENCODING
        )

    def _build_relax_step_from_output_data(
        self, relax_step_payload: GpawExternalRelaxStepOutputPayload
    ) -> StepProperties:
        atoms = Atoms(
            symbols=relax_step_payload.symbols,
            positions=relax_step_payload.positions,
            cell=relax_step_payload.cell,
            pbc=relax_step_payload.pbc,
        )
        structure: PMG_Structure = self._ase_atoms_adaptor.get_structure(atoms)
        return StepProperties(
            step=relax_step_payload.step,
            structure=structure,
            forces=[list(force_vector) for force_vector in relax_step_payload.forces],
            stress=list(relax_step_payload.stress),
            energy=relax_step_payload.energy,
            charges=None,
        )

    def _build_relax_properties_from_output_data(
        self, output_data: GpawExternalRelaxOutputPayload
    ) -> RelaxProperties:
        relax_steps = [
            self._build_relax_step_from_output_data(relax_step_payload)
            for relax_step_payload in output_data.relax_steps
        ]
        return RelaxProperties(relax_steps=relax_steps)

    def _process_relax_output(self, *, output_path: Path) -> RelaxProperties:
        output_data_object = json.loads(
            output_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
        )
        output_data = GpawExternalRelaxOutputPayload.model_validate(output_data_object)
        return self._build_relax_properties_from_output_data(output_data)

    def run_relax(self, run_directory: Path) -> RelaxProperties:  # noqa: D102
        execution_directory = run_directory.resolve()
        run_files = GpawExternalRelaxRunFiles(run_directory=execution_directory)
        prepare_external_gpaw_script(
            script_path=run_files.script_path,
            script_source_path=_GPAW_EXTERNAL_RELAX_SCRIPT_SOURCE_PATH,
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
                failure_message=GPAW_EXTERNAL_RELAX_RUN_FAILED_MESSAGE,
            )
            return self._process_relax_output(output_path=output_path)
        except (
            FileNotFoundError,
            PermissionError,
            json.JSONDecodeError,
            ValidationError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(GPAW_EXTERNAL_RELAX_RUN_FAILED_MESSAGE) from exc


def run_gpaw_relax(  # noqa: D103
    structure: PMG_Structure,
    simulation_settings: GpawRelaxSimulationSettings,
    run_directory: Path,
    number_of_processes: int = DEFAULT_NUMBER_OF_PROCESSES,
    local_scratch_policy: LocalScratchPolicy | None = None,
) -> RelaxProperties:
    external_runtime_config = (
        GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime()
    )
    runner = GpawExternalRelaxRunner(
        structure=structure,
        simulation_settings=simulation_settings,
        external_runtime_config=external_runtime_config,
        number_of_processes=number_of_processes,
    )
    policy = local_scratch_policy or LocalScratchPolicy()
    with local_scratch_execution_directory(
        policy=policy,
        destination_directory=run_directory,
        result_filenames=GPAW_RELAX_SCRATCH_RESULT_FILENAMES,
        text_result_filenames_replacing_execution_directory=(GPAW_TEXT_LOG_FILENAME,),
    ) as execution_directory:
        return runner.run_relax(run_directory=execution_directory)
