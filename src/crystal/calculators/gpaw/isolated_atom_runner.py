"""Energy-only isolated-atom runner using the external GPAW runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from pathlib import Path

from ase import Atoms
from ase.io import write
from pydantic import ConfigDict, Field, PositiveInt, ValidationError

from crystal.calculators.gpaw.external_runner import (
    prepare_external_gpaw_script,
    run_external_gpaw_script,
)
from crystal.constants.gpaw import (
    GPAW_EXTERNAL_ISOLATED_ATOM_RUN_FAILED_MESSAGE,
    GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_FILENAME,
    GPAW_ISOLATED_ATOM_INPUT_STRUCTURE_FILENAME,
    GPAW_ISOLATED_ATOM_KWARGS_FILENAME,
    GPAW_ISOLATED_ATOM_OUTPUT_FILENAME,
    GPAW_RUNTIME_PROVENANCE_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
)
from crystal.constants.process import DEFAULT_NUMBER_OF_PROCESSES
from crystal.constants.text import DEFAULT_TEXT_ENCODING
from crystal.helpers.scratch_execution import local_scratch_execution_directory
from crystal.schemas.elements import ElementStr
from crystal.schemas.gpaw import (
    GpawEigensolverName,
    GpawExternalInstallPathsConfig,
    GpawExternalInstallPathsConfigLoader,
    GpawExternalIsolatedAtomOutputPayload,
    GpawIsolatedAtomSimulationSettings,
    GpawRuntimeProvenance,
)
from crystal.schemas.scratch_execution import LocalScratchPolicy
from crystal.schemas.serialization import MSONableBaseModel

_GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_SOURCE_PATH = (
    Path(__file__).resolve().parent
    / "external_runtime"
    / GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_FILENAME
)
GPAW_ISOLATED_ATOM_SCRATCH_RESULT_FILENAMES = (
    GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_FILENAME,
    GPAW_RUNTIME_PROVENANCE_FILENAME,
    GPAW_ISOLATED_ATOM_INPUT_STRUCTURE_FILENAME,
    GPAW_ISOLATED_ATOM_KWARGS_FILENAME,
    GPAW_ISOLATED_ATOM_OUTPUT_FILENAME,
    GPAW_TEXT_LOG_FILENAME,
)
GPAW_ISOLATED_ATOM_RESULT_NONFINITE_ENERGY_MESSAGE = (
    "Isolated-atom energy must be finite."
)
GPAW_ISOLATED_ATOM_RESULT_NONFINITE_MAGNETIC_MOMENT_MESSAGE = (
    "Isolated-atom magnetic moment must be finite."
)
GPAW_ISOLATED_ATOM_RESULT_PROVENANCE_ELEMENT_MISMATCH_MESSAGE = (
    "Isolated-atom runtime provenance must cover its element only."
)


@dataclass(frozen=True)
class GpawIsolatedAtomRunResult:
    """Energy-only result produced by one external GPAW atom calculation."""

    element: ElementStr
    energy: float
    magnetic_moment: float
    runtime_provenance: GpawRuntimeProvenance
    eigensolver: GpawEigensolverName

    def __post_init__(self) -> None:  # noqa: D105
        if not isfinite(self.energy):
            raise ValueError(GPAW_ISOLATED_ATOM_RESULT_NONFINITE_ENERGY_MESSAGE)
        if not isfinite(self.magnetic_moment):
            raise ValueError(
                GPAW_ISOLATED_ATOM_RESULT_NONFINITE_MAGNETIC_MOMENT_MESSAGE
            )
        if self.runtime_provenance.elements != frozenset({self.element}):
            raise ValueError(
                GPAW_ISOLATED_ATOM_RESULT_PROVENANCE_ELEMENT_MISMATCH_MESSAGE
            )


@dataclass(frozen=True)
class GpawExternalIsolatedAtomRunFiles:  # noqa: D101
    run_directory: Path

    @property
    def script_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_FILENAME

    @property
    def input_structure_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_ISOLATED_ATOM_INPUT_STRUCTURE_FILENAME

    @property
    def kwargs_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_ISOLATED_ATOM_KWARGS_FILENAME

    @property
    def output_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_ISOLATED_ATOM_OUTPUT_FILENAME

    @property
    def runtime_provenance_module_path(self) -> Path:  # noqa: D102
        return self.run_directory / GPAW_RUNTIME_PROVENANCE_FILENAME


class GpawExternalIsolatedAtomRunner(MSONableBaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    element: ElementStr
    simulation_settings: GpawIsolatedAtomSimulationSettings
    external_runtime_config: GpawExternalInstallPathsConfig
    number_of_processes: PositiveInt = Field(default=DEFAULT_NUMBER_OF_PROCESSES)
    use_ppcg_fallback: bool = False

    @cached_property
    def _ase_atoms(self) -> Atoms:
        vacuum_angstrom = self.simulation_settings.vacuum_angstrom
        cell_length_angstrom = 2 * vacuum_angstrom
        return Atoms(
            symbols=[self.element],
            positions=[(vacuum_angstrom,) * 3],
            cell=(cell_length_angstrom,) * 3,
            pbc=False,
        )

    def _write_runtime_input_files(
        self, run_files: GpawExternalIsolatedAtomRunFiles
    ) -> None:
        write(run_files.input_structure_path.as_posix(), self._ase_atoms)
        if self.use_ppcg_fallback:
            calculator_kwargs = (
                self.simulation_settings.build_gpaw_ppcg_calculator_kwargs()
            )
        else:
            calculator_kwargs = self.simulation_settings.build_gpaw_calculator_kwargs()
        calculator_kwargs["txt"] = GPAW_TEXT_LOG_FILENAME
        run_files.kwargs_path.write_text(
            json.dumps({"calculator_kwargs": calculator_kwargs}),
            encoding=DEFAULT_TEXT_ENCODING,
        )

    def run(self, run_directory: Path) -> GpawIsolatedAtomRunResult:
        """Run the one-atom GPAW calculation in the external runtime."""
        execution_directory = run_directory.resolve()
        run_files = GpawExternalIsolatedAtomRunFiles(run_directory=execution_directory)
        prepare_external_gpaw_script(
            script_path=run_files.script_path,
            script_source_path=_GPAW_EXTERNAL_ISOLATED_ATOM_SCRIPT_SOURCE_PATH,
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
                failure_message=GPAW_EXTERNAL_ISOLATED_ATOM_RUN_FAILED_MESSAGE,
            )
            return self._process_isolated_atom_output(output_path=output_path)
        except (
            FileNotFoundError,
            PermissionError,
            json.JSONDecodeError,
            ValidationError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(GPAW_EXTERNAL_ISOLATED_ATOM_RUN_FAILED_MESSAGE) from exc

    def _process_isolated_atom_output(
        self, *, output_path: Path
    ) -> GpawIsolatedAtomRunResult:
        output_data_object = json.loads(
            output_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
        )
        output_data = GpawExternalIsolatedAtomOutputPayload.model_validate(
            output_data_object
        )
        return GpawIsolatedAtomRunResult(
            element=self.element,
            energy=output_data.energy,
            magnetic_moment=output_data.magnetic_moment,
            runtime_provenance=output_data.runtime_provenance,
            eigensolver=output_data.eigensolver,
        )


def run_gpaw_isolated_atom(  # noqa: D103
    element: ElementStr,
    simulation_settings: GpawIsolatedAtomSimulationSettings,
    run_directory: Path,
    number_of_processes: int = DEFAULT_NUMBER_OF_PROCESSES,
    local_scratch_policy: LocalScratchPolicy | None = None,
    *,
    use_ppcg_fallback: bool = False,
) -> GpawIsolatedAtomRunResult:
    external_runtime_config = (
        GpawExternalInstallPathsConfigLoader.load_and_validate_external_runtime()
    )
    runner = GpawExternalIsolatedAtomRunner(
        element=element,
        simulation_settings=simulation_settings,
        external_runtime_config=external_runtime_config,
        number_of_processes=number_of_processes,
        use_ppcg_fallback=use_ppcg_fallback,
    )
    policy = local_scratch_policy or LocalScratchPolicy()
    with local_scratch_execution_directory(
        policy=policy,
        destination_directory=run_directory,
        result_filenames=GPAW_ISOLATED_ATOM_SCRATCH_RESULT_FILENAMES,
        text_result_filenames_replacing_execution_directory=(GPAW_TEXT_LOG_FILENAME,),
    ) as execution_directory:
        return runner.run(run_directory=execution_directory)
