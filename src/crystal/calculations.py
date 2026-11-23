# noqa: D100

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobflow.core.flow import Flow
from jobflow.core.job import Job, Response, job
from jobflow.core.maker import Maker
from pydantic import BaseModel

from crystal.constants.calculations import (
    ADDITION_OUTPUT_FILENAME,
    ADDITION_RUN_DIRNAME,
    CUBE_OUTPUT_FILENAME,
    CUBE_RUN_DIRNAME,
)


class FileSchema(BaseModel):  # noqa: D101
    data: str
    path: str
    filename: str


def addition(a: int, b: int) -> int:  # noqa: D103
    return a + b


class ExternalExecutableRunMaker(Maker):  # noqa: D101
    name: str = "external_executable_run_maker"

    def make(self, ntasks: int) -> Job:  # noqa: D102
        return run_mpi_in_allocation(ntasks)


@job
def run_mpi_in_allocation(ntasks: int = 4) -> dict[str, Any]:  # noqa: D103
    cmd = f"srun -n {ntasks} bash -lc 'echo rank=$SLURM_PROCID host=$(hostname)'"
    r = subprocess.run(
        ["bash", "-lc", cmd],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "cmd": cmd,
    }


@dataclass
class AdditionMaker(Maker):  # noqa: D101
    name: str = "add"

    @job
    def make(self, a: int, b: int) -> dict[str, Any]:  # noqa: D102
        run_directory = Path.cwd() / ADDITION_RUN_DIRNAME
        run_directory.mkdir(parents=True, exist_ok=True)

        file = run_directory / ADDITION_OUTPUT_FILENAME

        Path.write_text(file, str(a + b))
        sum_of_integers = int(Path.read_text(file))
        return {"sum": sum_of_integers, "filepath": file.as_posix()}

    def make_two_addition_jobs(self, a: int, b: int) -> Flow:  # noqa: D102
        first_job = self.make(a=a, b=b)
        second_job = self.make(a=first_job.output["sum"], b=first_job.output["sum"])
        return Flow(jobs=[first_job, second_job], output=second_job.output)


@dataclass
class CubeMaker(Maker):  # noqa: D101
    name: str = "cube"

    @property
    def dedupe_property(self) -> str:  # noqa: D102
        return ""

    @job
    def make(self, a: int) -> Any:  # noqa: D102
        run_directory = Path.cwd() / CUBE_RUN_DIRNAME
        run_directory.mkdir(parents=True, exist_ok=True)

        file = run_directory / CUBE_OUTPUT_FILENAME

        Path.write_text(file, str(a * a * a))
        return int(Path.read_text(file))

    def make_two_cube_jobs(self, a: int) -> Flow:  # noqa: D102
        cube_job = self.make(a=a)
        second_cube_job = self.make(a=cube_job.output)
        return Flow(jobs=[cube_job, second_cube_job], output=second_cube_job.output)


@dataclass
class SubtractMaker(Maker):  # noqa: D101
    name: str = "subtract"

    @job
    def make(self, c: int, d: int) -> int:  # noqa: D102
        return c - d


class AddThenCube(Maker):  # noqa: D101
    name: str = "add_then_cube"

    def make(self, number_1: int, number_2: int) -> Flow:  # noqa: D102
        add_maker = AdditionMaker()
        add_job = add_maker.make(number_1, number_2)

        cube_maker = CubeMaker()
        cube_job = cube_maker.make(add_job.output["sum"])
        return Flow(jobs=[add_job, cube_job], output=cube_job.output)


class AddThenFilteredCubeMaker(Maker):  # noqa: D101
    name: str = "add_then_filter_and_cube"

    def make(  # noqa: D102
        self, number_1: int, number_2: int, filtering_threshold: int
    ) -> Flow:
        add_job = AdditionMaker().make(number_1, number_2)
        cube_jobs = filtered_cube_jobs(add_job.output["sum"], filtering_threshold)

        return Flow(jobs=[add_job, cube_jobs], output="Success")


@job
def filtered_cube_jobs(add_job_output: int, threshold: int) -> Response[Flow]:  # noqa: D103
    cube_jobs = []
    for i in range(add_job_output + 1):
        if i > threshold:
            cube_jobs.append(CubeMaker().make(i))  # noqa: PERF401
    return Response(replace=Flow(cube_jobs, output=[j.output for j in cube_jobs]))


class AddAndSaveFileMaker(Maker):  # noqa: D101
    name: str = "add_then_save_to_file"

    def make(self, a: int, b: int) -> Flow:  # noqa: D102
        add_maker = AdditionMaker().make(a, b)
        save_job = save_file(add_maker.output["filepath"])
        return Flow(jobs=[add_maker, save_job], output=save_job.output)


@job(output_schema=FileSchema, files=True)
def save_file(filepath: str) -> FileSchema:  # noqa: D103
    file = Path(filepath)
    data = file.read_bytes()
    return FileSchema(data=data, path=file.as_posix(), filename=file.name)
