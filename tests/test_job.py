import random

from jobflow.core.job import job

from crystal.helpers.job import canonicalize_call, compute_dedupe
from crystal.schemas.serialization import MSONableBaseModel


@job
def standard_addition(numbers: list[int]) -> int:
    return sum(numbers)


def test_duplication_in_unbounded_job() -> None:
    numbers_to_add = [1, 2]
    job = standard_addition(numbers=numbers_to_add)
    duplicate_job = standard_addition(numbers=numbers_to_add)
    assert compute_dedupe(job, {}) == compute_dedupe(duplicate_job, {})


class ExampleExpression(MSONableBaseModel):
    subtraction: str | None = None
    multiply: str | None = None
    divide: str | None = None
    addition: str | None = None


@job
def get_all_expressions(
    first_example: ExampleExpression, second_example: ExampleExpression
) -> dict[str, str]:
    return first_example.as_dict() | second_example.as_dict()


def test_deterministic_dedupe() -> None:
    first_example = {"subtraction": "sub", "addition": "add"}
    second_example = {"addition": "add", "multiply": "mul", "divide": "div"}
    job = get_all_expressions(
        first_example=first_example, second_example=second_example
    )

    shuffled_second_example = dict(
        random.sample(list(second_example.items()), len(second_example))
    )
    duplicate_job = get_all_expressions(
        second_example=shuffled_second_example, first_example=first_example
    )
    assert compute_dedupe(job) == compute_dedupe(duplicate_job)


def test_canonicalize() -> None:
    numbers_to_add = [1, 2]
    job_with_args = standard_addition(numbers_to_add)
    job_with_kwargs = standard_addition(numbers=numbers_to_add)

    assert job_with_args.function_args[0] == numbers_to_add
    assert getattr(job_with_args, "function_kwargs", {}) == {}

    assert job_with_kwargs.function_args == ()
    assert getattr(job_with_kwargs, "function_kwargs", {}) == {
        "numbers": numbers_to_add
    }

    kwargs_for_job_with_args = canonicalize_call(
        job_with_args.function,
        job_with_args.function_args,
        job_with_args.function_kwargs,
    )

    kwargs_for_job_with_kwargs = canonicalize_call(
        job_with_kwargs.function,
        job_with_kwargs.function_args,
        job_with_kwargs.function_kwargs,
    )
    assert kwargs_for_job_with_args == kwargs_for_job_with_kwargs
