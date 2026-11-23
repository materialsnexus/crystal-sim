import pytest

from crystal.schemas.base import BaseTaskDocument
from crystal.schemas.properties import StructureGenerationProperties
from tests.helpers.constants.structure import TEST_SI_STRUCTURE


@pytest.mark.parametrize(
    ("properties", "errors", "raises", "error_message"),
    [
        pytest.param(
            None,
            "",
            True,
            "Task document must include either properties or a non-empty error message.",
            id="invalid-no-properties-no-errors",
        ),
        pytest.param(
            None,
            "   ",
            True,
            "Error message cannot be whitespace only.",
            id="invalid-whitespace-only-errors",
        ),
        pytest.param(
            StructureGenerationProperties(structures=[TEST_SI_STRUCTURE]),
            "upstream failed",
            True,
            "Task document must include either properties or a non-empty error message.",
            id="invalid-both-properties-and-errors",
        ),
        pytest.param(
            StructureGenerationProperties(structures=[TEST_SI_STRUCTURE]),
            "",
            False,
            None,
            id="valid-properties-no-errors",
        ),
        pytest.param(
            None, "upstream failed", False, None, id="valid-errors-no-properties"
        ),
    ],
)
def test_base_task_document_validation(
    properties: object | None, errors: str, *, raises: bool, error_message: str | None
) -> None:
    if raises:
        with pytest.raises(ValueError, match=error_message):
            _ = BaseTaskDocument(properties=properties, errors=errors)
        return
    task_doc = BaseTaskDocument(properties=properties, errors=errors)
    assert task_doc.properties == properties
    assert task_doc.errors == errors
