from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import TypeAdapter

from crystal.schemas.backend import SinglePointBackend
from crystal.schemas.gpaw import (
    GpawChargemolTaskDocument,
    GpawScfChargemolMakerSettings,
)
from crystal.schemas.mlip import (
    MlipSinglePointMakerSettings,
    MlipSinglePointTaskDocument,
)
from crystal.schemas.qe import QeScfPpChargemolTaskDocument
from crystal.schemas.qe_pp import QeScfPpChargemolMakerSettings
from crystal.schemas.single_point import (
    SinglePointMakerSettings,
    SinglePointTaskDocument,
)
from tests.helpers.builders.process_run_info import (
    build_successful_process_run_info_for_test,
)
from tests.helpers.builders.properties import build_single_point_properties_for_test
from tests.helpers.builders.single_point_maker_settings import (
    build_gpaw_single_point_maker_settings_for_test,
    build_mlip_single_point_maker_settings_for_test,
    build_qe_single_point_maker_settings_for_test,
)
from tests.helpers.constants.structure import TEST_SI_STRUCTURE


@pytest.mark.parametrize(
    ("build_payload", "expected_settings_type", "expected_backend"),
    [
        (
            lambda: build_qe_single_point_maker_settings_for_test().model_dump(
                mode="python"
            ),
            QeScfPpChargemolMakerSettings,
            SinglePointBackend.qe,
        ),
        (
            lambda: build_mlip_single_point_maker_settings_for_test().model_dump(
                mode="python"
            ),
            MlipSinglePointMakerSettings,
            SinglePointBackend.mlip,
        ),
        (
            lambda: build_gpaw_single_point_maker_settings_for_test().model_dump(
                mode="python"
            ),
            GpawScfChargemolMakerSettings,
            SinglePointBackend.gpaw,
        ),
    ],
    ids=["qe", "mlip", "gpaw"],
)
def test_single_point_maker_settings_union_parses_backend_tagged_payloads(
    build_payload: Callable[[], dict[str, object]],
    expected_settings_type: type[
        QeScfPpChargemolMakerSettings
        | MlipSinglePointMakerSettings
        | GpawScfChargemolMakerSettings
    ],
    expected_backend: SinglePointBackend,
) -> None:
    single_point_settings_adapter: TypeAdapter[SinglePointMakerSettings] = TypeAdapter(
        SinglePointMakerSettings
    )
    validated_single_point_maker_settings = (
        single_point_settings_adapter.validate_python(build_payload())
    )

    assert isinstance(validated_single_point_maker_settings, expected_settings_type)
    assert validated_single_point_maker_settings.backend.value == expected_backend.value


@pytest.mark.parametrize(
    ("build_payload", "expected_task_document_type"),
    [
        (
            lambda: QeScfPpChargemolTaskDocument(
                dir_name="mock",
                errors="",
                properties=build_single_point_properties_for_test(TEST_SI_STRUCTURE),
                structure=TEST_SI_STRUCTURE,
                pp_run_info=build_successful_process_run_info_for_test(
                    command_arguments=["pp.x"]
                ),
                chargemol_run_info=build_successful_process_run_info_for_test(),
            ).model_dump(mode="python"),
            QeScfPpChargemolTaskDocument,
        ),
        (
            lambda: MlipSinglePointTaskDocument(
                properties=build_single_point_properties_for_test(TEST_SI_STRUCTURE),
                errors="",
            ).model_dump(mode="python"),
            MlipSinglePointTaskDocument,
        ),
        (
            lambda: GpawChargemolTaskDocument(
                dir_name="mock",
                errors="",
                properties=build_single_point_properties_for_test(TEST_SI_STRUCTURE),
                structure=TEST_SI_STRUCTURE,
                chargemol_run_info=build_successful_process_run_info_for_test(),
            ).model_dump(mode="python"),
            GpawChargemolTaskDocument,
        ),
    ],
    ids=["qe", "mlip", "gpaw"],
)
def test_single_point_task_document_union_parses_backend_outputs(
    build_payload: Callable[[], dict[str, object]],
    expected_task_document_type: type[
        QeScfPpChargemolTaskDocument
        | MlipSinglePointTaskDocument
        | GpawChargemolTaskDocument
    ],
) -> None:
    single_point_task_document_adapter: TypeAdapter[SinglePointTaskDocument] = (
        TypeAdapter(SinglePointTaskDocument)
    )
    validated_single_point_task_document = (
        single_point_task_document_adapter.validate_python(build_payload())
    )

    assert isinstance(validated_single_point_task_document, expected_task_document_type)
