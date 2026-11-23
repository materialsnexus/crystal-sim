from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import TypeAdapter

from crystal.schemas.backend import RelaxBackend
from crystal.schemas.gpaw import GpawRelaxMakerSettings
from crystal.schemas.mlip import MlipRelaxMakerSettings
from crystal.schemas.qe import QeRelaxMakerSettings
from crystal.schemas.relaxation import RelaxMakerSettings
from tests.helpers.builders.relax_maker_settings import (
    build_gpaw_relax_maker_settings_for_test,
    build_mlip_relax_maker_settings_for_test,
    build_qe_relax_maker_settings_for_test,
)


@pytest.mark.parametrize(
    ("build_payload", "expected_settings_type", "expected_backend"),
    [
        (
            lambda: build_qe_relax_maker_settings_for_test().model_dump(mode="python"),
            QeRelaxMakerSettings,
            RelaxBackend.qe,
        ),
        (
            lambda: build_mlip_relax_maker_settings_for_test().model_dump(
                mode="python"
            ),
            MlipRelaxMakerSettings,
            RelaxBackend.mlip,
        ),
        (
            lambda: build_gpaw_relax_maker_settings_for_test().model_dump(
                mode="python"
            ),
            GpawRelaxMakerSettings,
            RelaxBackend.gpaw,
        ),
    ],
    ids=["qe", "mlip", "gpaw"],
)
def test_relax_maker_settings_union_parses_backend_tagged_payloads(
    build_payload: Callable[[], dict[str, object]],
    expected_settings_type: type[
        QeRelaxMakerSettings | MlipRelaxMakerSettings | GpawRelaxMakerSettings
    ],
    expected_backend: RelaxBackend,
) -> None:
    relax_settings_adapter: TypeAdapter[RelaxMakerSettings] = TypeAdapter(
        RelaxMakerSettings
    )
    validated_relax_maker_settings = relax_settings_adapter.validate_python(
        build_payload()
    )

    assert isinstance(validated_relax_maker_settings, expected_settings_type)
    assert validated_relax_maker_settings.backend is expected_backend
