from __future__ import annotations

from collections.abc import Callable

import pytest

from crystal.calculators.gpaw.scf_chargemol_maker import GpawScfChargemolMaker
from crystal.calculators.mlip.single_point_maker import MlipSinglePointMaker
from crystal.calculators.qe.scf_pp_chargemol_maker import QeScfPpChargemolMaker
from crystal.calculators.single_point_maker_factory import build_single_point_maker
from crystal.schemas.gpaw import GpawScfChargemolMakerSettings
from crystal.schemas.mlip import MlipSinglePointMakerSettings
from crystal.schemas.qe_pp import QeScfPpChargemolMakerSettings
from tests.helpers.builders.single_point_maker_settings import (
    build_gpaw_single_point_maker_settings_for_test,
    build_mlip_single_point_maker_settings_for_test,
    build_qe_single_point_maker_settings_for_test,
)


@pytest.mark.parametrize(
    ("build_settings", "expected_maker_type"),
    [
        (build_qe_single_point_maker_settings_for_test, QeScfPpChargemolMaker),
        (build_mlip_single_point_maker_settings_for_test, MlipSinglePointMaker),
        (build_gpaw_single_point_maker_settings_for_test, GpawScfChargemolMaker),
    ],
    ids=["qe", "mlip", "gpaw"],
)
def test_build_single_point_maker_returns_supported_backend_maker(
    build_settings: Callable[
        [],
        QeScfPpChargemolMakerSettings
        | MlipSinglePointMakerSettings
        | GpawScfChargemolMakerSettings,
    ],
    expected_maker_type: type[
        QeScfPpChargemolMaker | MlipSinglePointMaker | GpawScfChargemolMaker
    ],
) -> None:
    single_point_maker_settings = build_settings()

    single_point_maker = build_single_point_maker(
        single_point_maker_settings=single_point_maker_settings
    )

    assert isinstance(single_point_maker, expected_maker_type)
    assert single_point_maker.simulation_settings == (
        single_point_maker_settings.simulation_settings
    )
    assert single_point_maker.executor_settings == (
        single_point_maker_settings.executor_settings
    )
