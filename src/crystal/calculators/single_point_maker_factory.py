"""Factory for single-point makers across supported backends."""

from __future__ import annotations

from typing import assert_never

from crystal.calculators.gpaw.scf_chargemol_maker import GpawScfChargemolMaker
from crystal.calculators.mlip.single_point_maker import MlipSinglePointMaker
from crystal.calculators.qe.scf_pp_chargemol_maker import QeScfPpChargemolMaker
from crystal.schemas.gpaw import GpawScfChargemolMakerSettings
from crystal.schemas.mlip import MlipSinglePointMakerSettings
from crystal.schemas.qe_pp import QeScfPpChargemolMakerSettings
from crystal.schemas.single_point import SinglePointMakerSettings

SinglePointMaker = QeScfPpChargemolMaker | MlipSinglePointMaker | GpawScfChargemolMaker


def build_single_point_maker(
    *, single_point_maker_settings: SinglePointMakerSettings
) -> SinglePointMaker:
    """Build a same-job single-point-and-charges maker for the backend."""
    if isinstance(single_point_maker_settings, QeScfPpChargemolMakerSettings):
        return QeScfPpChargemolMaker(
            simulation_settings=single_point_maker_settings.simulation_settings,
            executor_settings=single_point_maker_settings.executor_settings,
        )
    if isinstance(single_point_maker_settings, MlipSinglePointMakerSettings):
        return MlipSinglePointMaker(
            simulation_settings=single_point_maker_settings.simulation_settings,
            executor_settings=single_point_maker_settings.executor_settings,
        )
    if isinstance(single_point_maker_settings, GpawScfChargemolMakerSettings):
        return GpawScfChargemolMaker(
            simulation_settings=single_point_maker_settings.simulation_settings,
            executor_settings=single_point_maker_settings.executor_settings,
        )
    assert_never(single_point_maker_settings)
