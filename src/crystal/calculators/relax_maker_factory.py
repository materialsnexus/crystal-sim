"""Factory for relax makers across supported backends."""

from __future__ import annotations

from typing import assert_never

from crystal.calculators.gpaw.relax_maker import GpawRelaxMaker
from crystal.calculators.mlip.relax_maker import MlipRelaxMaker
from crystal.calculators.qe.relax_maker import QeRelaxMaker
from crystal.schemas.gpaw import GpawRelaxMakerSettings
from crystal.schemas.mlip import MlipRelaxMakerSettings
from crystal.schemas.qe import QeRelaxMakerSettings
from crystal.schemas.relaxation import RelaxMakerSettings

RelaxMaker = QeRelaxMaker | MlipRelaxMaker | GpawRelaxMaker


def build_relax_maker(*, relax_maker_settings: RelaxMakerSettings) -> RelaxMaker:
    """Build a relax maker for the backend encoded in settings."""
    if isinstance(relax_maker_settings, QeRelaxMakerSettings):
        return QeRelaxMaker(
            simulation_settings=relax_maker_settings.simulation_settings,
            executor_settings=relax_maker_settings.executor_settings,
        )
    if isinstance(relax_maker_settings, MlipRelaxMakerSettings):
        return MlipRelaxMaker(
            simulation_settings=relax_maker_settings.simulation_settings,
            executor_settings=relax_maker_settings.executor_settings,
        )
    if isinstance(relax_maker_settings, GpawRelaxMakerSettings):
        return GpawRelaxMaker(
            simulation_settings=relax_maker_settings.simulation_settings,
            executor_settings=relax_maker_settings.executor_settings,
        )
    assert_never(relax_maker_settings)
