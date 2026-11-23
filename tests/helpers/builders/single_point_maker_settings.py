from __future__ import annotations

from crystal.schemas.gpaw import (
    GpawChargemolSimulationSettings,
    GpawScfChargemolMakerSettings,
    GpawScfChargemolSimulationSettings,
    GpawScfSimulationSettings,
)
from crystal.schemas.mlip import (
    MlipSinglePointMakerSettings,
    MlipSinglePointSimulationSettings,
)
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.qe import QeConfig, QeScfSimulationSettings
from crystal.schemas.qe_pp import (
    QeScfPpChargemolMakerSettings,
    QeScfPpChargemolSimulationSettings,
)
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.test_paths import TEST_SI_PSEUDOPOTENTIAL_FILENAME

TEST_SI_ELEMENT_SYMBOL = "Si"


def build_qe_single_point_maker_settings_for_test(
    *,
    qe_scf_simulation_settings: QeScfSimulationSettings | None = None,
    executor_settings: ExecutorSettings | None = None,
) -> QeScfPpChargemolMakerSettings:
    return QeScfPpChargemolMakerSettings(
        simulation_settings=QeScfPpChargemolSimulationSettings(
            qe_scf_simulation_settings=qe_scf_simulation_settings
            or QeScfSimulationSettings(
                config=QeConfig(),
                pseudopotentials={
                    TEST_SI_ELEMENT_SYMBOL: TEST_SI_PSEUDOPOTENTIAL_FILENAME
                },
            )
        ),
        executor_settings=executor_settings or ExecutorSettings(),
    )


def build_mlip_single_point_maker_settings_for_test(
    *,
    mlip_single_point_simulation_settings: MlipSinglePointSimulationSettings
    | None = None,
    executor_settings: ExecutorSettings | None = None,
) -> MlipSinglePointMakerSettings:
    return MlipSinglePointMakerSettings(
        simulation_settings=mlip_single_point_simulation_settings
        or MlipSinglePointSimulationSettings(
            model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe)
        ),
        executor_settings=executor_settings or ExecutorSettings(),
    )


def build_gpaw_single_point_maker_settings_for_test(
    *,
    gpaw_scf_simulation_settings: GpawScfSimulationSettings | None = None,
    gpaw_chargemol_simulation_settings: GpawChargemolSimulationSettings | None = None,
    executor_settings: ExecutorSettings | None = None,
) -> GpawScfChargemolMakerSettings:
    return GpawScfChargemolMakerSettings(
        simulation_settings=GpawScfChargemolSimulationSettings(
            gpaw_scf_simulation_settings=gpaw_scf_simulation_settings
            or GpawScfSimulationSettings(),
            gpaw_chargemol_simulation_settings=gpaw_chargemol_simulation_settings
            or GpawChargemolSimulationSettings(),
        ),
        executor_settings=executor_settings or ExecutorSettings(),
    )
