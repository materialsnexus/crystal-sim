from __future__ import annotations

from crystal.schemas.gpaw import GpawRelaxMakerSettings, GpawRelaxSimulationSettings
from crystal.schemas.mlip import MlipRelaxMakerSettings, MlipRelaxSimulationSettings
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.qe import QeConfig, QeRelaxMakerSettings, QeRelaxSimulationSettings
from crystal.schemas.relax_settings import OptimizerType
from crystal.schemas.relaxation_constraints import RelaxationConstraints
from crystal.schemas.settings import ExecutorSettings
from tests.helpers.constants.structure import (
    TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
    TEST_RELAX_MAX_STEPS,
)
from tests.helpers.test_paths import TEST_SI_PSEUDOPOTENTIAL_FILENAME

TEST_SI_ELEMENT_SYMBOL = "Si"


def build_qe_relax_maker_settings_for_test(
    *,
    constraints: RelaxationConstraints | None = None,
    executor_settings: ExecutorSettings | None = None,
) -> QeRelaxMakerSettings:
    return QeRelaxMakerSettings(
        simulation_settings=QeRelaxSimulationSettings(
            config=QeConfig(),
            pseudopotentials={TEST_SI_ELEMENT_SYMBOL: TEST_SI_PSEUDOPOTENTIAL_FILENAME},
            optimizer_type=OptimizerType.BFGS,
            max_steps=TEST_RELAX_MAX_STEPS,
            force_threshold_in_ev_per_angstrom=TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
            constraints=constraints,
        ),
        executor_settings=executor_settings or ExecutorSettings(),
    )


def build_mlip_relax_maker_settings_for_test(
    *, executor_settings: ExecutorSettings | None = None
) -> MlipRelaxMakerSettings:
    return MlipRelaxMakerSettings(
        simulation_settings=MlipRelaxSimulationSettings(
            optimizer_type=OptimizerType.BFGS,
            max_steps=TEST_RELAX_MAX_STEPS,
            force_threshold_in_ev_per_angstrom=TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
            model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe),
        ),
        executor_settings=executor_settings or ExecutorSettings(),
    )


def build_gpaw_relax_maker_settings_for_test(
    *, executor_settings: ExecutorSettings | None = None
) -> GpawRelaxMakerSettings:
    return GpawRelaxMakerSettings(
        simulation_settings=GpawRelaxSimulationSettings(
            optimizer_type=OptimizerType.BFGS,
            max_steps=TEST_RELAX_MAX_STEPS,
            force_threshold_in_ev_per_angstrom=TEST_RELAX_FORCE_THRESHOLD_IN_EV_PER_ANGSTROM,
        ),
        executor_settings=executor_settings or ExecutorSettings(),
    )
