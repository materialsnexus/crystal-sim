from types import SimpleNamespace
from unittest.mock import Mock

from crystal.schemas.mlip import MlipRelaxSimulationSettings
from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.schemas.relax_settings import OptimizerType

MOCK_TORCH = SimpleNamespace(
    cuda=SimpleNamespace(
        is_available=Mock(return_value=True),
        current_device=Mock(),
        get_device_name=Mock(),
    ),
    set_default_device=Mock(),
)

TEST_MLIP_RELAX_SIMULATION_SETTINGS = MlipRelaxSimulationSettings(
    optimizer_type=OptimizerType.BFGS,
    max_steps=2,
    force_threshold_in_ev_per_angstrom=0.0,
    model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe),
)
