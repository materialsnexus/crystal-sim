"""External GPAW runtime integration-test prerequisites."""

from crystal import CRYSTAL_CONFIG


def assert_external_gpaw_runtime_configuration_is_present() -> None:
    """Assert that the user-local external GPAW paths are configured."""
    assert CRYSTAL_CONFIG.programs.gpaw_env_dir is not None
    assert CRYSTAL_CONFIG.programs.gpaw_setups_dir is not None
