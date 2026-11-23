from types import SimpleNamespace
from unittest.mock import MagicMock

from crystal.firework_helpers import workflow_reached_terminal_state


def test_workflow_reached_terminal_state_accepts_custom_states() -> None:
    launchpad = MagicMock()
    launchpad.get_wf_by_fw_id.return_value = SimpleNamespace(
        fw_states={1: "COMPLETED", 2: "DEFUSED"}
    )

    assert workflow_reached_terminal_state(
        1, launchpad, accepted_states={"COMPLETED", "DEFUSED"}
    )
