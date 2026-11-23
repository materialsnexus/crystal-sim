# noqa: D100

from collections.abc import Collection

from fireworks.core.fworker import FWorker
from fireworks.core.launchpad import LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from fireworks.queue.queue_launcher import rapidfire as queue_rapidfire
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter


def workflow_reached_terminal_state(  # noqa: D103
    firework_id: int,
    launchpad: LaunchPad,
    *,
    accepted_states: Collection[str] = ("COMPLETED", "FIZZLED"),
) -> bool:
    workflow = launchpad.get_wf_by_fw_id(firework_id)
    states = workflow.fw_states
    return all(state in accepted_states for state in states.values())


def run_queue_launcher(  # noqa: D103
    launchpad: LaunchPad, fireworker: FWorker, qadapter: CommonAdapter
) -> None:
    queue_rapidfire(
        launchpad=launchpad,
        fworker=fireworker,
        qadapter=qadapter,
        nlaunches=0,
        reserve=True,
    )


def run_rapidfire_launcher(  # noqa: D103
    launchpad: LaunchPad, fireworker: FWorker, sleep_time: int = 1
) -> None:
    rapidfire(launchpad=launchpad, fworker=fireworker, sleep_time=sleep_time)
