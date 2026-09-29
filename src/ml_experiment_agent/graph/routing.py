from typing import Literal
from .state import ExperimentGraphState


def route_after_monitor(state: ExperimentGraphState) -> Literal["monitor", "submit", "finish"]:
    if state.get("stop_reason"):
        return "finish"
    if state.get("active_run_ids"):
        return "monitor"
    if state.get("ready_run_ids"):
        # The monitor node records the bound before another submission wave is allowed.
        return "monitor" if state["monitor_cycle"] >= state["max_monitor_cycles"] else "submit"
    return "finish"


def route_recovery(state: ExperimentGraphState) -> Literal["retry", "approval", "finish"]:
    if state.get("retry_source_attempt_ids"):
        return "retry"
    if state.get("approval_attempt_ids"):
        return "approval"
    return "finish"
