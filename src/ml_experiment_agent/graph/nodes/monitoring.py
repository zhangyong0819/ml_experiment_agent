from ...models.enums import ErrorType
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import ACTIVE, FAILED, audit_once, fail_run, runs, snapshot, require_attempt


def monitor_runs(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    current = snapshot(deps, state)
    cycle = state["monitor_cycle"]
    if not current["active_run_ids"] and not state.get("ready_run_ids"):
        return current | {"newly_succeeded_run_ids": [], "newly_failed_run_ids": []}
    if cycle >= state["max_monitor_cycles"]:
        reason = f"monitor_cycle limit reached ({state['max_monitor_cycles']}); outstanding jobs require reconciliation"
        audit_once(deps, state["experiment_id"], "monitor_limit_reached", outputs={"reason": reason}, outcome="blocked")
        return current | {"stop_reason": reason, "workflow_status": "STOPPED"}
    succeeded, failed = [], []
    for run in runs(deps, state):
        if run.id not in current["active_run_ids"]:
            continue
        attempt = require_attempt(deps, run)
        before = attempt.status
        try:
            deps.experiment_service.refresh_attempt_status(attempt.id)
        except KeyError as exc:
            # The in-memory mock cannot resurrect a job after process restart.
            fail_run(deps, run, "monitor_unavailable", str(exc), ErrorType.UNKNOWN, blocked=True)
            continue
        except OSError as exc:
            fail_run(deps, run, "monitor_unavailable", str(exc), ErrorType.RESOURCE_ERROR, blocked=True)
            continue
        if attempt.status != before:
            audit_once(deps, run.experiment_id, "job_status_changed", run_id=run.id, attempt_id=attempt.id,
                       outputs={"before": before, "after": attempt.status})
        if attempt.status == "SUCCEEDED":
            succeeded.append(run.id)
        elif attempt.status in FAILED:
            failed.append(run.id)
        elif attempt.status not in ACTIVE:
            raise RuntimeError(f"Unexpected backend transition: {before} -> {attempt.status}")
    return snapshot(deps, state) | {"monitor_cycle": cycle + 1,
                                   "newly_succeeded_run_ids": succeeded, "newly_failed_run_ids": failed}


def handle_failed_runs(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    """Record execution failures; optional diagnosis and recovery live downstream."""
    for run in runs(deps, state):
        if run.status not in FAILED:
            continue
        attempt = deps.repository.latest_attempt(run.id)
        if attempt is None or not attempt.job_id or attempt.status not in FAILED:
            continue
        if deps.repository.latest_audit(run.experiment_id, "job_failed", run_id=run.id, attempt_id=attempt.id):
            continue
        try:
            deps.experiment_service.collect_logs(attempt.id)
        except (KeyError, OSError, ValueError) as exc:
            audit_once(deps, run.experiment_id, "logs_unavailable", run_id=run.id, attempt_id=attempt.id,
                       outputs={"reason": str(exc)[:2000]}, outcome="failed")
        audit_once(deps, run.experiment_id, "job_failed", run_id=run.id, attempt_id=attempt.id,
                   outputs={"reason": attempt.error_message or "Run failed", "error_type": attempt.error_type}, outcome="failed")
    return snapshot(deps, state)
