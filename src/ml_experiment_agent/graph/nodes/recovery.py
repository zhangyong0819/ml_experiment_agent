from langgraph.types import interrupt
from ...db.models import Attempt, Run
from ...models.recovery import RecoveryAction
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import runs, snapshot, fail_run


def propose_fixes(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    if deps.recovery_service is not None and not state.get("stop_reason"):
        for run in runs(deps, state):
            attempt = deps.repository.latest_attempt(run.id)
            if run.status in {"FAILED", "TIMED_OUT"} and attempt and attempt.status in {"FAILED", "TIMED_OUT"}:
                deps.recovery_service.propose(attempt.id)
    return {}


def validate_recovery(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    retries, approvals = [], []
    recovery = deps.recovery_service
    if recovery is not None and not state.get("stop_reason") and state["monitor_cycle"] < state["max_monitor_cycles"]:
        for run in runs(deps, state):
            attempt = deps.repository.latest_attempt(run.id)
            if run.status not in {"FAILED", "TIMED_OUT"} or not attempt or attempt.status not in {"FAILED", "TIMED_OUT"}:
                continue
            decision = recovery.validate(attempt.id, resources=deps.options(run.name).resources)
            row = recovery.row(attempt.id)
            approved = row.proposed_patch_json.get("approval") if row else None
            if approved is False:
                continue
            if decision.action == RecoveryAction.AUTO_RETRY or (decision.action == RecoveryAction.REQUIRE_APPROVAL and approved is True):
                retries.append(attempt.id)
            elif decision.action == RecoveryAction.REQUIRE_APPROVAL:
                approvals.append(attempt.id)
    return {"retry_source_attempt_ids": retries, "approval_attempt_ids": approvals}


def approval_gate(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    """Before interrupt: reads only. Resume restarts this node from the beginning."""
    attempt_id = state["approval_attempt_ids"][0]
    recovery = deps.recovery_service
    row = recovery.row(attempt_id)
    approved = row.proposed_patch_json.get("approval")
    if approved is None:
        payload = recovery.approval_payload(attempt_id)
        answer = interrupt(payload)
        # Do not coerce strings, dictionaries, or truthy objects to approval.
        recovery.approve(attempt_id, answer)
        approved = answer
    retries = list(state.get("retry_source_attempt_ids", []))
    if approved is True and attempt_id not in retries:
        retries.append(attempt_id)
    return {"retry_source_attempt_ids": retries, "approval_attempt_ids": state["approval_attempt_ids"][1:]}


def prepare_retry(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    ready = []
    for attempt_id in state.get("retry_source_attempt_ids", []):
        source = deps.repository.get(Attempt, attempt_id)
        run = deps.repository.get(Run, source.run_id)
        try:
            attempt = deps.recovery_service.prepare_retry(attempt_id, resources=deps.options(run.name).resources)
        except (ValueError, OSError) as exc:
            row = deps.recovery_service.row(attempt_id)
            if row is not None:
                deps.recovery_service.stop(row, f"Retry preparation failed: {type(exc).__name__}")
            fail_run(deps, run, "retry_preparation_failed", "Retry configuration could not be prepared", blocked=True)
            continue
        if attempt and attempt.status in {"CREATED", "QUEUED", "RUNNING"}:
            ready.append(run.id)
    return snapshot(deps, state) | {"ready_run_ids": ready, "retry_source_attempt_ids": []}
