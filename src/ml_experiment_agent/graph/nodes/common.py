from typing import Any
from ...db.models import Run, Attempt
from ...models.enums import ErrorType
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState

ACTIVE = {"QUEUED", "RUNNING"}
FAILED = {"FAILED", "TIMED_OUT", "CANCELLED"}


def runs(deps: WorkflowDependencies, state: ExperimentGraphState) -> list[Run]:
    return sorted(deps.repository.list(Run, experiment_id=state["experiment_id"]), key=lambda r: r.name)


def audit_once(deps: WorkflowDependencies, experiment_id: str, action: str, *, run_id: str | None = None,
               attempt_id: str | None = None, outputs: dict[str, Any] | None = None, outcome: str = "success") -> None:
    repo = deps.repository
    previous = repo.latest_audit(experiment_id, action, run_id=run_id, attempt_id=attempt_id)
    if previous is None or previous.output_json != (outputs or {}) or previous.outcome != outcome:
        repo.audit(experiment_id, action, run_id=run_id, attempt_id=attempt_id, outputs=outputs, outcome=outcome)
    repo.commit()


def fail_run(deps: WorkflowDependencies, run: Run, action: str, reason: str,
             error_type: ErrorType = ErrorType.UNKNOWN, *, blocked: bool = False) -> None:
    """Run outcome may fail after successful compute; retain the actual Attempt job status."""
    run.status = "BLOCKED" if blocked else "FAILED"
    attempt = deps.repository.latest_attempt(run.id)
    audit_once(deps, run.experiment_id, action, run_id=run.id, attempt_id=attempt.id if attempt else None,
               outcome="blocked" if blocked else "failed", outputs={"reason": reason[:2000], "error_type": error_type.value})


def snapshot(deps: WorkflowDependencies, state: ExperimentGraphState) -> ExperimentGraphState:
    result: ExperimentGraphState = {"pending_run_ids": [], "active_run_ids": [], "succeeded_run_ids": [],
                                    "failed_run_ids": [], "blocked_run_ids": []}
    for run in runs(deps, state):
        attempt = deps.repository.latest_attempt(run.id)
        if run.status == "BLOCKED":
            result["blocked_run_ids"].append(run.id)
        elif run.status in FAILED:
            result["failed_run_ids"].append(run.id)
        elif run.status == "SUCCEEDED":
            result["succeeded_run_ids"].append(run.id)
        elif attempt and attempt.job_id and attempt.status in ACTIVE:
            result["active_run_ids"].append(run.id)
        elif run.status == "CREATED":
            result["pending_run_ids"].append(run.id)
        else:
            raise RuntimeError(f"Inconsistent run/attempt lifecycle: {run.id}")
    return result


def require_attempt(deps: WorkflowDependencies, run: Run) -> Attempt:
    attempt = deps.repository.latest_attempt(run.id)
    if attempt is None:
        raise RuntimeError(f"Run {run.id} has no required attempt")
    return attempt
