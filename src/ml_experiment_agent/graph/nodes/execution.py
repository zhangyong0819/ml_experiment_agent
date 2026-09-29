from ...models.enums import ErrorType
from ...models.experiment import RunConfig
from ...models.recovery import RecoveryAction
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import audit_once, fail_run, runs, snapshot


def submit_ready_runs(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    """Reuse DB attempts; never resubmit an active or ambiguously submitted job."""
    if state["monitor_cycle"] >= state["max_monitor_cycles"]:
        remaining = [r.id for r in runs(deps, state) if r.status == "CREATED" and r.id in state.get("ready_run_ids", [])]
        return snapshot(deps, state) | {"ready_run_ids": remaining}
    waiting = []
    for run in runs(deps, state):
        if run.status != "CREATED" or run.id not in state.get("ready_run_ids", []):
            continue
        attempt = deps.repository.latest_attempt(run.id)
        if attempt and (attempt.job_id or attempt.status != "CREATED"):
            # A stale checkpoint may be behind the application commit.
            if attempt.job_id:
                run.status = attempt.status
                deps.repository.commit()
            continue
        if attempt and deps.repository.latest_audit(run.experiment_id, "job_submission_intent",
                                                    run_id=run.id, attempt_id=attempt.id):
            fail_run(deps, run, "submission_unknown", "Submission was started without a committed job_id; manual reconciliation required",
                     ErrorType.UNKNOWN, blocked=True)
            continue
        if deps.repository.active_count() >= deps.policy_engine.config.max_parallel_jobs:
            waiting.append(run.id)
            continue
        options = deps.options(run.name)
        number = attempt.attempt_no if attempt else deps.repository.next_attempt_no(run.id)
        scenario = options.scenario_for(number)
        try:
            if attempt and attempt.attempt_no > 1 and deps.recovery_service is not None:
                authorization = deps.recovery_service.validate_submission(attempt.id, resources=options.resources)
                if authorization.action == RecoveryAction.STOP:
                    fail_run(deps, run, "fix_rejected", authorization.reason, blocked=True)
                    continue
            # Recheck config/policy at the point of use; the Phase 1 service also rechecks.
            job = deps.experiment_service.build_job_spec(run.id,
                config=RunConfig.model_validate(attempt.config_json) if attempt else None,
                scenario=scenario, **options.resources)
            decision = deps.policy_engine.validate(job, active_jobs=deps.repository.active_count(), retries=number - 1)
            if not decision.allowed or decision.requires_approval:
                fail_run(deps, run, "policy_rejected", decision.reason, ErrorType.RESOURCE_ERROR, blocked=True)
                continue
            attempt = attempt or deps.experiment_service.create_attempt(run.id)
            audit_once(deps, run.experiment_id, "job_submission_intent", run_id=run.id, attempt_id=attempt.id)
            deps.experiment_service.submit_attempt(attempt.id, scenario, **options.resources)
        except (ValueError, OSError) as exc:
            # OSError can be an ambiguous transport outcome; never automatically retry it.
            fail_run(deps, run, "submission_failed", str(exc), ErrorType.RESOURCE_ERROR, blocked=isinstance(exc, OSError))
            continue
        audit_once(deps, run.experiment_id, "job_submitted", run_id=run.id, attempt_id=attempt.id,
                   outputs={"job_id": attempt.job_id})
        if attempt.attempt_no > 1:
            audit_once(deps, run.experiment_id, "job_retried", run_id=run.id, attempt_id=attempt.id,
                       outputs={"job_id": attempt.job_id, "attempt_no": attempt.attempt_no})
    return snapshot(deps, state) | {"ready_run_ids": waiting}
