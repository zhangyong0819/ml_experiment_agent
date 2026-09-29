from ...models.enums import ErrorType, Scenario
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import audit_once, fail_run, runs, snapshot


def validate_configs(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    ready = []
    for run in runs(deps, state):
        if run.status != "CREATED":
            continue
        try:
            deps.experiment_service.validate_run_config(run.id)
            if deps.options(run.name).scenario == Scenario.INVALID_CONFIG:
                raise ValueError("Reserved INVALID_CONFIG fixture rejected")
        except (ValueError, OSError) as exc:
            fail_run(deps, run, "config_rejected", str(exc), ErrorType.CONFIG_ERROR)
            continue
        ready.append(run.id)
        audit_once(deps, run.experiment_id, "config_validated", run_id=run.id)
    return snapshot(deps, state) | {"ready_run_ids": ready}


def policy_check(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    ready = []
    for run in runs(deps, state):
        if run.status != "CREATED" or run.id not in state.get("ready_run_ids", []):
            continue
        options = deps.options(run.name)
        try:
            job = deps.experiment_service.build_job_spec(run.id, scenario=options.scenario, **options.resources)
            decision = deps.policy_engine.validate(job)
            if not decision.allowed or decision.requires_approval:
                fail_run(deps, run, "policy_rejected", decision.reason, ErrorType.RESOURCE_ERROR, blocked=True)
                continue
            if options.scenario == Scenario.POLICY_VIOLATION:
                raise ValueError("Reserved POLICY_VIOLATION fixture rejected")
        except (ValueError, OSError) as exc:
            fail_run(deps, run, "policy_rejected", str(exc), ErrorType.RESOURCE_ERROR, blocked=True)
            continue
        ready.append(run.id)
        audit_once(deps, run.experiment_id, "policy_approved", run_id=run.id)
    return snapshot(deps, state) | {"ready_run_ids": ready}
