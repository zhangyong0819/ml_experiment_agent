from ...db.models import Experiment
from ...services.comparison_service import ComparisonService
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import audit_once, snapshot


def finalize_experiment(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    experiment = deps.repository.get(Experiment, state["experiment_id"])
    comparison = ComparisonService(deps.repository).compare(experiment.id)
    if state.get("stop_reason") or comparison.incomplete_runs:
        status = "BLOCKED"
    elif comparison.status == "COMPLETE":
        status = "SUCCEEDED"
    elif comparison.successful_runs:
        status = "COMPLETED_WITH_FAILURES"
    elif comparison.blocked_runs:
        status = "BLOCKED"
    else:
        status = "FAILED"
    experiment.status = status
    workflow_status = "STOPPED" if state.get("stop_reason") else status
    audit_once(deps, experiment.id, "experiment_finalized", outputs={"status": status, "workflow_status": workflow_status})
    return snapshot(deps, state) | {"workflow_status": workflow_status,
                                   "newly_succeeded_run_ids": [], "newly_failed_run_ids": []}
