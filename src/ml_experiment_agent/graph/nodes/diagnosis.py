from ...db.models import Diagnosis
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import runs


def diagnose_failed_runs(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    """Sequentially diagnose failed compute attempts; persisted results are replay guards."""
    diagnosed = []
    for run in runs(deps, state):
        attempt = deps.repository.latest_attempt(run.id)
        if attempt is None or attempt.status not in {"FAILED", "TIMED_OUT"}:
            continue
        existing = deps.repository.list(Diagnosis, attempt_id=attempt.id)
        if not existing and deps.diagnosis_agent is not None:
            deps.diagnosis_agent.diagnose(attempt.id)
            existing = deps.repository.list(Diagnosis, attempt_id=attempt.id)
        if existing:
            diagnosed.append(run.id)
    return {"diagnosed_run_ids": diagnosed}
